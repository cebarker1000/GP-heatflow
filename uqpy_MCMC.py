import yaml
import argparse
from analysis.config_utils import create_uqpy_distributions, get_param_defs_from_config
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.distributions.collection.Normal import Normal
from UQpy.inference.inference_models.LogLikelihoodModel import LogLikelihoodModel
from UQpy.sampling.mcmc.MetropolisHastings import MetropolisHastings
from UQpy.inference.BayesParameterEstimation import BayesParameterEstimation
from scipy.special import logsumexp
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# grab surrogate model
from uqpy_surrogate import timeseries_model

logging.getLogger("UQpy").setLevel(logging.DEBUG)

# get param dists
param_defs = get_param_defs_from_config(config_path="configs/distributions.yaml")
uqpy_dists = create_uqpy_distributions(param_defs)
non_k_prior = JointIndependent(marginals=uqpy_dists[:8])
k_prior = JointIndependent(marginals=uqpy_dists[8:])

# Load experimental data
def load_experimental_data():
    """Load and preprocess experimental data"""
    data = pd.read_csv("data/experimental/geballe_heat_data.csv")
    # Use the 'oside' column and normalize it like the surrogate expects
    oside_data = data['oside'].values
    # Normalize: (oside - oside[0]) / (temp_max - temp[0])
    y_obs = (oside_data - oside_data[0]) / (data['temp'].max() - data['temp'].iloc[0])
    
    # Get experimental time grid
    exp_time = data['time'].values
    
    return y_obs, exp_time

def interpolate_to_surrogate_grid(exp_data, exp_time):
    """
    Interpolate experimental data to the surrogate model's time grid.
    
    Parameters:
    -----------
    exp_data : array-like
        Experimental data (normalized oside)
    exp_time : array-like
        Experimental time grid
        
    Returns:
    --------
    array-like
        Interpolated data on surrogate time grid
    """
    from scipy.interpolate import interp1d
    
    # Get surrogate time grid (from the surrogate model)
    # The surrogate uses 50 time points from 0 to 7.5e-6 seconds
    sim_t_final = 7.5e-6  # seconds
    sim_num_steps = 50
    surrogate_time_grid = np.linspace(0, sim_t_final, sim_num_steps)
    
    # Create interpolation function
    interp_func = interp1d(exp_time, exp_data, kind='linear', 
                           bounds_error=False, fill_value=exp_data[-1])
    
    # Interpolate to surrogate grid
    interpolated_data = interp_func(surrogate_time_grid)
    
    print(f"Interpolated experimental data:")
    print(f"  Original: {len(exp_data)} points, time range: [{exp_time[0]:.2e}, {exp_time[-1]:.2e}] s")
    print(f"  Surrogate: {len(surrogate_time_grid)} points, time range: [{surrogate_time_grid[0]:.2e}, {surrogate_time_grid[-1]:.2e}] s")
    print(f"  Interpolated data range: [{interpolated_data.min():.4f}, {interpolated_data.max():.4f}]")
    
    return interpolated_data

# Global constant for sensor variance (as requested for testing)
SENSOR_VARIANCE = 0.1  # You can adjust this value based on your sensor characteristics

# ------------------------------------------------------------
# PSEUDO-MARGINAL MONTE-CARLO LIKELIHOOD WITH NUISANCE DRAWS
# ------------------------------------------------------------
M_MC = 1                        # φ draws per κ proposal (tune for variance)
RNG   = np.random.default_rng(42)
PHI_STORE = []                   # stores φ batches in evaluation order


def _gaussian_loglike(y_pred: np.ndarray, y_obs: np.ndarray, *, sigma2: float) -> np.ndarray:
    """Vectorised Gaussian log-likelihood for independent observations."""
    resid = y_pred - y_obs              # (m, T)
    return -0.5 * np.sum(resid ** 2 / sigma2 + np.log(2 * np.pi * sigma2), axis=1)


CALLS = 0
def log_likelihood_mc(params=None, data=None):
    """Monte-Carlo log-likelihood   log L(κ) = log(1/M ∑_j exp ℓ(κ, φ_j))."""
    global CALLS
    params = np.atleast_2d(params)      # (n, 3)
    n, _ = params.shape
    log_L = np.empty(n)

    for i in range(n):
        # 1) draw φ ~ p(φ)
        phi_batch = non_k_prior.rvs(nsamples=M_MC)  # (M, 8)

        # 2) build full parameter matrix [φ | κ]
        params_full = np.hstack([phi_batch, np.tile(params[i], (M_MC, 1))])  # (M, 11)

        # 3) surrogate prediction
        y_pred = timeseries_model(params_full)  # (M, T)

        # 4) inner log-likelihoods and MC average in log-space
        ll_batch = _gaussian_loglike(y_pred, data, sigma2=SENSOR_VARIANCE)
        log_L[i] = logsumexp(ll_batch) - np.log(M_MC)

        # 5) stash φ draws for post-analysis
        PHI_STORE.append(phi_batch)
    CALLS += params.shape[0]
    if CALLS % 1000 == 0:                  # every 1 000 proposals
        print(f"{CALLS:,} proposals evaluated")
    return log_L


# ------------------------------------------------------------
# MAIN DRIVER – builds InferenceModel, Sampler, runs BPE
# ------------------------------------------------------------

def main():
    # load & align experimental data
    y_obs, exp_time = load_experimental_data()
    y_obs_interp = interpolate_to_surrogate_grid(y_obs, exp_time)  # shape (50,)

    # Inference model: κ-only, custom MC likelihood, with prior
    ll_model = LogLikelihoodModel(n_parameters=3, log_likelihood=log_likelihood_mc)
    ll_model.prior = k_prior

    # Proposal distribution (tune scales as needed)
    proposal = JointIndependent(marginals=[
        Normal(scale=0.2),   # k_sample
        Normal(scale=0.5),   # k_ins
        Normal(scale=10.0)   # k_coupler
    ])

    # Metropolis-Hastings sampler
    mh_sampler = MetropolisHastings(
        jump=5,                      # one proposal per chain step
        burn_length=5000,
        proposal=proposal,
        seed=[3.8, 10.0, 350.0],     # reasonable starting values
        save_log_pdf=True,
        random_state=123
    )

    # Run Bayesian estimation
    bpe = BayesParameterEstimation(
        inference_model=ll_model,
        data=y_obs_interp,
        sampling_class=mh_sampler,
        nsamples=10000
    )

    samples_kappa = bpe.sampler.samples               # (N, 3)
    # φ draws collected during every log-pdf evaluation (one batch per chain step)
    phi_chain = np.vstack(PHI_STORE)
    accepted_mask = np.isfinite(bpe.sampler.log_pdf_values)

    print("\nMCMC complete!  Accepted samples:", samples_kappa.shape[0])
    print("k_sample mean ± σ:", samples_kappa[:, 0].mean(), samples_kappa[:, 0].std())
    print("k_ins    mean ± σ:", samples_kappa[:, 1].mean(), samples_kappa[:, 1].std())
    print("k_coupler mean ± σ:", samples_kappa[:, 2].mean(), samples_kappa[:, 2].std())

    # Optional: quick scatter
    plt.figure(figsize=(6,4))
    plt.scatter(samples_kappa[:, 0], samples_kappa[:, 1], s=2, alpha=0.4)
    plt.xlabel("k_sample")
    plt.ylabel("k_ins")
    plt.title("Posterior samples (projection)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

