import yaml
import argparse
from analysis.config_utils import create_uqpy_distributions, get_param_defs_from_config
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sampling.mcmc.Stretch import Stretch
from UQpy.inference.inference_models.LogLikelihoodModel import LogLikelihoodModel
from UQpy.sampling.mcmc.MetropolisHastings import MetropolisHastings
from UQpy.sampling.mcmc.DREAM import DREAM
from UQpy.sampling.mcmc.DRAM import DRAM
from UQpy.inference.BayesParameterEstimation import BayesParameterEstimation
from scipy.special import logsumexp
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# grab surrogate model
from uqpy_surrogate import timeseries_model

logging.getLogger("UQpy").setLevel(logging.DEBUG)

param_defs = get_param_defs_from_config(config_path="configs/distributions.yaml")
uqpy_dists = create_uqpy_distributions(param_defs)
non_k_prior = JointIndependent(marginals=uqpy_dists[:8])
k_prior = JointIndependent(marginals=uqpy_dists[8:])

# Load experimental data
def load_experimental_data():
    data = pd.read_csv("data/experimental/geballe_heat_data.csv")
    oside_data = data['oside'].values
    y_obs = (oside_data - oside_data[0]) / (data['temp'].max() - data['temp'].iloc[0])
    exp_time = data['time'].values
    return y_obs, exp_time

def interpolate_to_surrogate_grid(exp_data, exp_time):
    from scipy.interpolate import interp1d
    sim_t_final = 7.5e-6  # seconds
    sim_num_steps = 50
    surrogate_time_grid = np.linspace(0, sim_t_final, sim_num_steps)
    interp_func = interp1d(exp_time, exp_data, kind='linear', 
                           bounds_error=False, fill_value=(exp_data[0], exp_data[-1]))
    interpolated_data = interp_func(surrogate_time_grid)
    return interpolated_data

SENSOR_VARIANCE = 1e-4  # You can adjust this value based on your sensor characteristics

M_MC = 200                       # total phi draws
PHI_STORE = []                   # stores phi draws

def _gaussian_loglike(y_pred: np.ndarray, y_obs: np.ndarray, *, sigma2: float) -> np.ndarray:
    resid = y_pred - y_obs              # (m, T)
    return -0.5 * np.sum(resid ** 2 / sigma2 + np.log(2 * np.pi * sigma2), axis=1)

CALLS = 0
phi_batch = non_k_prior.rvs(nsamples=M_MC)  # (M, 8)
PHI_STORE.append(phi_batch)

def log_likelihood_mc(params=None, data=None, m_mc_test=None):
    global CALLS
    params = np.atleast_2d(params)      # (n, 3)
    n, _ = params.shape
    log_L = np.empty(n)
    
    # Use test M_MC value if provided, otherwise use global M_MC
    m_mc_to_use = m_mc_test if m_mc_test is not None else M_MC
    
    for i in range(n):
        # For actual MCMC (m_mc_test is None), always use the global phi_batch
        # For testing (m_mc_test is not None), generate new batches
        if m_mc_test is not None:
            # For testing, generate new batch with test size
            phi_batch_test = non_k_prior.rvs(nsamples=m_mc_to_use)
        else:
            # For actual MCMC, always use the global phi_batch
            phi_batch_test = phi_batch
            
        params_full = np.hstack([phi_batch_test, np.tile(params[i], (m_mc_to_use, 1))])  # (M, 11)
        
        # Debug: Check parameter values occasionally
        if CALLS % 1000 == 0 and i == 0:
            print(f"Debug - params[i]: {params[i]}")
            print(f"Debug - phi_batch_test range: [{phi_batch_test.min():.3f}, {phi_batch_test.max():.3f}]")
            print(f"Debug - params_full range: [{params_full.min():.3f}, {params_full.max():.3f}]")
        
        y_pred = timeseries_model(params_full)  # (M, T)
        ll_batch = _gaussian_loglike(y_pred, data, sigma2=SENSOR_VARIANCE)
        log_L[i] = logsumexp(ll_batch) - np.log(m_mc_to_use)
        
        # Only store phi_batch for actual MCMC (not during testing)
        if m_mc_test is None:
            PHI_STORE.append(phi_batch)  # Store the global batch
            
    CALLS += params.shape[0]
    if CALLS % 100 == 0:  # More frequent progress reporting
        print(f"{CALLS:,} proposals evaluated")
    return log_L



def main():
    global M_MC
    import time
    start_time = time.time()
    
    y_obs, exp_time = load_experimental_data()
    y_obs_interp = interpolate_to_surrogate_grid(y_obs, exp_time)  # shape (50,)
    
    print("\n" + "=" * 60)
    print("MCMC SIMULATION SETUP")
    print("=" * 60)
    print(f"Current M_MC: {M_MC}")
    print("You can modify M_MC at the top of the file if needed.")
    
    print("\nProceed with MCMC simulation?")
    response = input("Enter 'y' to continue, 'q' to quit: ").lower().strip()
    if response == 'q':
        print("Exiting...")
        return
    
    print("\n" + "=" * 60)
    print("STARTING MCMC SIMULATION")
    print("=" * 60)
    
    # Reset CALLS counter for actual MCMC
    global CALLS
    CALLS = 0
    print(f"Reset CALLS counter to {CALLS}")
    
    ll_model = LogLikelihoodModel(n_parameters=3, log_likelihood=log_likelihood_mc)
    ll_model.prior = k_prior
    # Note: Stretch sampler doesn't use a proposal distribution - it generates proposals
    # by stretching the ensemble of walkers. The prior is used in the Bayesian inference.
    n_walkers = 12  # Number of walkers in the ensemble
    # Draw initial positions from k_prior instead of hardcoded values
    best_fit = [3.3, 10.0, 350.0]
    seed_std_devs = [0.1, 1, 10]
    initial_positions = np.random.normal(loc=best_fit, scale=seed_std_devs, size=(n_walkers, 3))
    initial_positions = initial_positions.tolist()
    # initial_positions = k_prior.rvs(nsamples=n_walkers).tolist()
    stretch_sampler = Stretch(
        burn_length=10000,           # Longer burn-in
        jump=1,                      # No thinning during sampling
        dimension=3,
        seed=initial_positions,
        save_log_pdf=True,
        scale=2.8,                   # Reduced scale for better acceptance
        n_chains=n_walkers,
        concatenate_chains=False     # Keep chains separate for proper ESS calculation
    )
    mh_sampler = MetropolisHastings(
        burn_length=10000,
        jump=5,                      # Thinning parameter
        dimension=3,
        seed=initial_positions,
        save_log_pdf=True,
    )
    dram_sampler = DRAM(
        burn_length=20000,           # Longer burn-in
        jump=1,                      # No thinning during sampling
        dimension=3,
        seed=initial_positions,
        save_log_pdf=True,
    )
    
    bpe = BayesParameterEstimation(
        inference_model=ll_model,
        data=y_obs_interp,
        sampling_class=stretch_sampler,
        nsamples=40000 
    )
    samples_kappa = bpe.sampler.samples               # (N, 3)
    phi_chain = np.vstack(PHI_STORE)
    accepted_mask = np.isfinite(bpe.sampler.log_pdf_values)
    elapsed_time = time.time() - start_time
    np.savez("mcmc_results.npz", 
             samples_kappa=samples_kappa,
             phi_chain=phi_chain,
             accepted_mask=accepted_mask,
             log_pdf_values=bpe.sampler.log_pdf_values)
    print("\nMCMC complete!  Accepted samples:", samples_kappa.shape[0])
    print("Acceptance rate:", bpe.sampler.acceptance_rate)
    
    # Handle different sample formats (2D vs 3D)
    if len(samples_kappa.shape) == 3:
        # Samples are in format (n_samples, n_chains, n_dimensions)
        # Flatten to (n_samples * n_chains, n_dimensions) for statistics
        samples_flat = samples_kappa.reshape(-1, samples_kappa.shape[2])
        print(f"Sample format: 3D with shape {samples_kappa.shape}, flattened to {samples_flat.shape}")
    else:
        # Samples are in flat format (n_samples, n_dimensions)
        samples_flat = samples_kappa
        print(f"Sample format: 2D with shape {samples_kappa.shape}")
    
    print("k_sample mean ± σ:", samples_flat[:, 0].mean(), samples_flat[:, 0].std())
    print("k_ins    mean ± σ:", samples_flat[:, 1].mean(), samples_flat[:, 1].std())
    print("k_coupler mean ± σ:", samples_flat[:, 2].mean(), samples_flat[:, 2].std())
    
    # Print convergence diagnostics
    print(f"\nConvergence Diagnostics:")
    if len(samples_kappa.shape) == 3:
        print(f"  Total samples: {samples_kappa.shape[0] * samples_kappa.shape[1]}")
        print(f"  Samples per walker: {samples_kappa.shape[0]}")
        print(f"  Number of walkers: {samples_kappa.shape[1]}")
    else:
        print(f"  Total samples: {samples_kappa.shape[0]}")
        print(f"  Samples per walker: {samples_kappa.shape[0] // n_walkers}")
    print(f"  Total time: {elapsed_time:.1f} seconds")
    if len(samples_kappa.shape) == 3:
        print(f"  Time per sample: {elapsed_time/(samples_kappa.shape[0] * samples_kappa.shape[1]):.3f} seconds")
    else:
        print(f"  Time per sample: {elapsed_time/samples_kappa.shape[0]:.3f} seconds")
    if hasattr(bpe.sampler, 'scale'):
        print(f"  Scale parameter: {bpe.sampler.scale}")
    
    
    print("MCMC results saved to mcmc_results.npz")
    
    # Handle different sample formats for plotting
    if len(samples_kappa.shape) == 3:
        # Flatten samples for plotting
        samples_plot = samples_kappa.reshape(-1, samples_kappa.shape[2])
    else:
        samples_plot = samples_kappa
    
    plt.figure(figsize=(6,4))
    plt.scatter(samples_plot[:, 0], samples_plot[:, 1], s=2, alpha=0.4)
    plt.xlabel("k_sample")
    plt.ylabel("k_ins")
    plt.title("Posterior samples (projection)")
    plt.tight_layout()
    plt.show()
    try:
        import corner
        fig_corner = corner.corner(
            samples_plot,
            labels=[r"$k_{\text{sample}}$", r"$k_{\text{ins}}$", r"$k_{\text{coupler}}$"],
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 10}
        )
        fig_corner.savefig("corner_plot.png", dpi=300, bbox_inches="tight")
        print("Corner plot saved to corner_plot.png")
        plt.show()
    except ImportError:
        import pandas as pd
        import seaborn as sns
        df_samples = pd.DataFrame(samples_plot, columns=["k_sample", "k_ins", "k_coupler"])
        g = sns.pairplot(df_samples, corner=True, diag_kind="kde", plot_kws={"s": 5, "alpha": 0.4})
        g.figure.suptitle("Pair plot (corner style)")
        g.figure.tight_layout()
        g.figure.subplots_adjust(top=0.95)
        g.figure.savefig("corner_plot.png", dpi=300, bbox_inches="tight")
        print("Pair plot saved to corner_plot.png (corner library not installed)")
        plt.show()

if __name__ == "__main__":
    main()


