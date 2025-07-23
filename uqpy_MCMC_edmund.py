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
from train_surrogate_models import FullSurrogateModel

logging.getLogger("UQpy").setLevel(logging.DEBUG)

# Load Edmund-specific parameter definitions
param_defs = get_param_defs_from_config(config_path="configs/distributions_edmund.yaml")
uqpy_dists = create_uqpy_distributions(param_defs)
full_prior = JointIndependent(marginals=uqpy_dists)  # All parameters for Edmund

# Load Edmund surrogate model
def load_edmund_surrogate():
    """Load the Edmund-specific surrogate model"""
    surrogate = FullSurrogateModel.load_model("outputs/edmund2/full_surrogate_model.pkl")
    return surrogate

# Load experimental data for Edmund
def load_experimental_data():
    data = pd.read_csv("data/experimental/edmund_71Gpa_run2.csv")
    oside_data = data['oside'].values
    y_obs = (oside_data - oside_data[0]) / (data['temp'].max() - data['temp'].min())
    exp_time = data['time'].values
    return y_obs, exp_time

def interpolate_to_surrogate_grid(exp_data, exp_time, surrogate):
    from scipy.interpolate import interp1d
    surrogate_time_grid = surrogate.time_grid
    interp_func = interp1d(exp_time, exp_data, kind='linear', 
                           bounds_error=False, fill_value=(exp_data[0], exp_data[-1]))
    interpolated_data = interp_func(surrogate_time_grid)
    return interpolated_data

SENSOR_VARIANCE = 0.0012

# Toggle: include surrogate predictive uncertainty in the likelihood
INCLUDE_SURROGATE_UNCERT = True  # Set to False to reproduce the old behaviour

def _gaussian_loglike(y_pred: np.ndarray, y_obs: np.ndarray, *, sigma2) -> np.ndarray:
    """Vectorised Gaussian log-likelihood.

    Parameters
    ----------
    y_pred : np.ndarray (m, T)
        Model (surrogate) predictions.
    y_obs : np.ndarray (T,) or (m, T)
        Observations (broadcastable against *y_pred*).
    sigma2 : float or np.ndarray (T,) or (m, T)
        Total variance per data point.  Can be a scalar (old behaviour) or a
        vector/matrix matching *y_pred* shape.
    """
    resid = y_pred - y_obs                          # (m, T)
    sigma2 = np.asarray(sigma2)                     # ensure array for broadcasting
    return -0.5 * np.sum(resid ** 2 / sigma2 + np.log(2 * np.pi * sigma2), axis=1)

CALLS = 0

def log_likelihood_full(params=None, data=None):
    """Log likelihood function for full parameter MCMC with Edmund data"""
    global CALLS
    params = np.atleast_2d(params)      # (n, n_params)
    n, _ = params.shape
    log_L = np.empty(n)
    
    # Load surrogate model (do this inside to avoid global state issues)
    surrogate = load_edmund_surrogate()
    
    for i in range(n):
        # Generate predictions and predictive uncertainty using the surrogate model
        y_pred, _, _, curve_uncert = surrogate.predict_temperature_curves(params[i:i+1])  # Shapes (1, T)

        if INCLUDE_SURROGATE_UNCERT:
            sigma2 = SENSOR_VARIANCE + curve_uncert**2  # (1, T)
        else:
            sigma2 = SENSOR_VARIANCE  # scalar – old behaviour

        ll = _gaussian_loglike(y_pred, data, sigma2=sigma2)
        log_L[i] = ll[0]  # Extract scalar value
            
    CALLS += params.shape[0]
    if CALLS % 2000 == 0:  # More frequent progress reporting
        print(f"{CALLS:,} proposals evaluated")
    return log_L

def main():
    import time
    start_time = time.time()
    
    surrogate = load_edmund_surrogate()
    y_obs, exp_time = load_experimental_data()
    y_obs_interp = interpolate_to_surrogate_grid(y_obs, exp_time, surrogate)  # shape (70,)
    
    print("\n" + "=" * 60)
    print("EDMUND MCMC SIMULATION SETUP")
    print("=" * 60)
    print("Full parameter MCMC with Edmund experimental data")
    print("All parameters sampled from their Edmund priors")
    
    print("\nProceed with MCMC simulation?")
    response = input("Enter 'y' to continue, 'q' to quit: ").lower().strip()
    if response == 'q':
        print("Exiting...")
        return
    
    print("\n" + "=" * 60)
    print("STARTING EDMUND MCMC SIMULATION")
    print("=" * 60)
    
    # Reset CALLS counter for actual MCMC
    global CALLS
    CALLS = 0
    print(f"Reset CALLS counter to {CALLS}")
    
    n_params = len(param_defs)
    ll_model = LogLikelihoodModel(n_parameters=n_params, log_likelihood=log_likelihood_full)
    ll_model.prior = full_prior
    
    # Set up sampler
    n_walkers = 60  # More walkers for multiple dimensions
    param_names = [param_def['name'] for param_def in param_defs]
    
    # Draw initial positions from prior - Edmund-specific best fit
    best_fit = np.array([     
        2.9e-6,    # d_sample
        7072500,   # rho_cv_sample
        2621310,   # rho_cv_ins
        4.0e-6,    # d_ins_pside
        4.2e-6,    # d_ins_oside
        7.0e-6,    # fwhm
        50.0,      # k_sample (midpoint of uniform range)
        14.0,      # k_ins (midpoint of uniform range)
    ])

    n_walkers = 44                     
    jitter_frac = 0.1                 

    noise = jitter_frac * np.abs(best_fit) * np.random.randn(n_walkers, best_fit.size)
    initial_positions = best_fit + noise        # shape (n_walkers, n_params)
    initial_positions = initial_positions.tolist()

    
    stretch_sampler = Stretch(
        burn_length=20000,          
        jump=1,                    
        dimension=n_params,
        seed=initial_positions,
        save_log_pdf=True,
        scale=2.4,                   
        n_chains=n_walkers,
        concatenate_chains=False     
    )
    dream_sampler = DREAM(
        burn_length=20000,
        jump=1,
        dimension=n_params,
        seed=initial_positions,
        save_log_pdf=True,
        n_chains=n_walkers,
        concatenate_chains=False,
        c_star = 1e-11,
    )
    
    bpe = BayesParameterEstimation(
        inference_model=ll_model,
        data=y_obs_interp,
        sampling_class=stretch_sampler,
        nsamples=1000000 
    )
    
    samples_full = bpe.sampler.samples               # (N, n_params)
    accepted_mask = np.isfinite(bpe.sampler.log_pdf_values)
    elapsed_time = time.time() - start_time
    
    np.savez("mcmc_results_edmund.npz", 
             samples_full=samples_full,
             accepted_mask=accepted_mask,
             log_pdf_values=bpe.sampler.log_pdf_values)
    
    print("\nMCMC complete!  Accepted samples:", samples_full.shape[0])
    print("Acceptance rate:", bpe.sampler.acceptance_rate)
    
    # Handle different sample formats (2D vs 3D)
    if len(samples_full.shape) == 3:
        # Samples are in format (n_samples, n_chains, n_dimensions)
        # Flatten to (n_samples * n_chains, n_dimensions) for statistics
        samples_flat = samples_full.reshape(-1, samples_full.shape[2])
        print(f"Sample format: 3D with shape {samples_full.shape}, flattened to {samples_flat.shape}")
    else:
        # Samples are in flat format (n_samples, n_dimensions)
        samples_flat = samples_full
        print(f"Sample format: 2D with shape {samples_full.shape}")
    
    # Print parameter statistics
    print(f"\nParameter Statistics (mean ± σ):")
    print(f"{'Parameter':<15} {'Posterior Mean':<15} {'Posterior Std':<15}")
    print("-" * 50)
    
    for i, name in enumerate(param_names):
        post_mean = samples_flat[:, i].mean()
        post_std = samples_flat[:, i].std()
        print(f"{name:<15} {post_mean:<15.3e} {post_std:<15.3e}")
    
    # Print convergence diagnostics
    print(f"\nConvergence Diagnostics:")
    if len(samples_full.shape) == 3:
        print(f"  Total samples: {samples_full.shape[0] * samples_full.shape[1]}")
        print(f"  Samples per walker: {samples_full.shape[0]}")
        print(f"  Number of walkers: {samples_full.shape[1]}")
    else:
        print(f"  Total samples: {samples_full.shape[0]}")
        print(f"  Samples per walker: {samples_full.shape[0] // n_walkers}")
    print(f"  Total time: {elapsed_time:.1f} seconds")
    if len(samples_full.shape) == 3:
        print(f"  Time per sample: {elapsed_time/(samples_full.shape[0] * samples_full.shape[1]):.3f} seconds")
    else:
        print(f"  Time per sample: {elapsed_time/samples_full.shape[0]:.3f} seconds")
    if hasattr(bpe.sampler, 'scale'):
        print(f"  Scale parameter: {bpe.sampler.scale}")
    
    print("MCMC results saved to mcmc_results_edmund.npz")
    
    # Handle different sample formats for plotting
    if len(samples_full.shape) == 3:
        # For 3D samples, use the first chain for plotting
        samples_plot = samples_full[:, 0, :]
    else:
        samples_plot = samples_full
    
    # Create corner plot
    try:
        import corner
        fig = corner.corner(samples_plot, labels=param_names, 
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={"fontsize": 12})
        plt.savefig("corner_plot_edmund.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("Corner plot saved as corner_plot_edmund.png")
    except ImportError:
        print("corner package not available, skipping corner plot")
    
    # Create trace plots
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 2*n_params))
    if n_params == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(samples_plot[:, i], alpha=0.7)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Sample Index')
    plt.tight_layout()
    plt.savefig("trace_plots_edmund.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Trace plots saved as trace_plots_edmund.png")

if __name__ == "__main__":
    main() 