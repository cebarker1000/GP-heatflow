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
full_prior = JointIndependent(marginals=uqpy_dists)  # All 11 parameters

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

def _gaussian_loglike(y_pred: np.ndarray, y_obs: np.ndarray, *, sigma2: float) -> np.ndarray:
    resid = y_pred - y_obs              # (m, T)
    return -0.5 * np.sum(resid ** 2 / sigma2 + np.log(2 * np.pi * sigma2), axis=1)

CALLS = 0

def log_likelihood_full(params=None, data=None):
    """Log likelihood function for full 11-parameter MCMC"""
    global CALLS
    params = np.atleast_2d(params)      # (n, 11)
    n, _ = params.shape
    log_L = np.empty(n)
    
    for i in range(n):
        # Generate predictions using the surrogate model
        y_pred = timeseries_model(params[i:i+1])  # Shape (1, T)
        ll = _gaussian_loglike(y_pred, data, sigma2=SENSOR_VARIANCE)
        log_L[i] = ll[0]  # Extract scalar value
            
    CALLS += params.shape[0]
    if CALLS % 2000 == 0:  # More frequent progress reporting
        print(f"{CALLS:,} proposals evaluated")
    return log_L

def main():
    import time
    start_time = time.time()
    
    y_obs, exp_time = load_experimental_data()
    y_obs_interp = interpolate_to_surrogate_grid(y_obs, exp_time)  # shape (50,)
    
    print("\n" + "=" * 60)
    print("MCMC SIMULATION SETUP")
    print("=" * 60)
    print("Full 11-parameter MCMC with experimental data")
    print("All parameters (nuisance + k values) sampled from their priors")
    
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
    
    ll_model = LogLikelihoodModel(n_parameters=11, log_likelihood=log_likelihood_full)
    ll_model.prior = full_prior
    
    # Set up sampler
    n_walkers = 60  # More walkers for 11 dimensions
    param_names = [param_def['name'] for param_def in param_defs]
    
    # Draw initial positions from prior
    best_fit = np.array([     
    1.88e-6, 6.1e6, 3.44e6, 2.75e6, 6.2e-8,
    3.20e-6, 6.30e-6, 1.33e-5, 3.56, 9.81, 4.0e2
    ])

    n_walkers = 44                     
    jitter_frac = 0.1                 

    noise = jitter_frac * np.abs(best_fit) * np.random.randn(n_walkers, best_fit.size)
    initial_positions = best_fit + noise        # shape (n_walkers, 11)
    initial_positions = initial_positions.tolist()

    
    stretch_sampler = Stretch(
        burn_length=20000,          
        jump=1,                    
        dimension=11,
        seed=initial_positions,
        save_log_pdf=True,
        scale=2.4,                   
        n_chains=n_walkers,
        concatenate_chains=False     
    )
    dream_sampler = DREAM(
        burn_length=20000,
        jump=1,
        dimension=11,
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
        nsamples=1500000 
    )
    
    samples_full = bpe.sampler.samples               # (N, 11)
    accepted_mask = np.isfinite(bpe.sampler.log_pdf_values)
    elapsed_time = time.time() - start_time
    
    np.savez("mcmc_results.npz", 
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
    
    print("MCMC results saved to mcmc_results.npz")
    
    # Handle different sample formats for plotting
    if len(samples_full.shape) == 3:
        # Flatten samples for plotting
        samples_plot = samples_full.reshape(-1, samples_full.shape[2])
    else:
        samples_plot = samples_full
    
    # Create corner plot for key parameters (k values)
    k_indices = [8, 9, 10]  # k_sample, k_ins, k_coupler
    k_names = ['k_sample', 'k_ins', 'k_coupler']
    k_samples = samples_plot[:, k_indices]
    
    plt.figure(figsize=(6,4))
    plt.scatter(k_samples[:, 0], k_samples[:, 1], s=2, alpha=0.4)
    plt.xlabel("k_sample")
    plt.ylabel("k_ins")
    plt.title("Posterior samples (k_sample vs k_ins)")
    plt.tight_layout()
    plt.savefig("k_corner_plot.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    try:
        import corner
        fig_corner = corner.corner(
            k_samples,
            labels=[r"$k_{\text{sample}}$", r"$k_{\text{ins}}$", r"$k_{\text{coupler}}$"],
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 10}
        )
        fig_corner.savefig("k_corner_plot.png", dpi=300, bbox_inches="tight")
        print("Corner plot saved to k_corner_plot.png")
        plt.show()
    except ImportError:
        import pandas as pd
        import seaborn as sns
        df_samples = pd.DataFrame(k_samples, columns=k_names)
        g = sns.pairplot(df_samples, corner=True, diag_kind="kde", plot_kws={"s": 5, "alpha": 0.4})
        g.figure.suptitle("Pair plot (corner style) - k parameters")
        g.figure.tight_layout()
        g.figure.subplots_adjust(top=0.95)
        g.figure.savefig("k_corner_plot.png", dpi=300, bbox_inches="tight")
        print("Pair plot saved to k_corner_plot.png (corner library not installed)")
        plt.show()

if __name__ == "__main__":
    main()


