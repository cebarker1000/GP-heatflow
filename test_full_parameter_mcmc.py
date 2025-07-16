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

# Define fixed k values for synthetic data generation
FIXED_K_VALUES = {
    'k_sample': 3.3,
    'k_ins': 10.0,
    'k_coupler': 350.0
}

# Convert to array in the same order as the distributions
def get_true_parameter_array():
    """Generate true parameters with fixed k values but random nuisance parameters from priors"""
    param_names = [param_def['name'] for param_def in param_defs]
    
    # Draw nuisance parameters from their priors
    nuisance_prior = JointIndependent(marginals=uqpy_dists[:8])  # First 8 parameters
    nuisance_params = nuisance_prior.rvs(nsamples=1)[0]  # Shape (8,)
    
    # Create full parameter array
    true_array = np.zeros(11)
    true_array[:8] = nuisance_params  # Nuisance parameters from priors
    
    # Set fixed k values
    k_indices = [8, 9, 10]  # k_sample, k_ins, k_coupler
    for i, k_name in enumerate(['k_sample', 'k_ins', 'k_coupler']):
        true_array[k_indices[i]] = FIXED_K_VALUES[k_name]
    
    return true_array

def generate_synthetic_data():
    """Generate synthetic data using fixed k values but random nuisance parameters from priors"""
    true_params = get_true_parameter_array()
    
    # Print parameter summary
    param_names = [param_def['name'] for param_def in param_defs]
    print("True parameters used for synthetic data:")
    print(f"{'Parameter':<15} {'Value':<15} {'Source':<15}")
    print("-" * 45)
    for i, name in enumerate(param_names):
        if i < 8:
            source = "Prior draw"
        else:
            source = "Fixed"
        print(f"{name:<15} {true_params[i]:<15.3e} {source:<15}")
    
    # Generate synthetic data using the surrogate model
    synthetic_data = timeseries_model(true_params.reshape(1, -1))[0]  # Shape (T,)
    
    # Add noise to simulate experimental uncertainty
    SENSOR_VARIANCE = 1e-4
    noise = np.random.normal(0, np.sqrt(SENSOR_VARIANCE), synthetic_data.shape)
    noisy_data = synthetic_data + noise
    
    print(f"\nSynthetic data shape: {noisy_data.shape}")
    print(f"Data range: [{noisy_data.min():.6f}, {noisy_data.max():.6f}]")
    
    return noisy_data, true_params

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
        
        # Debug: Check parameter values occasionally
        if CALLS % 1000 == 0 and i == 0:
            print(f"Debug - params[i]: {params[i]}")
            print(f"Debug - y_pred range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
            
    CALLS += params.shape[0]
    if CALLS % 100 == 0:  # More frequent progress reporting
        print(f"{CALLS:,} proposals evaluated")
    return log_L

def main():
    import time
    start_time = time.time()
    
    # Generate synthetic data
    print("\n" + "=" * 60)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 60)
    print("Using fixed k values but random nuisance parameters from priors")
    synthetic_data, true_params = generate_synthetic_data()
    
    print("\n" + "=" * 60)
    print("MCMC SIMULATION SETUP")
    print("=" * 60)
    print("Testing full 11-parameter MCMC with synthetic data")
    print("True k values are known and should be recovered")
    print("Nuisance parameters were drawn from priors (representing experimental uncertainty)")
    
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
    n_walkers = 24  # More walkers for 11 dimensions
    param_names = [param_def['name'] for param_def in param_defs]
    
    # Draw initial positions from prior with some centered near true values
    initial_positions = full_prior.rvs(nsamples=n_walkers).tolist()
    
    # Replace a few walkers with positions near true values for better convergence
    for i in range(min(4, n_walkers)):
        noise = np.random.normal(0, 0.1 * np.abs(true_params), true_params.shape)
        initial_positions[i] = (true_params + noise).tolist()
    
    stretch_sampler = Stretch(
        burn_length=15000,           # Longer burn-in for 11 dimensions
        jump=1,                      # No thinning during sampling
        dimension=11,
        seed=initial_positions,
        save_log_pdf=True,
        scale=2.8,                   # Reduced scale for better acceptance
        n_chains=n_walkers,
        concatenate_chains=False     # Keep chains separate for proper ESS calculation
    )
    
    bpe = BayesParameterEstimation(
        inference_model=ll_model,
        data=synthetic_data,
        sampling_class=stretch_sampler,
        nsamples=30000  # Fewer samples for testing
    )
    
    samples_full = bpe.sampler.samples               # (N, 11)
    accepted_mask = np.isfinite(bpe.sampler.log_pdf_values)
    elapsed_time = time.time() - start_time
    
    # Save results
    np.savez("test_full_mcmc_results.npz", 
             samples_full=samples_full,
             accepted_mask=accepted_mask,
             log_pdf_values=bpe.sampler.log_pdf_values,
             true_parameters=true_params,
             synthetic_data=synthetic_data)
    
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
    
    # Print parameter statistics and compare with true values
    print(f"\nParameter Statistics (mean ± σ):")
    print(f"{'Parameter':<15} {'True':<12} {'Posterior Mean':<15} {'Posterior Std':<15} {'Recovery %':<10}")
    print("-" * 75)
    
    for i, name in enumerate(param_names):
        true_val = true_params[i]
        post_mean = samples_flat[:, i].mean()
        post_std = samples_flat[:, i].std()
        recovery_pct = abs(post_mean - true_val) / abs(true_val) * 100
        
        print(f"{name:<15} {true_val:<12.3e} {post_mean:<15.3e} {post_std:<15.3e} {recovery_pct:<10.1f}%")
    
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
    
    print("MCMC results saved to test_full_mcmc_results.npz")
    
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
    k_true = true_params[k_indices]
    
    plt.figure(figsize=(6,4))
    plt.scatter(k_samples[:, 0], k_samples[:, 1], s=2, alpha=0.4)
    plt.scatter(k_true[0], k_true[1], c='red', s=100, marker='*', label='True values')
    plt.xlabel("k_sample")
    plt.ylabel("k_ins")
    plt.title("Posterior samples (k_sample vs k_ins)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("test_k_corner_plot.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    try:
        import corner
        fig_corner = corner.corner(
            k_samples,
            labels=[r"$k_{\text{sample}}$", r"$k_{\text{ins}}$", r"$k_{\text{coupler}}$"],
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 10},
            truths=k_true,
            truth_color='red'
        )
        fig_corner.savefig("test_k_corner_plot.png", dpi=300, bbox_inches="tight")
        print("Corner plot saved to test_k_corner_plot.png")
        plt.show()
    except ImportError:
        import pandas as pd
        import seaborn as sns
        df_samples = pd.DataFrame(k_samples, columns=k_names)
        g = sns.pairplot(df_samples, corner=True, diag_kind="kde", plot_kws={"s": 5, "alpha": 0.4})
        g.figure.suptitle("Pair plot (corner style) - k parameters")
        g.figure.tight_layout()
        g.figure.subplots_adjust(top=0.95)
        g.figure.savefig("test_k_corner_plot.png", dpi=300, bbox_inches="tight")
        print("Pair plot saved to test_k_corner_plot.png (corner library not installed)")
        plt.show()

if __name__ == "__main__":
    main() 