import yaml
import argparse
from analysis.config_utils import create_uqpy_distributions, get_param_defs_from_config
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sampling.mcmc.Stretch import Stretch
from UQpy.inference.inference_models.LogLikelihoodModel import LogLikelihoodModel
from UQpy.inference.BayesParameterEstimation import BayesParameterEstimation
from scipy.special import logsumexp
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# grab surrogate model
from train_surrogate_models import FullSurrogateModel

logging.getLogger("UQpy").setLevel(logging.DEBUG)

def load_experimental_data(data_path):
    data = pd.read_csv(data_path)
    oside_data = data['oside'].values
    temp_data = data['temp'].values
    
    # Calculate normalization factors
    oside_min = oside_data[0]
    temp_range = temp_data.max() - temp_data.min()
    
    # Apply normalization
    y_obs = (oside_data - oside_min) / temp_range
    exp_time = data['time'].values
    
    # Print normalization information
    print(f"Experimental data normalization:")
    print(f"  Original oside range: {oside_data.min():.6f} to {oside_data.max():.6f}")
    print(f"  Original temp range: {temp_data.min():.6f} to {temp_data.max():.6f}")
    print(f"  Normalization factor (temp_range): {temp_range:.6f}")
    print(f"  Normalized oside range: {y_obs.min():.6f} to {y_obs.max():.6f}")
    print(f"  Normalized oside std: {y_obs.std():.6f}")
    
    return y_obs, exp_time

def interpolate_to_surrogate_grid(exp_data, exp_time, surrogate):
    from scipy.interpolate import interp1d
    
    # Use the number of steps defined in the surrogate model
    n_time_steps = surrogate.num_steps
    surrogate_time_grid = np.linspace(surrogate.time_grid[0], surrogate.time_grid[-1], n_time_steps)

    interp_func = interp1d(exp_time, exp_data, kind='linear',
                           bounds_error=False, fill_value=(exp_data[0], exp_data[-1]))
    interpolated_data = interp_func(surrogate_time_grid)
    return interpolated_data

SENSOR_VARIANCE = 1.3e-4  # You can adjust this value based on your sensor characteristics
INCLUDE_SURROGATE_UNCERT = True  # Set to False to use only sensor variance

def _gaussian_loglike(y_pred: np.ndarray, y_obs: np.ndarray, *, sigma2) -> np.ndarray:
    """Vectorised Gaussian log-likelihood."""
    resid = y_pred - y_obs                          # (m, T)
    sigma2 = np.asarray(sigma2)                     # ensure array for broadcasting
    return -0.5 * np.sum(resid ** 2 / sigma2 + np.log(2 * np.pi * sigma2), axis=1)

CALLS = 0

def log_likelihood_full(params=None, data=None, surrogate=None):
    """Log likelihood function for full parameter MCMC"""
    global CALLS
    params = np.atleast_2d(params)      # (n, n_params)
    n, _ = params.shape
    log_L = np.empty(n)
    
    for i in range(n):
        # Generate predictions and predictive uncertainty using the surrogate model
        y_pred, _, _, curve_uncert = surrogate.predict_temperature_curves(params[i:i+1])  # Shapes (1, T)

        if INCLUDE_SURROGATE_UNCERT:
            sigma2 = SENSOR_VARIANCE + curve_uncert**2  # (1, T)
        else:
            sigma2 = SENSOR_VARIANCE  # scalar

        ll = _gaussian_loglike(y_pred, data, sigma2=sigma2)
        log_L[i] = ll[0]  # Extract scalar value
            
    CALLS += params.shape[0]
    if CALLS % 10000 == 0:  # More frequent progress reporting
        print(f"{CALLS:,} proposals evaluated")
    return log_L

def main():
    parser = argparse.ArgumentParser(description="Run MCMC analysis for heat flow model.")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the distributions YAML file.")
    parser.add_argument('--surrogate_path', type=str, required=True, help="Path to the trained surrogate model file.")
    parser.add_argument('--exp_data_path', type=str, required=True, help="Path to the experimental data CSV file.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the MCMC results NPZ file.")
    parser.add_argument('--plot_path_prefix', type=str, required=True, help="Prefix for output plot filenames.")
    parser.add_argument('--n_walkers', type=int, default=60, help="Number of MCMC walkers.")
    parser.add_argument('--n_samples', type=int, default=1000000, help="Number of MCMC samples to generate.")
    parser.add_argument('--burn_length', type=int, default=20000, help="Number of burn-in samples.")

    args = parser.parse_args()

    import time
    start_time = time.time()
    
    # Load components based on command-line arguments
    param_defs = get_param_defs_from_config(config_path=args.config_path)
    uqpy_dists = create_uqpy_distributions(param_defs)
    full_prior = JointIndependent(marginals=uqpy_dists)
    
    surrogate = FullSurrogateModel.load_model(args.surrogate_path)
    y_obs, exp_time = load_experimental_data(args.exp_data_path)
    y_obs_interp = interpolate_to_surrogate_grid(y_obs, exp_time, surrogate)
    
    print("\n" + "=" * 60)
    print("MCMC SIMULATION SETUP")
    print("=" * 60)
    print(f"Config: {args.config_path}")
    print(f"Surrogate: {args.surrogate_path}")
    print(f"Experimental Data: {args.exp_data_path}")
    print(f"Output: {args.output_path}")
    print(f"Including surrogate uncertainty: {INCLUDE_SURROGATE_UNCERT}")
    
    print("\n" + "=" * 60)
    print("STARTING MCMC SIMULATION")
    print("=" * 60)
    
    # Reset CALLS counter
    global CALLS
    CALLS = 0
    print(f"Reset CALLS counter to {CALLS}")
    
    n_params = len(param_defs)
    log_likelihood_with_surrogate = lambda params, data: log_likelihood_full(params=params, data=data, surrogate=surrogate)
    ll_model = LogLikelihoodModel(n_parameters=n_params, log_likelihood=log_likelihood_with_surrogate)
    ll_model.prior = full_prior
    
    # Set up sampler
    param_names = [param_def['name'] for param_def in param_defs]
    
    # Draw initial positions from the prior
    initial_positions = full_prior.rvs(nsamples=args.n_walkers)
    
    stretch_sampler = Stretch(
        burn_length=args.burn_length,
        jump=1,
        dimension=n_params,
        seed=initial_positions.tolist(),
        save_log_pdf=True,
        scale=2.4,
        n_chains=args.n_walkers,
        concatenate_chains=False
    )
    
    bpe = BayesParameterEstimation(
        inference_model=ll_model,
        data=y_obs_interp,
        sampling_class=stretch_sampler,
        nsamples=args.n_samples
    )
    
    samples_full = bpe.sampler.samples
    accepted_mask = np.isfinite(bpe.sampler.log_pdf_values)
    elapsed_time = time.time() - start_time
    
    np.savez(args.output_path, 
             samples_full=samples_full,
             accepted_mask=accepted_mask,
             log_pdf_values=bpe.sampler.log_pdf_values,
             param_names=param_names)
    
    print(f"\nMCMC complete! Total samples: {samples_full.size}")
    print(f"Acceptance rate: {bpe.sampler.acceptance_rate}")
    
    # Handle sample format (always 3D from Stretch with concatenate_chains=False)
    # Flatten to (n_samples * n_chains, n_dimensions) for statistics
    samples_flat = samples_full.reshape(-1, samples_full.shape[2])
    print(f"Sample format: 3D with shape {samples_full.shape}, flattened to {samples_flat.shape}")
    
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
    print(f"  Total samples: {samples_full.shape[0] * samples_full.shape[1]}")
    print(f"  Samples per walker: {samples_full.shape[0]}")
    print(f"  Number of walkers: {args.n_walkers}")
    print(f"  Total time: {elapsed_time:.1f} seconds")
    print(f"  Time per sample: {elapsed_time/(samples_full.shape[0] * samples_full.shape[1]):.3f} seconds")
    if hasattr(bpe.sampler, 'scale'):
        print(f"  Scale parameter: {bpe.sampler.scale}")
    
    print(f"MCMC results saved to {args.output_path}")
    
    # Create corner plot for all parameters
    try:
        import corner
        corner_plot_path = f"{args.plot_path_prefix}_corner.png"
        fig_corner = corner.corner(
            samples_flat,
            labels=param_names,
            show_titles=True,
            title_fmt=".2e",
            title_kwargs={"fontsize": 10}
        )
        fig_corner.savefig(corner_plot_path, dpi=300, bbox_inches="tight")
        print(f"Corner plot saved to {corner_plot_path}")
        plt.close(fig_corner)
    except ImportError:
        print("Could not import 'corner' library, skipping corner plot.")

    # Create trace plots for all parameters
    trace_plot_path = f"{args.plot_path_prefix}_trace.png"
    fig_trace, axes = plt.subplots(n_params, 1, figsize=(12, 2 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]
    
    for i in range(n_params):
        for j in range(args.n_walkers):
            axes[i].plot(samples_full[:, j, i], alpha=0.5)
        axes[i].set_ylabel(param_names[i])
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Sample Index')
    fig_trace.suptitle("MCMC Trace Plots", fontsize=16)
    fig_trace.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(trace_plot_path, dpi=300, bbox_inches='tight')
    print(f"Trace plots saved to {trace_plot_path}")
    plt.close(fig_trace)

if __name__ == "__main__":
    main()


