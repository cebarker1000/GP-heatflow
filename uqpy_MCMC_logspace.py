#!/usr/bin/env python3
"""
Log-space MCMC sampling for thermal conductivity estimation.
This script implements MCMC sampling in log-space for all parameters to improve
convergence and mixing, with proper Jacobian corrections.
"""

import yaml
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import logging

from analysis.config_utils import (
    get_param_defs_from_config, 
    create_logspace_distributions,
    real_to_log_space,
    log_to_real_space,
    compute_jacobian_correction,
    get_logspace_bounds
)
from UQpy.distributions.collection.JointIndependent import JointIndependent
from UQpy.sampling.mcmc.Stretch import Stretch
from UQpy.inference.inference_models.LogLikelihoodModel import LogLikelihoodModel
from UQpy.inference.BayesParameterEstimation import BayesParameterEstimation

# Import surrogate model
from uqpy_surrogate import timeseries_model

# Set up logging
logging.getLogger("UQpy").setLevel(logging.DEBUG)

# ------------------------------------------------------------
# Baseline helper
# ------------------------------------------------------------


def _compute_baseline(times: np.ndarray, temps: np.ndarray, *, cfg_path: str | None = None):
    if cfg_path is None or not os.path.exists(cfg_path):
        return float(temps[0])

    import yaml
    with open(cfg_path, "r") as f:
        sim_cfg = yaml.safe_load(f)

    baseline_cfg = sim_cfg.get("baseline", {})
    if not baseline_cfg.get("use_average", False):
        return float(temps[0])

    t_window = float(baseline_cfg.get("time_window", 0.0))
    mask = times <= t_window
    if mask.any():
        return float(np.mean(temps[mask]))
    return float(temps[0])

# Global variables
CALLS = 0
SENSOR_VARIANCE = 1e-4  # Sensor noise variance

def load_experimental_data(cfg_path):
    """Load experimental data and normalise with configurable baseline."""
    data = pd.read_csv("data/experimental/geballe_heat_data.csv")

    oside_data = data["oside"].values
    temp_data = data["temp"].values  # pside
    times = data["time"].values

    baseline_pside = _compute_baseline(times, temp_data, cfg_path=cfg_path)
    baseline_oside = _compute_baseline(times, oside_data, cfg_path=cfg_path)

    excursion_pside = (temp_data - baseline_pside).max() - (temp_data - baseline_pside).min()
    if excursion_pside <= 0:
        raise ValueError("Temp excursion is zero")

    y_obs = (oside_data - baseline_oside) / excursion_pside
    return y_obs, times

def interpolate_to_surrogate_grid(exp_data, exp_time):
    """Interpolate experimental data to surrogate model time grid."""
    from scipy.interpolate import interp1d
    sim_t_final = 7.5e-6  # seconds
    sim_num_steps = 50
    surrogate_time_grid = np.linspace(0, sim_t_final, sim_num_steps)
    interp_func = interp1d(exp_time, exp_data, kind='linear', 
                           bounds_error=False, fill_value=(exp_data[0], exp_data[-1]))
    interpolated_data = interp_func(surrogate_time_grid)
    return interpolated_data

def _gaussian_loglike(y_pred: np.ndarray, y_obs: np.ndarray, *, sigma2: float) -> np.ndarray:
    """Compute Gaussian log-likelihood."""
    resid = y_pred - y_obs              # (m, T)
    return -0.5 * np.sum(resid ** 2 / sigma2 + np.log(2 * np.pi * sigma2), axis=1)

def log_likelihood_logspace(params_log=None, data=None, param_defs=None):
    """
    Log-likelihood function for log-space MCMC sampling.
    
    This function:
    1. Transforms log-space parameters back to real-space
    2. Computes the likelihood using the surrogate model
    3. Applies Jacobian correction for the transformation
    
    Args:
        params_log: Parameters in log-space, shape (n, 11)
        data: Experimental data
        param_defs: Parameter definitions from config
        
    Returns:
        Log-likelihood values with Jacobian correction
    """
    global CALLS
    
    if params_log is None or data is None or param_defs is None:
        raise ValueError("All arguments must be provided")
    
    params_log = np.atleast_2d(params_log)  # (n, 11)
    n, _ = params_log.shape
    log_L = np.empty(n)
    
    for i in range(n):
        # Transform from log-space to real-space
        params_real = log_to_real_space(params_log[i:i+1], param_defs)
        
        # Generate predictions using the surrogate model
        y_pred = timeseries_model(params_real)  # Shape (1, T)
        
        # Compute likelihood in real-space
        ll_real = _gaussian_loglike(y_pred, data, sigma2=SENSOR_VARIANCE)
        
        # Apply Jacobian correction
        jacobian_correction = np.log(compute_jacobian_correction(params_real, param_defs))
        
        # Total log-likelihood = real-space likelihood + Jacobian correction
        log_L[i] = ll_real[0] + jacobian_correction
        
        # Debug: Check parameter values occasionally
        if CALLS % 1000 == 0 and i == 0:
            """
            print(f"Debug - params_log[i]: {params_log[i]}")
            print(f"Debug - params_real[i]: {params_real[0]}")
            print(f"Debug - y_pred range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
            print(f"Debug - Jacobian correction: {float(jacobian_correction):.6f}")
            """
            
    CALLS += params_log.shape[0]
    if CALLS % 2000 == 0:
        print(f"{CALLS:,} proposals evaluated")
    
    return log_L

def main():
    """Main function for log-space MCMC sampling."""
    start_time = time.time()
    
    # Load parameter definitions
    param_defs = get_param_defs_from_config(config_path="configs/distributions.yaml")
    param_names = [param_def['name'] for param_def in param_defs]
    
    print("\n" + "=" * 80)
    print("LOG-SPACE MCMC SIMULATION SETUP")
    print("=" * 80)
    print("Full 11-parameter MCMC with experimental data")
    print("All parameters sampled in log-space for improved convergence")
    print(f"Parameter names: {param_names}")
    
    # Determine sim config path for baseline
    cfg_path = "configs/config_5_materials.yaml"
    for arg in sys.argv:
        if arg.startswith("--sim_cfg="):
            cfg_path = arg.split("=", 1)[1]

    # Load experimental data
    y_obs, exp_time = load_experimental_data(cfg_path)
    y_obs_interp = interpolate_to_surrogate_grid(y_obs, exp_time)  # shape (50,)
    
    print("\nProceed with log-space MCMC simulation?")
    response = input("Enter 'y' to continue, 'q' to quit: ").lower().strip()
    if response == 'q':
        print("Exiting...")
        return
    
    print("\n" + "=" * 80)
    print("STARTING LOG-SPACE MCMC SIMULATION")
    print("=" * 80)
    
    # Reset CALLS counter for actual MCMC
    global CALLS
    CALLS = 0
    print(f"Reset CALLS counter to {CALLS}")
    
    # Create log-space distributions
    print("Creating log-space distributions...")
    logspace_dists = create_logspace_distributions(param_defs)
    logspace_prior = JointIndependent(marginals=logspace_dists)
    
    # Create likelihood model with log-space transformation
    def log_likelihood_wrapper(params=None, data=None):
        return log_likelihood_logspace(params, data, param_defs)
    
    ll_model = LogLikelihoodModel(n_parameters=11, log_likelihood=log_likelihood_wrapper)
    ll_model.prior = logspace_prior
    
    # Set up sampler parameters
    n_walkers = 44
    print(f"Number of walkers: {n_walkers}")
    
    # Create initial positions in log-space
    print("Creating initial positions in log-space...")
    
    # Start from reasonable values in real-space
    best_fit_real = np.array([     
        1.88e-6, 6.1e6, 3.44e6, 2.75e6, 6.2e-8,
        3.20e-6, 6.30e-6, 1.33e-5, 3.56, 9.81, 4.0e2
    ])
    
    # Transform to log-space
    best_fit_log = real_to_log_space(best_fit_real, param_defs)
    
    # Add jitter in log-space
    jitter_frac = 0.1
    noise = jitter_frac * np.abs(best_fit_log) * np.random.randn(n_walkers, best_fit_log.size)
    initial_positions_log = best_fit_log + noise
    
    # Ensure initial positions are within reasonable bounds
    logspace_bounds = get_logspace_bounds(param_defs)
    for i, (low, high) in enumerate(logspace_bounds):
        initial_positions_log[:, i] = np.clip(initial_positions_log[:, i], low, high)
    
    initial_positions_log = initial_positions_log.tolist()
    
    print(f"Initial positions in log-space:")
    for i, name in enumerate(param_names):
        print(f"  {name}: {best_fit_log[i]:.6f}")
    
    # Set up MCMC sampler
    print("Setting up Stretch sampler...")
    stretch_sampler = Stretch(
        burn_length=20000,          
        jump=1,                    
        dimension=11,
        seed=initial_positions_log,
        save_log_pdf=True,
        scale=2.8,                   
        n_chains=n_walkers,
        concatenate_chains=False     
    )
    
    # Run MCMC
    print("Running MCMC sampling...")
    bpe = BayesParameterEstimation(
        inference_model=ll_model,
        data=y_obs_interp,
        sampling_class=stretch_sampler,
        nsamples=200000 
    )
    
    # Get results
    samples_logspace = bpe.sampler.samples  # (N, 11) in log-space
    accepted_mask = np.isfinite(bpe.sampler.log_pdf_values)
    elapsed_time = time.time() - start_time
    
    print(f"\nMCMC complete! Accepted samples: {samples_logspace.shape[0]}")
    print(f"Acceptance rate: {bpe.sampler.acceptance_rate}")
    
    # Transform samples back to real-space
    print("Transforming samples to real-space...")
    
    # Handle different sample formats
    if len(samples_logspace.shape) == 3:
        # Samples are in format (n_samples, n_chains, n_dimensions)
        # Reshape to 2D for transformation, then reshape back
        original_shape = samples_logspace.shape
        samples_logspace_flat = samples_logspace.reshape(-1, samples_logspace.shape[2])
        samples_realspace_flat = log_to_real_space(samples_logspace_flat, param_defs)
        samples_realspace = samples_realspace_flat.reshape(original_shape)
    else:
        # Samples are in flat format (n_samples, n_dimensions)
        samples_realspace = log_to_real_space(samples_logspace, param_defs)
    
    # Save results in both spaces
    print("Saving results...")
    np.savez("mcmc_results_logspace.npz", 
             samples_logspace=samples_logspace,
             accepted_mask=accepted_mask,
             log_pdf_values=bpe.sampler.log_pdf_values,
             param_names=param_names,
             sampling_space="logspace")
    
    np.savez("mcmc_results_realspace.npz", 
             samples_realspace=samples_realspace,
             accepted_mask=accepted_mask,
             log_pdf_values=bpe.sampler.log_pdf_values,
             param_names=param_names,
             sampling_space="realspace")
    
    # Handle different sample formats for analysis
    if len(samples_logspace.shape) == 3:
        # Samples are in format (n_samples, n_chains, n_dimensions)
        samples_logspace_flat = samples_logspace.reshape(-1, samples_logspace.shape[2])
        samples_realspace_flat = samples_realspace.reshape(-1, samples_realspace.shape[2])
        print(f"Sample format: 3D with shape {samples_logspace.shape}, flattened to {samples_logspace_flat.shape}")
    else:
        # Samples are in flat format (n_samples, n_dimensions)
        samples_logspace_flat = samples_logspace
        samples_realspace_flat = samples_realspace
        print(f"Sample format: 2D with shape {samples_logspace.shape}")
    
    # Print parameter statistics in both spaces
    print(f"\n" + "=" * 80)
    print("PARAMETER STATISTICS")
    print("=" * 80)
    
    print(f"\nLog-space Statistics (mean ± σ):")
    print(f"{'Parameter':<15} {'Log-space Mean':<15} {'Log-space Std':<15}")
    print("-" * 50)
    
    for i, name in enumerate(param_names):
        post_mean_log = samples_logspace_flat[:, i].mean()
        post_std_log = samples_logspace_flat[:, i].std()
        print(f"{name:<15} {post_mean_log:<15.6f} {post_std_log:<15.6f}")
    
    print(f"\nReal-space Statistics (mean ± σ):")
    print(f"{'Parameter':<15} {'Real-space Mean':<15} {'Real-space Std':<15}")
    print("-" * 50)
    
    for i, name in enumerate(param_names):
        post_mean_real = samples_realspace_flat[:, i].mean()
        post_std_real = samples_realspace_flat[:, i].std()
        print(f"{name:<15} {post_mean_real:<15.3e} {post_std_real:<15.3e}")
    
    # Print convergence diagnostics
    print(f"\n" + "=" * 80)
    print("CONVERGENCE DIAGNOSTICS")
    print("=" * 80)
    
    if len(samples_logspace.shape) == 3:
        print(f"  Total samples: {samples_logspace.shape[0] * samples_logspace.shape[1]}")
        print(f"  Samples per walker: {samples_logspace.shape[0]}")
        print(f"  Number of walkers: {samples_logspace.shape[1]}")
    else:
        print(f"  Total samples: {samples_logspace.shape[0]}")
        print(f"  Samples per walker: {samples_logspace.shape[0] // n_walkers}")
    
    print(f"  Total time: {elapsed_time:.1f} seconds")
    if len(samples_logspace.shape) == 3:
        print(f"  Time per sample: {elapsed_time/(samples_logspace.shape[0] * samples_logspace.shape[1]):.3f} seconds")
    else:
        print(f"  Time per sample: {elapsed_time/samples_logspace.shape[0]:.3f} seconds")
    
    if hasattr(bpe.sampler, 'scale'):
        print(f"  Scale parameter: {bpe.sampler.scale}")
    
    print("\nResults saved to:")
    print("  mcmc_results_logspace.npz (log-space samples)")
    print("  mcmc_results_realspace.npz (real-space samples)")
    
    # Create basic trace plots for k parameters
    print("\nCreating trace plots for k parameters...")
    k_indices = [8, 9, 10]  # k_sample, k_ins, k_coupler
    k_names = ['k_sample', 'k_ins', 'k_coupler']
    
    # Trace plots in log-space
    plt.figure(figsize=(15, 5))
    for i, (idx, name) in enumerate(zip(k_indices, k_names)):
        plt.subplot(1, 3, i+1)
        plt.plot(samples_logspace_flat[:, idx], alpha=0.6, linewidth=0.5)
        plt.title(f"Log-space Trace: {name}")
        plt.xlabel("Sample Index")
        plt.ylabel("Log Parameter Value")
    plt.tight_layout()
    plt.savefig("trace_plots_logspace.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Trace plots in real-space
    plt.figure(figsize=(15, 5))
    for i, (idx, name) in enumerate(zip(k_indices, k_names)):
        plt.subplot(1, 3, i+1)
        plt.plot(samples_realspace_flat[:, idx], alpha=0.6, linewidth=0.5)
        plt.title(f"Real-space Trace: {name}")
        plt.xlabel("Sample Index")
        plt.ylabel("Parameter Value")
    plt.tight_layout()
    plt.savefig("trace_plots_realspace.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Create corner plots for k parameters
    print("Creating corner plots for k parameters...")
    k_samples_logspace = samples_logspace_flat[:, k_indices]
    k_samples_realspace = samples_realspace_flat[:, k_indices]
    
    try:
        import corner
        
        # Corner plot in log-space
        fig_corner_log = corner.corner(
            k_samples_logspace,
            labels=[r"$\log(k_{\text{sample}})$", r"$\log(k_{\text{ins}})$", r"$\log(k_{\text{coupler}})$"],
            show_titles=True,
            title_fmt=".3f",
            title_kwargs={"fontsize": 10}
        )
        fig_corner_log.suptitle("Log-space Posterior - κ Parameters", fontsize=14)
        fig_corner_log.savefig("k_corner_plot_logspace.png", dpi=300, bbox_inches="tight")
        print("Log-space corner plot saved to k_corner_plot_logspace.png")
        plt.show()
        
        # Corner plot in real-space
        fig_corner_real = corner.corner(
            k_samples_realspace,
            labels=[r"$k_{\text{sample}}$", r"$k_{\text{ins}}$", r"$k_{\text{coupler}}$"],
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 10}
        )
        fig_corner_real.suptitle("Real-space Posterior - κ Parameters", fontsize=14)
        fig_corner_real.savefig("k_corner_plot_realspace.png", dpi=300, bbox_inches="tight")
        print("Real-space corner plot saved to k_corner_plot_realspace.png")
        plt.show()
        
    except ImportError:
        print("Corner library not installed, skipping corner plots")
        # Fallback to simple scatter plots
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.scatter(k_samples_logspace[:, 0], k_samples_logspace[:, 1], s=2, alpha=0.4)
        plt.xlabel(r"$\log(k_{\text{sample}})$")
        plt.ylabel(r"$\log(k_{\text{ins}})$")
        plt.title("Log-space Posterior (k_sample vs k_ins)")
        
        plt.subplot(1, 2, 2)
        plt.scatter(k_samples_realspace[:, 0], k_samples_realspace[:, 1], s=2, alpha=0.4)
        plt.xlabel(r"$k_{\text{sample}}$")
        plt.ylabel(r"$k_{\text{ins}}$")
        plt.title("Real-space Posterior (k_sample vs k_ins)")
        
        plt.tight_layout()
        plt.savefig("k_scatter_plots.png", dpi=300, bbox_inches="tight")
        plt.show()
    
    print("\nLog-space MCMC simulation completed successfully!")

if __name__ == "__main__":
    main() 