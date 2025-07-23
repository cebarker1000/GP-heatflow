#!/usr/bin/env python3
"""
Debug script to investigate why k_ins is hitting the lower bound in MCMC.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from train_surrogate_models import FullSurrogateModel
from uqpy_MCMC_edmund import log_likelihood_full, load_experimental_data, interpolate_to_surrogate_grid
from analysis.config_utils import get_param_defs_from_config

def debug_k_ins_boundary():
    """Debug why k_ins is hitting the lower bound."""
    
    print("=" * 60)
    print("K_INS BOUNDARY DEBUG INVESTIGATION")
    print("=" * 60)
    
    # Load experimental data
    print("\n1. Loading experimental data...")
    y_obs, exp_time = load_experimental_data()
    y_obs_interp = interpolate_to_surrogate_grid(y_obs, exp_time)
    
    # Load MCMC results
    print("\n2. Loading MCMC results...")
    mcmc_data = np.load('mcmc_results_edmund.npz')
    samples_full = mcmc_data['samples_full']
    log_pdf_values = mcmc_data['log_pdf_values']
    
    # Flatten samples
    if len(samples_full.shape) == 3:
        samples_flat = samples_full.reshape(-1, samples_full.shape[2])
        log_pdf_flat = log_pdf_values.reshape(-1)
    else:
        samples_flat = samples_full
        log_pdf_flat = log_pdf_values
    
    # Get parameter names and definitions
    param_defs = get_param_defs_from_config(config_path="configs/distributions_edmund.yaml")
    param_names = [param_def['name'] for param_def in param_defs]
    
    # Find parameter indices
    k_ins_idx = param_names.index('k_ins')
    k_sample_idx = param_names.index('k_sample')
    
    print(f"\n3. Analyzing k_ins distribution...")
    
    # Extract k_ins values
    k_ins_values = samples_flat[:, k_ins_idx]
    k_sample_values = samples_flat[:, k_sample_idx]
    
    print(f"k_ins statistics:")
    print(f"  Mean: {k_ins_values.mean():.3f}")
    print(f"  Std:  {k_ins_values.std():.3f}")
    print(f"  Min:  {k_ins_values.min():.3f}")
    print(f"  Max:  {k_ins_values.max():.3f}")
    print(f"  Median: {np.median(k_ins_values):.3f}")
    
    # Check for boundary hitting
    k_ins_prior_low = 1.0
    k_ins_prior_high = 30.0
    
    near_lower_bound = k_ins_values < (k_ins_prior_low + 0.1)
    near_upper_bound = k_ins_values > (k_ins_prior_high - 0.1)
    
    print(f"\nBoundary analysis:")
    print(f"  Samples near lower bound (< {k_ins_prior_low + 0.1}): {near_lower_bound.sum()} ({100*near_lower_bound.mean():.1f}%)")
    print(f"  Samples near upper bound (> {k_ins_prior_high - 0.1}): {near_upper_bound.sum()} ({100*near_upper_bound.mean():.1f}%)")
    print(f"  Samples at exact lower bound: {(k_ins_values == k_ins_prior_low).sum()} ({100*(k_ins_values == k_ins_prior_low).mean():.1f}%)")
    
    # Test 1: Likelihood surface exploration
    print(f"\n4. Testing likelihood surface around k_ins boundary...")
    
    # Create a grid of k_ins values while holding other parameters at posterior mean
    posterior_mean = samples_flat.mean(axis=0)
    k_ins_test_range = np.linspace(0.5, 5.0, 20)  # Test below and above the boundary
    
    ll_values = []
    for k_ins_test in k_ins_test_range:
        test_params = posterior_mean.copy()
        test_params[k_ins_idx] = k_ins_test
        ll = log_likelihood_full(test_params.reshape(1, -1), y_obs_interp)[0]
        ll_values.append(ll)
    
    ll_values = np.array(ll_values)
    
    print(f"Likelihood at k_ins = 0.5: {ll_values[0]:.3f}")
    print(f"Likelihood at k_ins = 1.0: {ll_values[4]:.3f}")
    print(f"Likelihood at k_ins = 2.0: {ll_values[8]:.3f}")
    print(f"Likelihood at k_ins = 5.0: {ll_values[-1]:.3f}")
    
    # Find where likelihood drops significantly
    max_ll = ll_values.max()
    significant_drop = ll_values < (max_ll - 10)  # 10 log units drop
    if significant_drop.any():
        drop_point = k_ins_test_range[significant_drop][0]
        print(f"Significant likelihood drop starts at k_ins ≈ {drop_point:.2f}")
    
    # Test 2: Parameter correlation analysis
    print(f"\n5. Analyzing parameter correlations...")
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(samples_flat.T)
    
    # Find correlations with k_ins
    k_ins_correlations = corr_matrix[k_ins_idx, :]
    print(f"Correlations with k_ins:")
    for i, name in enumerate(param_names):
        if i != k_ins_idx:
            print(f"  {name}: {k_ins_correlations[i]:.3f}")
    
    # Strongest correlation
    strongest_corr_idx = np.argmax(np.abs(k_ins_correlations))
    strongest_corr_val = k_ins_correlations[strongest_corr_idx]
    print(f"  Strongest correlation: {param_names[strongest_corr_idx]} ({strongest_corr_val:.3f})")
    
    # Test 3: Surrogate model behavior at low k_ins
    print(f"\n6. Testing surrogate model behavior...")
    
    try:
        surrogate = FullSurrogateModel.load_model("outputs/edmund1/full_surrogate_model.pkl")
        
        # Test predictions at very low k_ins
        test_params_low = posterior_mean.copy()
        test_params_low[k_ins_idx] = 0.5
        
        test_params_normal = posterior_mean.copy()
        test_params_normal[k_ins_idx] = 5.0
        
        y_pred_low, _, _, unc_low = surrogate.predict_temperature_curves(test_params_low.reshape(1, -1))
        y_pred_normal, _, _, unc_normal = surrogate.predict_temperature_curves(test_params_normal.reshape(1, -1))
        
        print(f"Surrogate predictions:")
        print(f"  k_ins = 0.5: prediction range [{y_pred_low[0].min():.6f}, {y_pred_low[0].max():.6f}]")
        print(f"  k_ins = 5.0: prediction range [{y_pred_normal[0].min():.6f}, {y_pred_normal[0].max():.6f}]")
        print(f"  Uncertainty at k_ins = 0.5: max = {unc_low[0].max():.6f}")
        print(f"  Uncertainty at k_ins = 5.0: max = {unc_normal[0].max():.6f}")
        
        # Check if surrogate uncertainty is very high at low k_ins
        if unc_low[0].max() > 10 * unc_normal[0].max():
            print(f"  WARNING: Surrogate uncertainty is very high at low k_ins!")
            
    except Exception as e:
        print(f"  Could not load surrogate model: {e}")
    
    # Test 4: Prior influence analysis
    print(f"\n7. Analyzing prior influence...")
    
    # Check if the prior is uniform
    k_ins_prior_type = param_defs[k_ins_idx]['type']
    print(f"k_ins prior type: {k_ins_prior_type}")
    
    if k_ins_prior_type == 'uniform':
        k_ins_prior_low = param_defs[k_ins_idx]['low']
        k_ins_prior_high = param_defs[k_ins_idx]['high']
        print(f"k_ins prior range: [{k_ins_prior_low}, {k_ins_prior_high}]")
        
        # Check if posterior is hitting the prior boundary
        posterior_5th = np.percentile(k_ins_values, 5)
        posterior_95th = np.percentile(k_ins_values, 95)
        
        print(f"Posterior 5th percentile: {posterior_5th:.3f}")
        print(f"Posterior 95th percentile: {posterior_95th:.3f}")
        
        if posterior_5th < (k_ins_prior_low + 0.1):
            print(f"  WARNING: Posterior is hitting lower prior boundary!")
    
    # Test 5: MCMC convergence analysis
    print(f"\n8. Checking MCMC convergence...")
    
    # Check for stuck chains
    if len(samples_full.shape) == 3:
        n_chains = samples_full.shape[1]
        chain_means = []
        for i in range(n_chains):
            chain_means.append(samples_full[:, i, k_ins_idx].mean())
        
        chain_std = np.std(chain_means)
        print(f"k_ins mean across {n_chains} chains: {np.mean(chain_means):.3f} ± {chain_std:.3f}")
        
        if chain_std > 0.5:
            print(f"  WARNING: High variance in k_ins across chains - possible convergence issues!")
    
    # Test 6: Likelihood vs log-posterior comparison
    print(f"\n9. Comparing likelihood vs log-posterior...")
    
    # Take samples near the boundary and compute pure likelihood
    boundary_samples = samples_flat[k_ins_values < 1.5]
    boundary_log_pdf = log_pdf_flat[k_ins_values < 1.5]
    
    if len(boundary_samples) > 0:
        boundary_likelihoods = []
        for params in boundary_samples[:10]:  # Test first 10
            ll = log_likelihood_full(params.reshape(1, -1), y_obs_interp)[0]
            boundary_likelihoods.append(ll)
        
        boundary_likelihoods = np.array(boundary_likelihoods)
        boundary_log_pdf_subset = boundary_log_pdf[:10]
        
        print(f"Boundary samples (k_ins < 1.5):")
        print(f"  Log-posterior range: [{boundary_log_pdf_subset.min():.3f}, {boundary_log_pdf_subset.max():.3f}]")
        print(f"  Pure likelihood range: [{boundary_likelihoods.min():.3f}, {boundary_likelihoods.max():.3f}]")
        print(f"  Prior contribution (approx): {boundary_log_pdf_subset.mean() - boundary_likelihoods.mean():.3f}")
    
    # Create diagnostic plots
    print(f"\n10. Creating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: k_ins histogram
    axes[0, 0].hist(k_ins_values, bins=50, alpha=0.7, density=True)
    axes[0, 0].axvline(k_ins_prior_low, color='red', linestyle='--', label='Prior lower bound')
    axes[0, 0].axvline(k_ins_prior_high, color='red', linestyle='--', label='Prior upper bound')
    axes[0, 0].set_xlabel('k_ins')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('k_ins Posterior Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: k_ins vs k_sample correlation
    axes[0, 1].scatter(k_sample_values, k_ins_values, alpha=0.1, s=1)
    axes[0, 1].set_xlabel('k_sample')
    axes[0, 1].set_ylabel('k_ins')
    axes[0, 1].set_title('k_ins vs k_sample Correlation')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Likelihood surface
    axes[1, 0].plot(k_ins_test_range, ll_values)
    axes[1, 0].axvline(k_ins_prior_low, color='red', linestyle='--', label='Prior lower bound')
    axes[1, 0].set_xlabel('k_ins')
    axes[1, 0].set_ylabel('Log-likelihood')
    axes[1, 0].set_title('Likelihood Surface (other params at posterior mean)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Trace plot for k_ins
    if len(samples_full.shape) == 3:
        trace_data = samples_full[:, 0, k_ins_idx]  # First chain
    else:
        trace_data = samples_flat[:, k_ins_idx]
    
    axes[1, 1].plot(trace_data, alpha=0.7)
    axes[1, 1].axhline(k_ins_prior_low, color='red', linestyle='--', label='Prior lower bound')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('k_ins')
    axes[1, 1].set_title('k_ins Trace Plot')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('k_ins_boundary_debug.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("Potential causes for k_ins hitting lower bound:")
    
    if near_lower_bound.mean() > 0.1:
        print("✓ HIGH: Significant fraction of samples near lower bound")
    else:
        print("✗ LOW: Few samples near lower bound")
    
    if ll_values[0] < (ll_values.max() - 10):
        print("✓ HIGH: Likelihood drops significantly below k_ins = 1")
    else:
        print("✗ LOW: Likelihood remains high below k_ins = 1")
    
    if abs(strongest_corr_val) > 0.5:
        print(f"✓ HIGH: Strong correlation with {param_names[strongest_corr_idx]} ({strongest_corr_val:.3f})")
    else:
        print("✗ LOW: No strong parameter correlations")
    
    if posterior_5th < (k_ins_prior_low + 0.1):
        print("✓ HIGH: Posterior hitting prior boundary")
    else:
        print("✗ LOW: Posterior not hitting prior boundary")
    
    print(f"\nRecommendations:")
    print("1. If likelihood drops sharply: Consider extending prior range or investigating surrogate")
    print("2. If strong correlations: Check parameter degeneracies")
    print("3. If hitting boundary: Consider wider prior or different parameterization")
    print("4. If convergence issues: Run longer chains or adjust sampler settings")
    
    return {
        'k_ins_values': k_ins_values,
        'k_ins_test_range': k_ins_test_range,
        'll_values': ll_values,
        'correlations': k_ins_correlations,
        'near_boundary_fraction': near_lower_bound.mean()
    }

if __name__ == "__main__":
    debug_k_ins_boundary() 