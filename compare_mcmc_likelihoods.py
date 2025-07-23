#!/usr/bin/env python3
"""
Compare manually computed log-likelihood values with those saved by the MCMC.
"""

import numpy as np
import pandas as pd
from train_surrogate_models import FullSurrogateModel
from analysis.config_utils import get_param_defs_from_config

def compare_mcmc_likelihoods():
    """Compare manually computed likelihoods with MCMC saved values."""
    
    print("=" * 60)
    print("MCMC LIKELIHOOD COMPARISON")
    print("=" * 60)
    
    # Load experimental data
    print("\n1. Loading experimental data...")
    data = pd.read_csv("data/experimental/edmund_71Gpa_run1.csv")
    oside_data = data['oside'].values
    y_obs = (oside_data - oside_data[0]) / (data['temp'].max() - data['temp'].min())
    exp_time = data['time'].values
    
    # Load surrogate model
    print("\n2. Loading surrogate model...")
    surrogate = FullSurrogateModel.load_model("outputs/edmund1/full_surrogate_model.pkl")

    # Interpolate to surrogate grid
    from scipy.interpolate import interp1d
    surrogate_time_grid = surrogate.time_grid
    interp_func = interp1d(exp_time, y_obs, kind='linear', 
                           bounds_error=False, fill_value=(y_obs[0], y_obs[-1]))
    y_obs_interp = interp_func(surrogate_time_grid)
    
    # Load MCMC results
    print("\n3. Loading MCMC results...")
    mcmc_data = np.load('mcmc_results_edmund.npz')
    samples_full = mcmc_data['samples_full']
    log_pdf_values = mcmc_data['log_pdf_values']
    
    print(f"MCMC samples shape: {samples_full.shape}")
    print(f"Log PDF values shape: {log_pdf_values.shape}")
    
    # Flatten samples if needed
    if len(samples_full.shape) == 3:
        samples_flat = samples_full.reshape(-1, samples_full.shape[2])
        log_pdf_flat = log_pdf_values.reshape(-1)
    else:
        samples_flat = samples_full
        log_pdf_flat = log_pdf_values
    
    print(f"Flattened samples shape: {samples_flat.shape}")
    print(f"Flattened log PDF shape: {log_pdf_flat.shape}")
    
    # Get parameter names
    param_defs = get_param_defs_from_config(config_path="configs/distributions_edmund.yaml")
    param_names = [param_def['name'] for param_def in param_defs]
    
    # Find parameter indices
    k_ins_idx = param_names.index('k_ins')
    k_sample_idx = param_names.index('k_sample')
    
    print(f"\nParameter names: {param_names}")
    print(f"k_ins index: {k_ins_idx}")
    print(f"k_sample index: {k_sample_idx}")
    
    # Define the exact likelihood function used in MCMC
    SENSOR_VARIANCE = 0.0012
    INCLUDE_SURROGATE_UNCERT = True
    
    def _gaussian_loglike(y_pred, y_obs, sigma2):
        """Exact likelihood function from uqpy_MCMC_edmund.py"""
        resid = y_pred - y_obs
        return -0.5 * np.sum(resid ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
    
    def compute_likelihood_manual(params):
        """Compute likelihood manually using the same function as MCMC"""
        y_pred, _, _, curve_uncert = surrogate.predict_temperature_curves(params.reshape(1, -1))
        y_pred = y_pred[0]
        
        if INCLUDE_SURROGATE_UNCERT:
            sigma2 = SENSOR_VARIANCE + curve_uncert[0]**2
        else:
            sigma2 = SENSOR_VARIANCE
        
        return _gaussian_loglike(y_pred, y_obs_interp, sigma2)
    
    # Test 1: Compare a few random MCMC samples
    print(f"\n4. Testing random MCMC samples...")
    
    n_test = 10
    test_indices = np.random.choice(len(samples_flat), n_test, replace=False)
    
    print(f"{'Sample':<6} {'k_ins':<8} {'k_sample':<10} {'MCMC LL':<12} {'Manual LL':<12} {'Diff':<10}")
    print("-" * 70)
    
    for i, idx in enumerate(test_indices):
        params = samples_flat[idx]
        mcmc_ll = log_pdf_flat[idx]
        
        # Compute likelihood manually
        manual_ll = compute_likelihood_manual(params)
        
        diff = abs(mcmc_ll - manual_ll)
        
        print(f"{i+1:<6} {params[k_ins_idx]:<8.2f} {params[k_sample_idx]:<10.2f} "
              f"{mcmc_ll:<12.2f} {manual_ll:<12.2f} {diff:<10.2f}")
    
    # Test 2: Check the best MCMC samples
    print(f"\n5. Testing best MCMC samples...")
    
    # Find samples with highest log-likelihood
    best_indices = np.argsort(log_pdf_flat)[-5:][::-1]
    
    print(f"{'Rank':<6} {'k_ins':<8} {'k_sample':<10} {'MCMC LL':<12} {'Manual LL':<12} {'Diff':<10}")
    print("-" * 70)
    
    for i, idx in enumerate(best_indices):
        params = samples_flat[idx]
        mcmc_ll = log_pdf_flat[idx]
        manual_ll = compute_likelihood_manual(params)
        diff = abs(mcmc_ll - manual_ll)
        
        print(f"{i+1:<6} {params[k_ins_idx]:<8.2f} {params[k_sample_idx]:<10.2f} "
              f"{mcmc_ll:<12.2f} {manual_ll:<12.2f} {diff:<10.2f}")
    
    # Test 3: Check if there's a systematic difference
    print(f"\n6. Systematic comparison...")
    
    # Test a larger sample
    n_systematic = 100
    systematic_indices = np.random.choice(len(samples_flat), n_systematic, replace=False)
    
    mcmc_lls = log_pdf_flat[systematic_indices]
    manual_lls = []
    
    for idx in systematic_indices:
        params = samples_flat[idx]
        manual_ll = compute_likelihood_manual(params)
        manual_lls.append(manual_ll)
    
    manual_lls = np.array(manual_lls)
    differences = mcmc_lls - manual_lls
    
    print(f"Mean difference (MCMC - Manual): {differences.mean():.6f}")
    print(f"Std difference: {differences.std():.6f}")
    print(f"Max absolute difference: {np.abs(differences).max():.6f}")
    print(f"Min absolute difference: {np.abs(differences).min():.6f}")
    
    # Check if differences are significant
    significant_threshold = 0.1
    significant_diffs = np.abs(differences) > significant_threshold
    print(f"Number of significant differences (> {significant_threshold}): {significant_diffs.sum()}")
    
    if significant_diffs.sum() > 0:
        print("❌ WARNING: Significant differences found!")
        print("This suggests a bug in the MCMC likelihood computation or parameter ordering.")
    else:
        print("✅ No significant differences found.")
    
    # Test 4: Check if log_pdf_values includes prior contribution
    print(f"\n7. Checking if log_pdf_values includes prior contribution...")
    
    # The log_pdf_values might include the prior contribution
    # Let's check by computing the prior contribution
    from uqpy_MCMC_edmund import full_prior
    
    prior_contributions = []
    for idx in systematic_indices:
        params = samples_flat[idx]
        log_prior = full_prior.log_pdf(params)
        prior_contributions.append(log_prior)
    
    prior_contributions = np.array(prior_contributions)
    likelihood_only = mcmc_lls - prior_contributions
    
    differences_likelihood = likelihood_only - manual_lls
    
    print(f"Mean difference (MCMC likelihood only - Manual): {differences_likelihood.mean():.6f}")
    print(f"Std difference: {differences_likelihood.std():.6f}")
    print(f"Max absolute difference: {np.abs(differences_likelihood).max():.6f}")
    
    # Create diagnostic plot
    print(f"\n8. Creating diagnostic plot...")
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: MCMC vs Manual likelihoods
    axes[0, 0].scatter(manual_lls, mcmc_lls, alpha=0.6, s=20)
    axes[0, 0].plot([manual_lls.min(), manual_lls.max()], [manual_lls.min(), manual_lls.max()], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('Manual Log-Likelihood')
    axes[0, 0].set_ylabel('MCMC Log-Likelihood')
    axes[0, 0].set_title('MCMC vs Manual Likelihoods')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Differences histogram
    axes[0, 1].hist(differences, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.8)
    axes[0, 1].set_xlabel('Difference (MCMC - Manual)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Likelihood Differences')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: k_ins vs log-likelihood
    k_ins_values = samples_flat[systematic_indices, k_ins_idx]
    axes[1, 0].scatter(k_ins_values, manual_lls, alpha=0.6, s=20, label='Manual')
    axes[1, 0].scatter(k_ins_values, mcmc_lls, alpha=0.6, s=20, label='MCMC')
    axes[1, 0].set_xlabel('k_ins (W/(m·K))')
    axes[1, 0].set_ylabel('Log-Likelihood')
    axes[1, 0].set_title('k_ins vs Log-Likelihood')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: k_sample vs k_ins with likelihood color
    k_sample_values = samples_flat[systematic_indices, k_sample_idx]
    scatter = axes[1, 1].scatter(k_sample_values, k_ins_values, c=manual_lls, 
                                cmap='viridis', alpha=0.7, s=30)
    axes[1, 1].set_xlabel('k_sample (W/(m·K))')
    axes[1, 1].set_ylabel('k_ins (W/(m·K))')
    axes[1, 1].set_title('k_sample vs k_ins (colored by likelihood)')
    plt.colorbar(scatter, ax=axes[1, 1], label='Log-Likelihood')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("mcmc_likelihood_comparison.png", dpi=300, bbox_inches='tight')
    print("Diagnostic plot saved to mcmc_likelihood_comparison.png")
    plt.show()
    
    return {
        'differences': differences,
        'differences_likelihood': differences_likelihood,
        'manual_lls': manual_lls,
        'mcmc_lls': mcmc_lls
    }

if __name__ == "__main__":
    compare_mcmc_likelihoods() 