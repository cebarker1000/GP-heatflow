#!/usr/bin/env python3
"""
Comprehensive diagnostic and fix for Edmund MCMC k_ins issue.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from train_surrogate_models import FullSurrogateModel
from analysis.config_utils import get_param_defs_from_config

def comprehensive_diagnostic():
    """Run comprehensive diagnostics to understand the k_ins issue."""
    
    print("=" * 60)
    print("COMPREHENSIVE EDMUND MCMC DIAGNOSTIC")
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
    
    # Flatten samples
    if len(samples_full.shape) == 3:
        samples_flat = samples_full.reshape(-1, samples_full.shape[2])
    else:
        samples_flat = samples_full
    
    # Get parameter names
    param_defs = get_param_defs_from_config(config_path="configs/distributions_edmund.yaml")
    param_names = [param_def['name'] for param_def in param_defs]
    
    # Find parameter indices
    k_ins_idx = param_names.index('k_ins')
    k_sample_idx = param_names.index('k_sample')
    
    # Extract k_ins values from MCMC
    k_ins_mcmc = samples_flat[:, k_ins_idx]
    k_sample_mcmc = samples_flat[:, k_sample_idx]
    
    print(f"\nMCMC k_ins statistics:")
    print(f"  Mean: {k_ins_mcmc.mean():.2f} W/(m·K)")
    print(f"  Std: {k_ins_mcmc.std():.2f} W/(m·K)")
    print(f"  Range: {k_ins_mcmc.min():.2f} to {k_ins_mcmc.max():.2f} W/(m·K)")
    
    # Test 1: Check if MCMC samples give good fits
    print(f"\n4. Testing MCMC sample fits...")
    
    # Take a random sample of MCMC results
    n_test = 100
    test_indices = np.random.choice(len(samples_flat), n_test, replace=False)
    test_samples = samples_flat[test_indices]
    
    mcmc_fits = []
    for i, params in enumerate(test_samples):
        y_pred, _, _, _ = surrogate.predict_temperature_curves(params.reshape(1, -1))
        y_pred = y_pred[0]
        
        residuals = y_pred - y_obs_interp
        rms = np.sqrt(np.mean(residuals**2))
        log_like = -0.5 * np.sum(residuals**2 / 0.0012 + np.log(2 * np.pi * 0.0012))
        
        mcmc_fits.append({
            'k_ins': params[k_ins_idx],
            'k_sample': params[k_sample_idx],
            'rms': rms,
            'log_like': log_like
        })
    
    # Sort by log-likelihood
    mcmc_fits.sort(key=lambda x: x['log_like'], reverse=True)
    
    print(f"Top 5 MCMC samples by log-likelihood:")
    for i, fit in enumerate(mcmc_fits[:5]):
        print(f"  {i+1}. k_ins={fit['k_ins']:.2f}, k_sample={fit['k_sample']:.2f}, log_like={fit['log_like']:.1f}")
    
    # Test 2: Systematic k_ins test with MCMC mean parameters
    print(f"\n5. Testing systematic k_ins values with MCMC mean parameters...")
    
    # Use MCMC mean values for other parameters
    mcmc_mean_params = samples_flat.mean(axis=0)
    
    k_ins_test_values = np.linspace(1, 30, 30)
    systematic_fits = []
    
    for k_ins in k_ins_test_values:
        test_params = mcmc_mean_params.copy()
        test_params[k_ins_idx] = k_ins
        
        y_pred, _, _, _ = surrogate.predict_temperature_curves(test_params.reshape(1, -1))
        y_pred = y_pred[0]
        
        residuals = y_pred - y_obs_interp
        rms = np.sqrt(np.mean(residuals**2))
        log_like = -0.5 * np.sum(residuals**2 / 0.0012 + np.log(2 * np.pi * 0.0012))
        
        systematic_fits.append({
            'k_ins': k_ins,
            'rms': rms,
            'log_like': log_like
        })
    
    # Find best k_ins
    best_systematic = max(systematic_fits, key=lambda x: x['log_like'])
    print(f"Best k_ins (systematic test): {best_systematic['k_ins']:.2f} W/(m·K)")
    print(f"Best log-likelihood: {best_systematic['log_like']:.1f}")
    
    # Test 3: Check if there are parameter correlations causing issues
    print(f"\n6. Analyzing parameter correlations...")
    
    # Calculate correlations between all parameters
    correlations = np.corrcoef(samples_flat.T)
    
    print(f"Correlation between k_ins and other parameters:")
    for i, name in enumerate(param_names):
        if i != k_ins_idx:
            corr = correlations[k_ins_idx, i]
            print(f"  {name}: {corr:.3f}")
    
    # Test 4: Check if the issue is with the likelihood function
    print(f"\n7. Testing likelihood function consistency...")
    
    # Test the exact likelihood function used in MCMC
    from uqpy_MCMC_edmund import log_likelihood_full, SENSOR_VARIANCE
    
    # Test with a few parameter sets
    test_params = np.array([
        [mcmc_mean_params],  # MCMC mean
        [best_systematic['k_ins']] + list(mcmc_mean_params[1:]),  # Best k_ins
        [1.0] + list(mcmc_mean_params[1:]),  # Low k_ins
        [15.0] + list(mcmc_mean_params[1:]),  # High k_ins
    ])
    
    for i, params in enumerate(test_params):
        ll = log_likelihood_full(params, y_obs_interp)
        print(f"  Test {i+1}: k_ins={params[0, k_ins_idx]:.2f}, log_like={ll[0]:.1f}")
    
    # Create comprehensive plots
    print(f"\n8. Creating comprehensive diagnostic plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: MCMC k_ins distribution
    axes[0, 0].hist(k_ins_mcmc, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(1, color='red', linestyle='--', label='Prior lower bound')
    axes[0, 0].axvline(30, color='red', linestyle='--', label='Prior upper bound')
    axes[0, 0].axvline(best_systematic['k_ins'], color='green', linestyle='-', label='Best systematic')
    axes[0, 0].set_xlabel('k_ins (W/(m·K))')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('MCMC k_ins Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Systematic k_ins test
    k_ins_sys = [f['k_ins'] for f in systematic_fits]
    log_like_sys = [f['log_like'] for f in systematic_fits]
    axes[0, 1].plot(k_ins_sys, log_like_sys, 'bo-')
    axes[0, 1].axvline(k_ins_mcmc.mean(), color='red', linestyle='--', label='MCMC mean')
    axes[0, 1].set_xlabel('k_ins (W/(m·K))')
    axes[0, 1].set_ylabel('Log-likelihood')
    axes[0, 1].set_title('Systematic k_ins Test')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: MCMC sample fits
    k_ins_mcmc_fits = [f['k_ins'] for f in mcmc_fits]
    log_like_mcmc_fits = [f['log_like'] for f in mcmc_fits]
    axes[0, 2].scatter(k_ins_mcmc_fits, log_like_mcmc_fits, alpha=0.6, s=10)
    axes[0, 2].set_xlabel('k_ins (W/(m·K))')
    axes[0, 2].set_ylabel('Log-likelihood')
    axes[0, 2].set_title('MCMC Sample Fits')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: k_sample vs k_ins correlation
    axes[1, 0].scatter(k_sample_mcmc, k_ins_mcmc, alpha=0.5, s=1)
    axes[1, 0].set_xlabel('k_sample (W/(m·K))')
    axes[1, 0].set_ylabel('k_ins (W/(m·K))')
    axes[1, 0].set_title('k_sample vs k_ins')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Parameter correlations heatmap
    im = axes[1, 1].imshow(correlations, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(param_names)))
    axes[1, 1].set_yticks(range(len(param_names)))
    axes[1, 1].set_xticklabels(param_names, rotation=45)
    axes[1, 1].set_yticklabels(param_names)
    axes[1, 1].set_title('Parameter Correlations')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Plot 6: RMS vs k_ins
    rms_sys = [f['rms'] for f in systematic_fits]
    axes[1, 2].plot(k_ins_sys, rms_sys, 'ro-')
    axes[1, 2].set_xlabel('k_ins (W/(m·K))')
    axes[1, 2].set_ylabel('RMS')
    axes[1, 2].set_title('RMS vs k_ins')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("comprehensive_edmund_diagnostic.png", dpi=300, bbox_inches='tight')
    print("Comprehensive diagnostic plot saved to comprehensive_edmund_diagnostic.png")
    plt.show()
    
    # Summary and recommendations
    print(f"\n" + "=" * 60)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 60)
    
    print(f"Problem identified: MCMC is finding k_ins ≈ {k_ins_mcmc.mean():.2f} W/(m·K)")
    print(f"but systematic testing shows best fit at k_ins ≈ {best_systematic['k_ins']:.2f} W/(m·K)")
    
    print(f"\nPossible causes:")
    print(f"1. Parameter correlations affecting MCMC exploration")
    print(f"2. MCMC not converged to true posterior")
    print(f"3. Likelihood function implementation issue")
    print(f"4. Surrogate model training data range issue")
    
    print(f"\nRecommended fixes:")
    print(f"1. Run longer MCMC chains with better mixing")
    print(f"2. Check surrogate model training data coverage")
    print(f"3. Verify likelihood function implementation")
    print(f"4. Consider reparameterization to reduce correlations")
    print(f"5. Test with different MCMC samplers (DREAM, DRAM)")

if __name__ == "__main__":
    comprehensive_diagnostic() 