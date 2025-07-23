#!/usr/bin/env python3
"""
Test script to check if the surrogate model behaves correctly with different k_ins values.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from train_surrogate_models import FullSurrogateModel
from analysis.config_utils import get_param_defs_from_config

def test_surrogate_k_ins():
    """Test the surrogate model with different k_ins values."""
    
    print("=" * 60)
    print("SURROGATE MODEL K_INS TEST")
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
    
    print(f"Experimental data shape: {y_obs_interp.shape}")
    print(f"Experimental data range: {y_obs_interp.min():.6f} to {y_obs_interp.max():.6f}")
    
    # Get parameter definitions
    param_defs = get_param_defs_from_config(config_path="configs/distributions_edmund.yaml")
    param_names = [param_def['name'] for param_def in param_defs]
    
    print(f"Parameter names: {param_names}")
    print(f"Surrogate parameter names: {surrogate.parameter_names}")
    
    # Test with different k_ins values
    print("\n3. Testing with different k_ins values...")
    
    # Base parameters (use Edmund best fit as starting point)
    base_params = np.array([     
        2.9e-6,    # d_sample
        7072500,   # rho_cv_sample
        2621310,   # rho_cv_ins
        4.0e-6,    # d_ins_pside
        4.2e-6,    # d_ins_oside
        7.0e-6,    # fwhm
        50.0,      # k_sample
        14.0,      # k_ins (will be varied)
    ])
    
    # Test k_ins values
    k_ins_values = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    
    results = []
    
    for k_ins in k_ins_values:
        print(f"\nTesting k_ins = {k_ins} W/(m·K)...")
        
        # Update k_ins in base parameters
        test_params = base_params.copy()
        test_params[7] = k_ins  # k_ins is at index 7
        
        # Get prediction
        y_pred, fpca_coeffs, fpca_uncertainties, curve_uncertainties = surrogate.predict_temperature_curves(test_params.reshape(1, -1))
        y_pred = y_pred[0]  # Remove batch dimension
        
        # Calculate fit quality
        residuals = y_pred - y_obs_interp
        rms = np.sqrt(np.mean(residuals**2))
        log_like = -0.5 * np.sum(residuals**2 / 0.0012 + np.log(2 * np.pi * 0.0012))
        
        print(f"  RMS: {rms:.6f}")
        print(f"  Log-likelihood: {log_like:.3f}")
        print(f"  Prediction range: {y_pred.min():.6f} to {y_pred.max():.6f}")
        
        results.append({
            'k_ins': k_ins,
            'rms': rms,
            'log_like': log_like,
            'prediction': y_pred,
            'residuals': residuals
        })

    # -------------------------------------------------------------
    # Quick consistency check – same likelihood formula as MCMC
    # -------------------------------------------------------------
    print("\nQuick check with MCMC-style likelihood (sensor σ + GP σ):")

    SENSOR_VARIANCE = 0.0012

    def _gaussian_loglike(y_pred, y_obs, sigma2):
        resid = y_pred - y_obs
        return -0.5 * np.sum(resid ** 2 / sigma2 + np.log(2 * np.pi * sigma2))

    for k_test in [1.0, 4.0, 10.0, 15.0]:
        p_test = base_params.copy()
        p_test[7] = k_test  # update k_ins
        y_pred_test, _, _, curve_unc_test = surrogate.predict_temperature_curves(p_test.reshape(1, -1))
        y_pred_test = y_pred_test[0]
        sigma2_test = SENSOR_VARIANCE + curve_unc_test[0] ** 2
        ll_test = _gaussian_loglike(y_pred_test, y_obs_interp, sigma2_test)
        print(f"  k_ins = {k_test:4.1f}  ->  logL = {ll_test:7.1f}")

    # Create plots
    print("\n4. Creating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: RMS vs k_ins
    k_ins_array = [r['k_ins'] for r in results]
    rms_array = [r['rms'] for r in results]
    log_like_array = [r['log_like'] for r in results]
    
    axes[0, 0].plot(k_ins_array, rms_array, 'bo-')
    axes[0, 0].set_xlabel('k_ins (W/(m·K))')
    axes[0, 0].set_ylabel('RMS')
    axes[0, 0].set_title('RMS vs k_ins')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Log-likelihood vs k_ins
    axes[0, 1].plot(k_ins_array, log_like_array, 'ro-')
    axes[0, 1].set_xlabel('k_ins (W/(m·K))')
    axes[0, 1].set_ylabel('Log-likelihood')
    axes[0, 1].set_title('Log-likelihood vs k_ins')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Predictions for different k_ins values
    for i, result in enumerate(results):
        if i % 2 == 0:  # Plot every other to avoid clutter
            axes[1, 0].plot(surrogate.time_grid, result['prediction'], 
                           label=f'k_ins = {result["k_ins"]} W/(m·K)', alpha=0.7)
    
    axes[1, 0].plot(surrogate.time_grid, y_obs_interp, 'k-', linewidth=2, label='Experimental')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Normalized Temperature')
    axes[1, 0].set_title('Predictions vs Experimental Data')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Residuals for different k_ins values
    for i, result in enumerate(results):
        if i % 2 == 0:  # Plot every other to avoid clutter
            axes[1, 1].plot(surrogate.time_grid, result['residuals'], 
                           label=f'k_ins = {result["k_ins"]} W/(m·K)', alpha=0.7)
    
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Experimental Data')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("surrogate_k_ins_test.png", dpi=300, bbox_inches='tight')
    print("Test plots saved to surrogate_k_ins_test.png")
    plt.show()
    
    # Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    # Find best k_ins
    best_idx = np.argmax(log_like_array)
    best_k_ins = k_ins_array[best_idx]
    best_rms = rms_array[best_idx]
    
    print(f"Best k_ins value: {best_k_ins} W/(m·K)")
    print(f"Best RMS: {best_rms:.6f}")
    print(f"Best log-likelihood: {log_like_array[best_idx]:.3f}")
    
    # Check if low k_ins values give better fits
    low_k_ins_rms = [r['rms'] for r in results if r['k_ins'] <= 5]
    high_k_ins_rms = [r['rms'] for r in results if r['k_ins'] >= 20]
    
    if len(low_k_ins_rms) > 0 and len(high_k_ins_rms) > 0:
        low_avg = np.mean(low_k_ins_rms)
        high_avg = np.mean(high_k_ins_rms)
        
        print(f"\nLow k_ins (≤5) average RMS: {low_avg:.6f}")
        print(f"High k_ins (≥20) average RMS: {high_avg:.6f}")
        
        if low_avg < high_avg:
            print("✅ Low k_ins values give better fits - this explains the MCMC behavior")
        else:
            print("❌ High k_ins values give better fits - MCMC should not be pulled to low values")
    
    # Check if the surrogate model is working correctly
    print(f"\nSurrogate model behavior:")
    print(f"  - Predictions are reasonable: {all(r['prediction'].min() >= -1 and r['prediction'].max() <= 2 for r in results)}")
    print(f"  - Predictions vary with k_ins: {len(set([r['prediction'].mean() for r in results])) > 1}")
    
    return results

if __name__ == "__main__":
    test_surrogate_k_ins() 