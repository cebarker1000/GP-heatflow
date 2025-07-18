#!/usr/bin/env python3
"""
Diagnostic script to check surrogate model predictions vs experimental data
for Edmund MCMC likelihood calculation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_surrogate_models import FullSurrogateModel
from uqpy_MCMC_edmund import load_experimental_data, interpolate_to_surrogate_grid

def load_edmund_surrogate():
    """Load the Edmund-specific surrogate model"""
    surrogate = FullSurrogateModel.load_model("outputs/edmund1/full_surrogate_model.pkl")
    return surrogate

def main():
    print("=" * 60)
    print("EDMUND LIKELIHOOD DIAGNOSTIC")
    print("=" * 60)
    
    # Load experimental data
    print("\n1. Loading experimental data...")
    y_obs, exp_time = load_experimental_data()
    y_obs_interp = interpolate_to_surrogate_grid(y_obs, exp_time)
    
    print(f"Experimental data shape: {y_obs.shape}")
    print(f"Interpolated data shape: {y_obs_interp.shape}")
    print(f"Experimental data range: {y_obs.min():.6f} to {y_obs.max():.6f}")
    print(f"Interpolated data range: {y_obs_interp.min():.6f} to {y_obs_interp.max():.6f}")
    print(f"Interpolated data std: {y_obs_interp.std():.6f}")
    
    # Load surrogate model
    print("\n2. Loading surrogate model...")
    surrogate = load_edmund_surrogate()
    
    # Test with best fit parameters from Edmund MCMC
    print("\n3. Testing surrogate predictions...")
    best_fit = np.array([     
        2.9e-6,    # d_sample
        7072500,   # rho_cv_sample
        2621310,   # rho_cv_ins
        4.0e-6,    # d_ins_pside
        4.2e-6,    # d_ins_oside
        7.0e-6,    # fwhm
        50.0,      # k_sample (midpoint of uniform range)
        55.0,      # k_ins (midpoint of uniform range)
    ])
    
    print(f"Best fit parameters: {best_fit}")
    
    # Get prediction
    y_pred, fpca_coeffs, fpca_uncertainties, curve_uncertainties = surrogate.predict_temperature_curves(best_fit.reshape(1, -1))
    
    print(f"Prediction shape: {y_pred.shape}")
    print(f"Prediction range: {y_pred.min():.6f} to {y_pred.max():.6f}")
    print(f"Prediction std: {y_pred.std():.6f}")
    print(f"FPCA coefficients: {fpca_coeffs[0]}")
    
    # Calculate residuals
    residuals = y_pred[0] - y_obs_interp
    print(f"\nResiduals:")
    print(f"  Mean: {residuals.mean():.6f}")
    print(f"  Std: {residuals.std():.6f}")
    print(f"  Min: {residuals.min():.6f}")
    print(f"  Max: {residuals.max():.6f}")
    print(f"  RMS: {np.sqrt(np.mean(residuals**2)):.6f}")
    
    # Calculate likelihood with current settings
    SENSOR_VARIANCE = 1e-4
    log_like = -0.5 * np.sum(residuals**2 / SENSOR_VARIANCE + np.log(2 * np.pi * SENSOR_VARIANCE))
    print(f"\nLog-likelihood with SENSOR_VARIANCE={SENSOR_VARIANCE}: {log_like:.3f}")
    
    # Try different sensor variances
    print(f"\nLog-likelihood with different sensor variances:")
    for var in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        log_like = -0.5 * np.sum(residuals**2 / var + np.log(2 * np.pi * var))
        print(f"  Variance {var:.0e}: {log_like:.3f}")
    
    # Calculate optimal variance (maximum likelihood estimate)
    optimal_var = np.mean(residuals**2)
    optimal_log_like = -0.5 * len(residuals) * (1 + np.log(2 * np.pi * optimal_var))
    print(f"\nOptimal variance (MLE): {optimal_var:.2e}")
    print(f"Optimal log-likelihood: {optimal_log_like:.3f}")
    
    # Plot comparison
    print("\n4. Creating diagnostic plots...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Time series comparison
    time_grid = np.linspace(0, 8.5e-6, 50)
    ax1.plot(time_grid, y_obs_interp, 'b-', label='Experimental (interpolated)', linewidth=2)
    ax1.plot(time_grid, y_pred[0], 'r--', label='Surrogate prediction', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Normalized Temperature')
    ax1.set_title('Experimental vs Surrogate Prediction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax2.plot(time_grid, residuals, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Residual (Prediction - Experimental)')
    ax2.set_title(f'Residuals (RMS = {np.sqrt(np.mean(residuals**2)):.6f})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("likelihood_diagnostic_edmund.png", dpi=300, bbox_inches="tight")
    print("Diagnostic plot saved to likelihood_diagnostic_edmund.png")
    plt.show()
    
    # Test a few more parameter sets
    print("\n5. Testing additional parameter sets...")
    
    # Test with parameter means from priors
    param_defs = surrogate.param_ranges
    param_names = surrogate.parameter_names
    
    print(f"Parameter names: {param_names}")
    print(f"Parameter ranges: {param_defs}")
    
    # Create a few test parameter sets
    test_params = []
    
    # Best fit
    test_params.append(("Best fit", best_fit))
    
    # Mid-range values
    mid_params = []
    for name in param_names:
        if name in param_defs:
            low, high = param_defs[name]
            mid_params.append((low + high) / 2)
        else:
            mid_params.append(0.0)  # fallback
    test_params.append(("Mid-range", np.array(mid_params)))
    
    # Test each set
    for name, params in test_params:
        print(f"\nTesting {name} parameters...")
        y_pred_test, _, _, _ = surrogate.predict_temperature_curves(params.reshape(1, -1))
        residuals_test = y_pred_test[0] - y_obs_interp
        rms_test = np.sqrt(np.mean(residuals_test**2))
        log_like_test = -0.5 * np.sum(residuals_test**2 / SENSOR_VARIANCE + np.log(2 * np.pi * SENSOR_VARIANCE))
        
        print(f"  RMS: {rms_test:.6f}")
        print(f"  Log-likelihood: {log_like_test:.3f}")
        print(f"  Prediction range: {y_pred_test.min():.6f} to {y_pred_test.max():.6f}")

if __name__ == "__main__":
    main() 