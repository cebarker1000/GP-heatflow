#!/usr/bin/env python3
"""
Investigate the likelihood discrepancy between posterior mean and MCMC mean.
"""

import numpy as np
import pandas as pd
from train_surrogate_models import FullSurrogateModel
from uqpy_MCMC_edmund import log_likelihood_full, load_experimental_data, interpolate_to_surrogate_grid

def investigate_likelihood_discrepancy():
    """Investigate the likelihood discrepancy."""
    
    print("=" * 60)
    print("LIKELIHOOD DISCREPANCY INVESTIGATION")
    print("=" * 60)
    
    # Load experimental data
    print("\n1. Loading experimental data...")
    y_obs, exp_time = load_experimental_data()
    y_obs_interp = interpolate_to_surrogate_grid(y_obs, exp_time)
    
    # Load MCMC results to get actual samples
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
    
    print(f"MCMC samples shape: {samples_flat.shape}")
    print(f"Log PDF values shape: {log_pdf_flat.shape}")
    
    # Get parameter names
    from analysis.config_utils import get_param_defs_from_config
    param_defs = get_param_defs_from_config(config_path="configs/distributions_edmund.yaml")
    param_names = [param_def['name'] for param_def in param_defs]
    
    # Find parameter indices
    k_ins_idx = param_names.index('k_ins')
    k_sample_idx = param_names.index('k_sample')
    
    # Posterior mean parameters from your new run
    posterior_mean = np.array([
        2.909e-06,  # d_sample
        7.072e+06,  # rho_cv_sample
        2.622e+06,  # rho_cv_ins
        4.002e-06,  # d_ins_pside
        4.202e-06,  # d_ins_oside
        7.003e-06,  # fwhm
        39.0,       # k_sample
        9.345,      # k_ins
    ])
    
    print(f"\n3. Analyzing MCMC samples...")
    
    # Compute actual posterior mean from MCMC samples
    mcmc_posterior_mean = samples_flat.mean(axis=0)
    print(f"Actual MCMC posterior mean: {mcmc_posterior_mean}")
    print(f"Your reported posterior mean: {posterior_mean}")
    
    # Check if they match
    mean_diff = np.abs(mcmc_posterior_mean - posterior_mean)
    print(f"Mean difference: {mean_diff}")
    print(f"Max difference: {mean_diff.max():.2e}")
    
    # Test 1: Check if the issue is with the posterior mean calculation
    print(f"\n4. Testing likelihood of actual MCMC posterior mean...")
    ll_actual_mean = log_likelihood_full(mcmc_posterior_mean.reshape(1, -1), y_obs_interp)
    print(f"Actual MCMC posterior mean log-likelihood: {ll_actual_mean[0]:.3f}")
    print(f"Your reported posterior mean log-likelihood: 103.998")
    print(f"Difference: {ll_actual_mean[0] - 103.998:.3f}")
    
    # Test 2: Check if log_pdf_values includes prior contribution
    print(f"\n5. Checking if log_pdf_values includes prior contribution...")
    
    # The log_pdf_values might be log-posterior (likelihood + prior) rather than just likelihood
    # Let's check by looking at the distribution of values
    print(f"Log PDF statistics:")
    print(f"  Mean: {log_pdf_flat.mean():.3f}")
    print(f"  Std:  {log_pdf_flat.std():.3f}")
    print(f"  Min:  {log_pdf_flat.min():.3f}")
    print(f"  Max:  {log_pdf_flat.max():.3f}")
    
    # Test 3: Check if the issue is with the likelihood function
    print(f"\n6. Testing likelihood function consistency...")
    
    # Take a few random samples and compute their likelihood manually
    n_test = 10
    test_indices = np.random.choice(len(samples_flat), n_test, replace=False)
    
    print(f"Testing {n_test} random MCMC samples:")
    print(f"{'Sample':<6} {'k_ins':<8} {'k_sample':<10} {'MCMC LL':<12} {'Manual LL':<12} {'Diff':<10}")
    print("-" * 70)
    
    for i, idx in enumerate(test_indices):
        params = samples_flat[idx]
        mcmc_ll = log_pdf_flat[idx]
        manual_ll = log_likelihood_full(params.reshape(1, -1), y_obs_interp)[0]
        diff = mcmc_ll - manual_ll
        
        print(f"{i+1:<6} {params[k_ins_idx]:<8.2f} {params[k_sample_idx]:<10.2f} "
              f"{mcmc_ll:<12.2f} {manual_ll:<12.2f} {diff:<10.2f}")
    
    # Test 4: Check if the issue is with the experimental data
    print(f"\n7. Testing experimental data consistency...")
    
    # Check if the MCMC is using different experimental data
    y_obs_mcmc, exp_time_mcmc = load_experimental_data()
    y_obs_interp_mcmc = interpolate_to_surrogate_grid(y_obs_mcmc, exp_time_mcmc)
    
    print(f"Manual experimental data shape: {y_obs_interp.shape}")
    print(f"MCMC experimental data shape:  {y_obs_interp_mcmc.shape}")
    print(f"Data difference (MCMC - Manual): {np.mean(np.abs(y_obs_interp_mcmc - y_obs_interp)):.6f}")
    
    # Test with MCMC experimental data
    ll_mcmc_data = log_likelihood_full(posterior_mean.reshape(1, -1), y_obs_interp_mcmc)
    print(f"Likelihood with MCMC data: {ll_mcmc_data[0]:.3f}")
    print(f"Likelihood with manual data: 103.998")
    print(f"Difference (MCMC data - Manual data): {ll_mcmc_data[0] - 103.998:.3f}")
    
    # Test 5: Check if the issue is with the surrogate model
    print(f"\n8. Testing surrogate model consistency...")
    
    # Load surrogate model
    surrogate = FullSurrogateModel.load_model("outputs/edmund1/full_surrogate_model.pkl")
    
    # Test predictions
    y_pred, _, _, curve_unc = surrogate.predict_temperature_curves(posterior_mean.reshape(1, -1))
    y_pred = y_pred[0]
    curve_unc = curve_unc[0]
    
    print(f"Surrogate prediction range: {y_pred.min():.6f} to {y_pred.max():.6f}")
    print(f"Surrogate uncertainty range: {curve_unc.min():.6f} to {curve_unc.max():.6f}")
    
    # Manual likelihood computation
    SENSOR_VARIANCE = 0.0012
    INCLUDE_SURROGATE_UNCERT = True
    
    def _gaussian_loglike(y_pred, y_obs, sigma2):
        resid = y_pred - y_obs
        return -0.5 * np.sum(resid ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
    
    if INCLUDE_SURROGATE_UNCERT:
        sigma2 = SENSOR_VARIANCE + curve_unc**2
    else:
        sigma2 = SENSOR_VARIANCE
    
    ll_manual = _gaussian_loglike(y_pred, y_obs_interp, sigma2)
    print(f"Manual likelihood computation: {ll_manual:.3f}")
    print(f"Function likelihood computation: 103.998")
    print(f"Difference (Manual - Function): {ll_manual - 103.998:.3f}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("Possible explanations for the discrepancy:")
    print("1. log_pdf_values includes prior contribution (most likely)")
    print("2. Different experimental data processing")
    print("3. Different surrogate model or parameter ordering")
    print("4. Bug in likelihood function implementation")
    
    return {
        'mcmc_posterior_mean': mcmc_posterior_mean,
        'reported_posterior_mean': posterior_mean,
        'll_actual_mean': ll_actual_mean[0],
        'll_reported_mean': 103.998
    }

if __name__ == "__main__":
    investigate_likelihood_discrepancy() 