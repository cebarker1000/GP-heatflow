#!/usr/bin/env python3
"""
Test the log-likelihood of the posterior mean parameters.
"""

import numpy as np
import pandas as pd
from train_surrogate_models import FullSurrogateModel
from uqpy_MCMC_edmund import log_likelihood_full, load_experimental_data, interpolate_to_surrogate_grid

def test_posterior_mean():
    """Test the log-likelihood of the posterior mean parameters."""
    
    print("=" * 60)
    print("POSTERIOR MEAN LIKELIHOOD TEST")
    print("=" * 60)
    
    # Load experimental data
    print("\n1. Loading experimental data...")
    y_obs, exp_time = load_experimental_data()
    y_obs_interp = interpolate_to_surrogate_grid(y_obs, exp_time)
    
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
    
    print(f"Posterior mean parameters: {posterior_mean}")
    
    # Test the log-likelihood
    print(f"\n2. Computing log-likelihood...")
    ll_posterior_mean = log_likelihood_full(posterior_mean.reshape(1, -1), y_obs_interp)
    
    print(f"Log-likelihood of posterior mean: {ll_posterior_mean[0]:.3f}")
    print(f"MCMC mean log-likelihood: 130.007")
    print(f"Difference: {ll_posterior_mean[0] - 130.007:.3f}")
    
    # Test a few other parameter sets for comparison
    print(f"\n3. Testing other parameter sets...")
    
    # Test with the best fit from our earlier systematic test
    best_systematic = np.array([
        2.9e-6,     # d_sample
        7072500,    # rho_cv_sample
        2621310,    # rho_cv_ins
        4.0e-6,     # d_ins_pside
        4.2e-6,     # d_ins_oside
        7.0e-6,     # fwhm
        50.0,       # k_sample
        13.0,       # k_ins (best from systematic test)
    ])
    
    ll_best_systematic = log_likelihood_full(best_systematic.reshape(1, -1), y_obs_interp)
    print(f"Best systematic (k_ins=13.0): {ll_best_systematic[0]:.3f}")
    
    # Test with low k_ins
    low_k_ins = posterior_mean.copy()
    low_k_ins[7] = 1.0  # k_ins = 1.0
    ll_low_k_ins = log_likelihood_full(low_k_ins.reshape(1, -1), y_obs_interp)
    print(f"Low k_ins (k_ins=1.0): {ll_low_k_ins[0]:.3f}")
    
    # Test with high k_ins
    high_k_ins = posterior_mean.copy()
    high_k_ins[7] = 30.0  # k_ins = 30.0
    ll_high_k_ins = log_likelihood_full(high_k_ins.reshape(1, -1), y_obs_interp)
    print(f"High k_ins (k_ins=30.0): {ll_high_k_ins[0]:.3f}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"Posterior mean log-likelihood: {ll_posterior_mean[0]:.3f}")
    print(f"MCMC mean log-likelihood: 130.007")
    print(f"Difference: {ll_posterior_mean[0] - 130.007:.3f}")
    
    if abs(ll_posterior_mean[0] - 130.007) < 5.0:
        print("✅ Posterior mean likelihood is close to MCMC mean - this is reasonable!")
    else:
        print("❌ Large difference - this suggests an issue!")
    
    print(f"\nComparison with other values:")
    print(f"  Best systematic (k_ins=13.0): {ll_best_systematic[0]:.3f}")
    print(f"  Low k_ins (k_ins=1.0): {ll_low_k_ins[0]:.3f}")
    print(f"  High k_ins (k_ins=30.0): {ll_high_k_ins[0]:.3f}")
    
    return {
        'posterior_mean_ll': ll_posterior_mean[0],
        'best_systematic_ll': ll_best_systematic[0],
        'low_k_ins_ll': ll_low_k_ins[0],
        'high_k_ins_ll': ll_high_k_ins[0]
    }

if __name__ == "__main__":
    test_posterior_mean() 