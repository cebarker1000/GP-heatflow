#!/usr/bin/env python3
"""
Plot MCMC results from Edmund saved outputs.
Loads samples and creates corner plots for Edmund's 8 parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.signal import correlate
from analysis.config_utils import get_param_defs_from_config

def autocorr(x, nlags=None):
    """
    Compute autocorrelation function.
    
    Parameters:
    -----------
    x : np.ndarray
        Time series
    nlags : int, optional
        Number of lags to compute
        
    Returns:
    --------
    np.ndarray
        Autocorrelation function
    """
    if nlags is None:
        nlags = len(x) - 1
    
    # Remove mean
    x_centered = x - np.mean(x)
    
    # Compute autocorrelation
    acf = correlate(x_centered, x_centered, mode='full')
    acf = acf[len(x_centered)-1:len(x_centered)-1+nlags+1]
    
    # Normalize
    acf = acf / acf[0]
    
    return acf

def compute_ess_arviz(samples, param_names, n_walkers=44):
    """
    Compute Effective Sample Size (ESS) using ArviZ with proper data structure.
    """
    try:
        import arviz as az
        
        # Check if samples are already in chain format or need reshaping
        if len(samples.shape) == 3:
            # Samples are already in format (nsamples, n_chains, dimension)
            # Need to transpose to (n_chains, nsamples, dimension) for ArviZ
            chains = samples.transpose(1, 0, 2)  # (n_chains, nsamples, dimension)
            print(f"ESS Debug: samples already in chain format, shape = {samples.shape}")
        else:
            # Samples are in flat format (nsamples * n_chains, dimension)
            # Reshape to separate chains: (n_chains, n_samples_per_chain, dimension)
            total_samples = len(samples)
            samples_per_walker = total_samples // n_walkers
            chains = samples.reshape(n_walkers, samples_per_walker, samples.shape[1])
            print(f"ESS Debug: reshaped flat samples, total_samples={total_samples}, n_walkers={n_walkers}, samples_per_walker={samples_per_walker}")
        
        print(f"ESS Debug: chains shape = {chains.shape}")
        
        # Create InferenceData with proper structure for all parameters
        posterior_dict = {}
        for i, name in enumerate(param_names):
            posterior_dict[name] = chains[:, :, i]
        
        idata = az.from_dict(posterior=posterior_dict)
        
        # Compute ESS using ArviZ's built-in method
        ess_bulk = az.ess(idata, method="bulk")
        
        ess_values = np.array([ess_bulk[name].values for name in param_names])
        
        print(f"ESS Debug: computed ESS values = {ess_values}")
        
        return ess_values
        
    except ImportError:
        print("ArviZ not available, skipping ESS calculation")
        return np.array([np.nan] * len(param_names))
    except Exception as e:
        print(f"Error computing ESS: {e}")
        return np.array([np.nan] * len(param_names))

def compute_rhat_arviz(samples, param_names, n_walkers=44):
    """
    Compute R-hat (Gelman-Rubin diagnostic) using ArviZ with proper data structure.
    """
    try:
        import arviz as az
        
        # Check if samples are already in chain format or need reshaping
        if len(samples.shape) == 3:
            # Samples are already in format (nsamples, n_chains, dimension)
            # Need to transpose to (n_chains, nsamples, dimension) for ArviZ
            chains = samples.transpose(1, 0, 2)  # (n_chains, nsamples, dimension)
        else:
            # Samples are in flat format (nsamples * n_chains, dimension)
            # Reshape to separate chains: (n_chains, n_samples_per_chain, dimension)
            total_samples = len(samples)
            samples_per_walker = total_samples // n_walkers
            chains = samples.reshape(n_walkers, samples_per_walker, samples.shape[1])
        
        # Create InferenceData with proper structure for all parameters
        posterior_dict = {}
        for i, name in enumerate(param_names):
            posterior_dict[name] = chains[:, :, i]
        
        idata = az.from_dict(posterior=posterior_dict)
        
        # Compute R-hat using ArviZ's built-in method
        rhat = az.rhat(idata)
        
        rhat_values = np.array([rhat[name].values for name in param_names])
        
        return rhat_values
        
    except ImportError:
        print("ArviZ not available, skipping R-hat calculation")
        return np.array([np.nan] * len(param_names))
    except Exception as e:
        print(f"Error computing R-hat: {e}")
        return np.array([np.nan] * len(param_names))

def analyze_nuisance_parameter_influence(samples_flat, param_names):
    """
    Analyze which nuisance parameters contribute most to k parameter uncertainty.
    
    Parameters:
    -----------
    samples_flat : np.ndarray
        Flattened samples (n_samples, n_parameters)
    param_names : list
        Names of all parameters
        
    Returns:
    --------
    dict
        Analysis results including correlations and conditional variances
    """
    # Identify nuisance and k parameters for Edmund (6 nuisance, 2 k parameters)
    nuisance_indices = list(range(6))  # First 6 parameters are nuisance: d_sample, rho_cv_sample, rho_cv_ins, d_ins_pside, d_ins_oside, fwhm
    k_indices = [6, 7]  # k_sample, k_ins
    k_names = ['k_sample', 'k_ins']
    
    nuisance_names = [param_names[i] for i in nuisance_indices]
    
    print(f"\n" + "="*60)
    print("NUISANCE PARAMETER INFLUENCE ANALYSIS (EDMUND)")
    print("="*60)
    
    # 1. Correlation Analysis
    print("\n1. CORRELATION ANALYSIS")
    print("-" * 40)
    print(f"{'Nuisance Param':<15} {'k_sample':<12} {'k_ins':<12}")
    print("-" * 50)
    
    correlations = {}
    for i, nuisance_idx in enumerate(nuisance_indices):
        nuisance_param = samples_flat[:, nuisance_idx]
        corr_row = []
        for k_idx in k_indices:
            k_param = samples_flat[:, k_idx]
            corr = np.corrcoef(nuisance_param, k_param)[0, 1]
            corr_row.append(corr)
        correlations[nuisance_names[i]] = corr_row
        print(f"{nuisance_names[i]:<15} {corr_row[0]:<12.3f} {corr_row[1]:<12.3f}")
    
    # 2. Conditional Variance Analysis
    print("\n2. CONDITIONAL VARIANCE ANALYSIS")
    print("-" * 40)
    print("Variance reduction when nuisance parameter is fixed at its mean")
    print(f"{'Nuisance Param':<15} {'k_sample':<12} {'k_ins':<12}")
    print("-" * 50)
    
    # Calculate unconditional variances
    unconditional_var = np.array([np.var(samples_flat[:, k_idx]) for k_idx in k_indices])
    
    conditional_var_reduction = {}
    for i, nuisance_idx in enumerate(nuisance_indices):
        nuisance_param = samples_flat[:, nuisance_idx]
        nuisance_mean = np.mean(nuisance_param)
        
        # Find samples where nuisance parameter is close to its mean
        # Use samples within 1 standard deviation of the mean
        nuisance_std = np.std(nuisance_param)
        mask = np.abs(nuisance_param - nuisance_mean) <= nuisance_std
        
        if np.sum(mask) > 100:  # Need sufficient samples
            conditional_var = np.array([np.var(samples_flat[mask, k_idx]) for k_idx in k_indices])
            var_reduction = (unconditional_var - conditional_var) / unconditional_var * 100
            conditional_var_reduction[nuisance_names[i]] = var_reduction
            print(f"{nuisance_names[i]:<15} {var_reduction[0]:<12.1f}% {var_reduction[1]:<12.1f}%")
        else:
            print(f"{nuisance_names[i]:<15} {'Insufficient':<12} {'samples':<12}")
    
    return {
        'correlations': correlations,
        'conditional_var_reduction': conditional_var_reduction,
        'nuisance_names': nuisance_names,
        'k_names': k_names,
        'unconditional_var': unconditional_var
    }

def plot_nuisance_influence_analysis(correlations, conditional_var_reduction, nuisance_names, k_names):
    """
    Create visualization of nuisance parameter influence analysis.
    
    Parameters:
    -----------
    correlations : dict
        Correlation coefficients between nuisance and k parameters
    conditional_var_reduction : dict
        Variance reduction percentages
    nuisance_names : list
        Names of nuisance parameters
    k_names : list
        Names of k parameters
    """
    print("\n3. CREATING NUISANCE INFLUENCE PLOTS")
    print("-" * 40)
    
    # Create correlation heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Correlation heatmap
    corr_matrix = np.zeros((len(nuisance_names), len(k_names)))
    for i, nuisance_name in enumerate(nuisance_names):
        if nuisance_name in correlations:
            corr_matrix[i, :] = correlations[nuisance_name]
    
    im1 = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(len(k_names)))
    ax1.set_yticks(range(len(nuisance_names)))
    ax1.set_xticklabels(k_names, rotation=45)
    ax1.set_yticklabels(nuisance_names)
    ax1.set_title('Correlation with k Parameters')
    
    # Add correlation values as text
    for i in range(len(nuisance_names)):
        for j in range(len(k_names)):
            text = ax1.text(j, i, f'{corr_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    # Variance reduction heatmap
    var_matrix = np.zeros((len(nuisance_names), len(k_names)))
    for i, nuisance_name in enumerate(nuisance_names):
        if nuisance_name in conditional_var_reduction:
            var_matrix[i, :] = conditional_var_reduction[nuisance_name]
    
    im2 = ax2.imshow(var_matrix, cmap='viridis', aspect='auto')
    ax2.set_xticks(range(len(k_names)))
    ax2.set_yticks(range(len(nuisance_names)))
    ax2.set_xticklabels(k_names, rotation=45)
    ax2.set_yticklabels(nuisance_names)
    ax2.set_title('Variance Reduction (%)')
    
    # Add variance reduction values as text
    for i in range(len(nuisance_names)):
        for j in range(len(k_names)):
            text = ax2.text(j, i, f'{var_matrix[i, j]:.1f}%',
                           ha="center", va="center", color="white", fontsize=10)
    
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    plt.tight_layout()
    plt.savefig("nuisance_parameter_influence_edmund.png", dpi=300, bbox_inches="tight")
    print("Nuisance parameter influence plot saved to nuisance_parameter_influence_edmund.png")
    plt.show()
    
    # Create bar plots for most influential parameters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Correlation magnitude bar plot
    max_corr_magnitude = np.max(np.abs(corr_matrix), axis=1)
    bars1 = ax1.bar(range(len(nuisance_names)), max_corr_magnitude, color='skyblue')
    ax1.set_xticks(range(len(nuisance_names)))
    ax1.set_xticklabels(nuisance_names, rotation=45)
    ax1.set_ylabel('Max |Correlation|')
    ax1.set_title('Maximum Correlation Magnitude')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, max_corr_magnitude):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.2f}', ha='center', va='bottom')
    
    # Variance reduction bar plot
    max_var_reduction = np.max(var_matrix, axis=1)
    bars2 = ax2.bar(range(len(nuisance_names)), max_var_reduction, color='lightcoral')
    ax2.set_xticks(range(len(nuisance_names)))
    ax2.set_xticklabels(nuisance_names, rotation=45)
    ax2.set_ylabel('Max Variance Reduction (%)')
    ax2.set_title('Maximum Variance Reduction')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars2, max_var_reduction):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("nuisance_parameter_summary_edmund.png", dpi=300, bbox_inches="tight")
    print("Nuisance parameter summary plot saved to nuisance_parameter_summary_edmund.png")
    plt.show()

def plot_posterior_vs_prior(samples_flat, param_names, param_defs):
    """
    Plot posterior vs prior distributions for all parameters.
    
    Parameters:
    -----------
    samples_flat : np.ndarray
        Flattened samples (n_samples, n_parameters)
    param_names : list
        Names of all parameters
    param_defs : list
        Parameter definitions from config
    """
    print("\n" + "="*60)
    print("POSTERIOR VS PRIOR ANALYSIS")
    print("="*60)
    
    n_params = len(param_names)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (name, param_def) in enumerate(zip(param_names, param_defs)):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Plot posterior histogram
        posterior_samples = samples_flat[:, i]
        ax.hist(posterior_samples, bins=50, density=True, alpha=0.7, 
                label='Posterior', color='blue')
        
        # Plot prior distribution
        if param_def['type'] == 'uniform':
            # Uniform prior
            prior_low = param_def['low']
            prior_high = param_def['high']
            prior_density = 1.0 / (prior_high - prior_low)
            ax.axhline(y=prior_density, color='red', linestyle='--', 
                      label='Prior (Uniform)', linewidth=2)
        elif param_def['type'] == 'normal':
            # Normal prior
            prior_mean = param_def['center']
            prior_std = param_def['sigma']
            x_range = np.linspace(prior_mean - 4*prior_std, prior_mean + 4*prior_std, 100)
            prior_density = np.exp(-0.5 * ((x_range - prior_mean) / prior_std)**2) / (prior_std * np.sqrt(2*np.pi))
            ax.plot(x_range, prior_density, 'r--', label='Prior (Normal)', linewidth=2)
        
        ax.set_title(f'{name}')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_params, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("posterior_vs_prior_edmund.png", dpi=300, bbox_inches="tight")
    print("Posterior vs prior plot saved to posterior_vs_prior_edmund.png")
    plt.show()

def plot_likelihood_values(samples_full, log_pdf_values, param_names):
    """
    Plot likelihood values and their evolution.
    
    Parameters:
    -----------
    samples_full : np.ndarray
        Full parameter samples
    log_pdf_values : np.ndarray
        Log-likelihood values
    param_names : list
        Names of all parameters
    """
    print("\n" + "="*60)
    print("LIKELIHOOD ANALYSIS")
    print("="*60)
    
    # Handle different sample formats
    if len(samples_full.shape) == 3:
        # Flatten samples for analysis
        samples_flat = samples_full.reshape(-1, samples_full.shape[2])
        log_pdf_flat = log_pdf_values.reshape(-1)
    else:
        samples_flat = samples_full
        log_pdf_flat = log_pdf_values
    
    # Plot likelihood evolution
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot log-likelihood vs sample index
    ax1.plot(log_pdf_flat, alpha=0.7)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Log-Likelihood Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Plot histogram of log-likelihood values
    ax2.hist(log_pdf_flat, bins=50, alpha=0.7, density=True)
    ax2.set_xlabel('Log-Likelihood')
    ax2.set_ylabel('Density')
    ax2.set_title('Log-Likelihood Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("likelihood_analysis_edmund.png", dpi=300, bbox_inches="tight")
    print("Likelihood analysis plot saved to likelihood_analysis_edmund.png")
    plt.show()
    
    # Print likelihood statistics
    print(f"Log-likelihood statistics:")
    print(f"  Mean: {np.mean(log_pdf_flat):.3f}")
    print(f"  Std:  {np.std(log_pdf_flat):.3f}")
    print(f"  Min:  {np.min(log_pdf_flat):.3f}")
    print(f"  Max:  {np.max(log_pdf_flat):.3f}")

def plot_likelihood_analysis(samples_full, param_names, param_defs):
    """
    Create likelihood analysis plots by fixing non-k parameters and one k parameter,
    then plotting likelihood over the range of the remaining k parameter.
    
    Parameters:
    -----------
    samples_full : np.ndarray
        Full parameter samples
    param_names : list
        Names of all parameters
    param_defs : list
        Parameter definitions from config
    """
    print("\n" + "="*60)
    print("LIKELIHOOD ANALYSIS - K PARAMETER PROFILES")
    print("="*60)
    
    # Load experimental data and surrogate model
    try:
        # Load experimental data
        data = pd.read_csv("data/experimental/edmund_71Gpa_run1.csv")
        oside_data = data['oside'].values
        y_obs = (oside_data - oside_data[0]) / (data['temp'].max() - data['temp'].min())
        exp_time = data['time'].values
        
        # Interpolate to surrogate grid
        from scipy.interpolate import interp1d
        sim_t_final = 8.5e-6  # seconds (from Edmund config)
        sim_num_steps = 50    # from FPCA model (eigenfunctions shape)
        surrogate_time_grid = np.linspace(0, sim_t_final, sim_num_steps)
        interp_func = interp1d(exp_time, y_obs, kind='linear', 
                               bounds_error=False, fill_value=(y_obs[0], y_obs[-1]))
        interpolated_data = interp_func(surrogate_time_grid)
        
        # Load surrogate model
        from train_surrogate_models import FullSurrogateModel
        surrogate = FullSurrogateModel.load_model("outputs/edmund1/full_surrogate_model.pkl")
        
        print("Successfully loaded experimental data and surrogate model")
        
    except Exception as e:
        print(f"Error loading data or surrogate model: {e}")
        print("Skipping likelihood analysis plots")
        return
    
    # Identify parameter types
    k_param_names = ['k_sample', 'k_ins']
    nuisance_param_names = [name for name in param_names if name not in k_param_names]
    
    # Get posterior means for all parameters from MCMC results
    if len(samples_full.shape) == 3:
        samples_flat = samples_full.reshape(-1, samples_full.shape[2])
    else:
        samples_flat = samples_full
    
    posterior_means = {}
    for i, name in enumerate(param_names):
        posterior_means[name] = np.mean(samples_flat[:, i])
    
    print(f"Posterior means for fixed parameters:")
    for name, value in posterior_means.items():
        print(f"  {name}: {value:.3e}")
    
    # Create likelihood analysis plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Fix k_ins, vary k_sample
    print(f"\nAnalyzing likelihood profile for k_sample (k_ins fixed at {posterior_means['k_ins']:.1f})")
    
    # Define k_sample range (uniform distribution bounds)
    k_sample_range = np.linspace(25.0, 100.0, 100)  # k_sample uniform range
    likelihood_k_sample = []
    
    for k_sample_val in k_sample_range:
        # Create parameter vector with k_sample varying, others fixed
        params = np.array([posterior_means[name] for name in param_names])
        params[param_names.index('k_sample')] = k_sample_val
        
        # Compute likelihood
        try:
            y_pred, _, _, curve_uncert = surrogate.predict_temperature_curves(params.reshape(1, -1))
            
            # Use same likelihood calculation as in MCMC
            SENSOR_VARIANCE = 0.0012
            INCLUDE_SURROGATE_UNCERT = True
            
            if INCLUDE_SURROGATE_UNCERT:
                sigma2 = SENSOR_VARIANCE + curve_uncert**2
            else:
                sigma2 = SENSOR_VARIANCE
            
            # Gaussian log-likelihood
            resid = y_pred - interpolated_data
            ll = -0.5 * np.sum(resid**2 / sigma2 + np.log(2 * np.pi * sigma2))
            likelihood_k_sample.append(ll)
            
        except Exception as e:
            print(f"Error computing likelihood for k_sample={k_sample_val}: {e}")
            likelihood_k_sample.append(np.nan)
    
    # Plot k_sample likelihood profile
    ax1.plot(k_sample_range, likelihood_k_sample, 'b-', linewidth=2, label='Log-likelihood')
    ax1.set_xlabel('k_sample (W/(m·K))')
    ax1.set_ylabel('Log-likelihood')
    ax1.set_title(f'Likelihood Profile: k_sample\n(k_ins fixed at {posterior_means["k_ins"]:.1f} W/(m·K))')
    ax1.grid(True, alpha=0.3)
    
    # Add posterior mean and credible interval from MCMC results
    k_sample_samples = samples_flat[:, param_names.index('k_sample')]
    k_sample_mean = np.mean(k_sample_samples)
    k_sample_ci_low = np.percentile(k_sample_samples, 2.5)
    k_sample_ci_high = np.percentile(k_sample_samples, 97.5)
    
    ax1.axvline(k_sample_mean, color='red', linestyle='--', label=f'Posterior mean: {k_sample_mean:.1f}')
    ax1.axvline(k_sample_ci_low, color='orange', linestyle=':', label=f'95% CI: [{k_sample_ci_low:.1f}, {k_sample_ci_high:.1f}]')
    ax1.axvline(k_sample_ci_high, color='orange', linestyle=':')
    ax1.legend()
    
    # Plot 2: Fix k_sample, vary k_ins
    print(f"Analyzing likelihood profile for k_ins (k_sample fixed at {posterior_means['k_sample']:.1f})")
    
    # Define k_ins range (around posterior mean)
    k_ins_mean = posterior_means['k_ins']
    k_ins_std = param_defs[param_names.index('k_ins')]['sigma']
    k_ins_range = np.linspace(k_ins_mean - 3*k_ins_std, k_ins_mean + 3*k_ins_std, 100)
    likelihood_k_ins = []
    
    for k_ins_val in k_ins_range:
        # Create parameter vector with k_ins varying, others fixed
        params = np.array([posterior_means[name] for name in param_names])
        params[param_names.index('k_ins')] = k_ins_val
        
        # Compute likelihood
        try:
            y_pred, _, _, curve_uncert = surrogate.predict_temperature_curves(params.reshape(1, -1))
            
            if INCLUDE_SURROGATE_UNCERT:
                sigma2 = SENSOR_VARIANCE + curve_uncert**2
            else:
                sigma2 = SENSOR_VARIANCE
            
            # Gaussian log-likelihood
            resid = y_pred - interpolated_data
            ll = -0.5 * np.sum(resid**2 / sigma2 + np.log(2 * np.pi * sigma2))
            likelihood_k_ins.append(ll)
            
        except Exception as e:
            print(f"Error computing likelihood for k_ins={k_ins_val}: {e}")
            likelihood_k_ins.append(np.nan)
    
    # Plot k_ins likelihood profile
    ax2.plot(k_ins_range, likelihood_k_ins, 'g-', linewidth=2, label='Log-likelihood')
    ax2.set_xlabel('k_ins (W/(m·K))')
    ax2.set_ylabel('Log-likelihood')
    ax2.set_title(f'Likelihood Profile: k_ins\n(k_sample fixed at {posterior_means["k_sample"]:.1f} W/(m·K))')
    ax2.grid(True, alpha=0.3)
    
    # Add posterior mean and credible interval from MCMC results
    k_ins_samples = samples_flat[:, param_names.index('k_ins')]
    k_ins_mean = np.mean(k_ins_samples)
    k_ins_ci_low = np.percentile(k_ins_samples, 2.5)
    k_ins_ci_high = np.percentile(k_ins_samples, 97.5)
    
    ax2.axvline(k_ins_mean, color='red', linestyle='--', label=f'Posterior mean: {k_ins_mean:.1f}')
    ax2.axvline(k_ins_ci_low, color='orange', linestyle=':', label=f'95% CI: [{k_ins_ci_low:.1f}, {k_ins_ci_high:.1f}]')
    ax2.axvline(k_ins_ci_high, color='orange', linestyle=':')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("likelihood_analysis_k_params_edmund.png", dpi=300, bbox_inches="tight")
    print("Likelihood analysis plots saved to likelihood_analysis_k_params_edmund.png")
    plt.show()
    
    # Print summary statistics
    print(f"\nLikelihood Analysis Summary:")
    print(f"k_sample profile:")
    print(f"  Range: {k_sample_range[0]:.1f} to {k_sample_range[-1]:.1f} W/(m·K)")
    print(f"  Max likelihood at: {k_sample_range[np.nanargmax(likelihood_k_sample)]:.1f} W/(m·K)")
    print(f"  Posterior mean: {k_sample_mean:.1f} W/(m·K)")
    print(f"  Posterior 95% CI: [{k_sample_ci_low:.1f}, {k_sample_ci_high:.1f}] W/(m·K)")
    
    print(f"\nk_ins profile:")
    print(f"  Range: {k_ins_range[0]:.1f} to {k_ins_range[-1]:.1f} W/(m·K)")
    print(f"  Max likelihood at: {k_ins_range[np.nanargmax(likelihood_k_ins)]:.1f} W/(m·K)")
    print(f"  Posterior mean: {k_ins_mean:.1f} W/(m·K)")
    print(f"  Posterior 95% CI: [{k_ins_ci_low:.1f}, {k_ins_ci_high:.1f}] W/(m·K)")

def load_mcmc_results():
    """
    Load Edmund MCMC results from saved .npz file.
    """
    try:
        # Load from saved .npz file
        data = np.load("mcmc_results_edmund.npz")
        samples_full = data['samples_full']
        log_pdf_values = data.get('log_pdf_values', None)
        
        print(f"Loaded {len(samples_full)} accepted samples with {samples_full.shape[1]} parameters")
        if log_pdf_values is not None:
            print(f"Loaded {len(log_pdf_values)} log-likelihood values")
        
        return samples_full, log_pdf_values
        
    except FileNotFoundError:
        print("Could not find mcmc_results_edmund.npz. Make sure uqpy_MCMC_edmund.py has been run.")
        return None, None

def create_corner_plot(data, labels, title, filename):
    """
    Create a corner plot using either corner library or seaborn fallback.
    
    Parameters:
    -----------
    data : np.ndarray
        Data to plot (n_samples, n_dimensions) or (n_samples, n_chains, n_dimensions)
    labels : list
        Labels for each dimension
    title : str
        Plot title
    filename : str
        Output filename
    """
    # Handle different data formats
    if len(data.shape) == 3:
        # Data is in format (n_samples, n_chains, n_dimensions)
        # Flatten to (n_samples * n_chains, n_dimensions) for plotting
        data_flat = data.reshape(-1, data.shape[2])
        print(f"Corner Debug: data in chain format, flattened shape = {data_flat.shape}")
    else:
        # Data is in flat format (n_samples, n_dimensions)
        data_flat = data
        print(f"Corner Debug: data in flat format, shape = {data_flat.shape}")
    
    try:
        import corner
        thin = 20
        fig = corner.corner(
            data_flat,
            labels=labels,
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 10}
        )
        # Move title higher and add proper spacing to prevent overlap
        fig.suptitle(title, fontsize=14, y=1.0)
        # Adjust subplot parameters to create more space for title
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95)
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Corner plot saved to {filename}")
        plt.show()
        
    except ImportError:
        # Fallback using seaborn
        df = pd.DataFrame(data_flat, columns=labels)
        g = sns.pairplot(df, corner=True, diag_kind="kde", 
                        plot_kws={"s": 5, "alpha": 0.4})
        g.fig.suptitle(title, fontsize=14, y=1.0)
        g.fig.tight_layout()
        # Add extra space for title
        g.fig.subplots_adjust(top=0.92)
        g.fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Pair plot saved to {filename} (corner library not installed)")
        plt.show()

def plot_parameter_statistics(samples_full, param_names):
    """
    Print and plot parameter statistics for all parameters.
    
    Parameters:
    -----------
    samples_full : np.ndarray
        Full parameter samples (n_samples, 8) or (n_samples, n_chains, 8)
    param_names : list
        Names of all parameters
    """
    # Handle different sample formats
    if len(samples_full.shape) == 3:
        # Samples are in format (n_samples, n_chains, 8)
        # Flatten to (n_samples * n_chains, 8) for statistics
        samples_flat = samples_full.reshape(-1, samples_full.shape[2])
        print(f"Stats Debug: samples in chain format, flattened shape = {samples_flat.shape}")
    else:
        # Samples are in flat format (n_samples, 8)
        samples_flat = samples_full
        print(f"Stats Debug: samples in flat format, shape = {samples_flat.shape}")
    
    # All parameter statistics
    print("\n" + "="*60)
    print("ALL PARAMETER STATISTICS")
    print("="*60)
    print(f"{'Parameter':<15} {'Posterior Mean':<15} {'Posterior Std':<15}")
    print("-" * 60)
    for i, name in enumerate(param_names):
        mean_val = samples_flat[:, i].mean()
        std_val = samples_flat[:, i].std()
        print(f"{name:<15} {mean_val:<15.3e} {std_val:<15.3e}")
    
    # Create summary plot for all parameters
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create box plot for all parameters
    box_data = [samples_flat[:, i] for i in range(len(param_names))]
    ax.boxplot(box_data)
    ax.set_xticklabels(param_names)
    ax.set_title("All Parameter Distributions")
    ax.set_ylabel("Parameter Value")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("parameter_statistics_edmund.png", dpi=300, bbox_inches="tight")
    print("Parameter statistics plot saved to parameter_statistics_edmund.png")
    plt.show()

def plot_trace_plots(samples_full, param_names, n_walkers=44):
    """
    Plot trace plots to check convergence for all parameters.
    
    Parameters:
    -----------
    samples_full : np.ndarray
        Full parameter samples (n_samples, 8) or (n_samples, n_chains, 8)
    param_names : list
        Names of all parameters
    n_walkers : int
        Number of walkers used in the ensemble MCMC
    """
    # Handle different sample formats
    if len(samples_full.shape) == 3:
        # Samples are in format (n_samples, n_chains, 8)
        # Transpose to (n_chains, n_samples, 8) for easier plotting
        samples_reshaped = samples_full.transpose(1, 0, 2)  # (n_chains, n_samples, 8)
        print(f"Trace Debug: samples in chain format, shape = {samples_full.shape}")
    else:
        # Samples are in flat format (n_samples, 8)
        # Reshape to separate walkers
        nsamples_per_walker = len(samples_full) // n_walkers
        samples_reshaped = samples_full[:nsamples_per_walker * n_walkers].reshape(n_walkers, nsamples_per_walker, samples_full.shape[1])
        print(f"Trace Debug: reshaped flat samples, shape = {samples_reshaped.shape}")
    
    # Create trace plots for all parameters
    n_params = len(param_names)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow', 'black'][:n_params]
    
    for param_idx in range(n_params):
        row = param_idx // n_cols
        col = param_idx % n_cols
        ax = axes[row, col]
        
        # Plot each walker's trace (show first 8 walkers for clarity)
        for walker_idx in range(min(8, n_walkers)):
            trace = samples_reshaped[walker_idx, :, param_idx]
            ax.plot(trace, alpha=0.6, linewidth=0.5, color=colors[param_idx])
        
        # Plot mean across walkers
        mean_trace = np.mean(samples_reshaped[:, :, param_idx], axis=0)
        ax.plot(mean_trace, 'k-', linewidth=2, label='Mean across walkers')
        
        ax.set_title(f'Trace Plot: {param_names[param_idx]}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Parameter Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_params, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("trace_plots_edmund.png", dpi=300, bbox_inches="tight")
    print("Trace plots saved to trace_plots_edmund.png")
    plt.show()

def main():
    """Main function to load and plot Edmund MCMC results."""
    print("Loading Edmund MCMC results...")
    
    samples_full, log_pdf_values = load_mcmc_results()
    
    if samples_full is None:
        print("No Edmund MCMC results found. Please run uqpy_MCMC_edmund.py first.")
        return
    
    # Get parameter names from Edmund config
    param_defs = get_param_defs_from_config(config_path="configs/distributions_edmund.yaml")
    param_names = [param_def['name'] for param_def in param_defs]
    
    print(f"Parameter names: {param_names}")
    
    # Handle different sample formats for analysis
    if len(samples_full.shape) == 3:
        # Flatten samples for analysis
        samples_flat = samples_full.reshape(-1, samples_full.shape[2])
    else:
        samples_flat = samples_full
    
    # Compute convergence diagnostics
    print("\n" + "="*60)
    print("CONVERGENCE DIAGNOSTICS")
    print("="*60)
    
    # ESS and R-hat for all parameters (using 44 walkers as in Edmund MCMC)
    n_walkers = 44
    ess_all = compute_ess_arviz(samples_full, param_names, n_walkers=n_walkers)
    rhat_all = compute_rhat_arviz(samples_full, param_names, n_walkers=n_walkers)
    
    print("\nAll Parameters:")
    for i, name in enumerate(param_names):
        print(f"  {name:<15}: ESS = {ess_all[i]:.0f}, R-hat = {rhat_all[i]:.3f}")
    
    # Check convergence criteria
    min_ess = float(np.min(ess_all))
    max_rhat = float(np.max(rhat_all))
    
    print(f"\nConvergence Summary:")
    print(f"  Minimum ESS: {min_ess:.0f} (should be > 200)")
    print(f"  Maximum R-hat: {max_rhat:.3f} (should be < 1.01)")
    
    if min_ess < 200:
        print("  ⚠️  WARNING: ESS too low - consider running more samples")
    if max_rhat > 1.01:
        print("  ⚠️  WARNING: R-hat too high - consider longer chains or better mixing")
    if min_ess >= 200 and max_rhat <= 1.01:
        print("  ✅ Convergence looks good!")
    
    # Print statistics
    plot_parameter_statistics(samples_full, param_names)
    
    # Analyze nuisance parameter influence on k parameters
    print("\nAnalyzing nuisance parameter influence on k parameters...")
    influence_results = analyze_nuisance_parameter_influence(samples_flat, param_names)
    
    # Plot nuisance parameter influence analysis
    plot_nuisance_influence_analysis(
        influence_results['correlations'], 
        influence_results['conditional_var_reduction'], 
        influence_results['nuisance_names'], 
        influence_results['k_names']
    )
    
    # Plot likelihood values if available
    if log_pdf_values is not None:
        print("\nCreating likelihood analysis plots...")
        plot_likelihood_values(samples_full, log_pdf_values, param_names)
    else:
        print("\nNo log-likelihood values found in mcmc_results_edmund.npz")
        print("To include likelihood analysis, modify uqpy_MCMC_edmund.py to save log_pdf_values")
    
    # Create trace plots for convergence diagnostics
    print("\nCreating trace plots...")
    plot_trace_plots(samples_full, param_names, n_walkers=n_walkers)
    
    # Create corner plots
    print("\nCreating corner plots...")
    
    # Full parameter corner plot for all 8 parameters
    full_labels = [f"${name}$" if name.startswith('k_') else name for name in param_names]
    create_corner_plot(samples_full, full_labels, "All Parameters Posterior (Edmund)", "corner_plot_edmund.png")
    
    # K-only corner plot for thermal conductivity parameters
    k_param_names = ['k_sample', 'k_ins']
    k_indices = [param_names.index(name) for name in k_param_names]
    
    # Extract k parameters from samples
    if len(samples_full.shape) == 3:
        # (n_samples, n_chains, n_dimensions) -> (n_samples * n_chains, n_k_params)
        k_samples = samples_full[:, :, k_indices].reshape(-1, len(k_indices))
    else:
        # (n_samples, n_dimensions) -> (n_samples, n_k_params)
        k_samples = samples_full[:, k_indices]
    
    k_labels = [f"${name}$" for name in k_param_names]
    create_corner_plot(k_samples, k_labels, "Thermal Conductivity Parameters Posterior (Edmund)", "corner_plot_k_params_edmund.png")
    
    # Print k-parameter specific statistics
    print("\n" + "="*60)
    print("THERMAL CONDUCTIVITY PARAMETER STATISTICS")
    print("="*60)
    print(f"{'Parameter':<15} {'Posterior Mean':<15} {'Posterior Std':<15} {'95% CI':<20}")
    print("-" * 70)
    for i, name in enumerate(k_param_names):
        samples_k = k_samples[:, i]
        mean_val = samples_k.mean()
        std_val = samples_k.std()
        ci_low = np.percentile(samples_k, 2.5)
        ci_high = np.percentile(samples_k, 97.5)
        print(f"{name:<15} {mean_val:<15.3e} {std_val:<15.3e} [{ci_low:.3e}, {ci_high:.3e}]")
    
    # Plot posterior vs prior distributions for all parameters
    plot_posterior_vs_prior(samples_flat, param_names, param_defs)
    
    # Create likelihood analysis plots for k parameters
    print("\nCreating likelihood analysis plots for k parameters...")
    plot_likelihood_analysis(samples_full, param_names, param_defs)

    print("\nAll Edmund plots completed!")

if __name__ == "__main__":
    main() 