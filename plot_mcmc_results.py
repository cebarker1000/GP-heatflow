#!/usr/bin/env python3
"""
Plot MCMC results from saved outputs.
Loads samples and creates corner plots for full 11 parameters and κ parameters.
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

def compute_ess_arviz(samples, param_names, n_walkers=24):
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

def compute_rhat_arviz(samples, param_names, n_walkers=24):
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
    # Identify nuisance and k parameters
    nuisance_indices = list(range(8))  # First 8 parameters are nuisance
    k_indices = [8, 9, 10]  # k_sample, k_ins, k_coupler
    k_names = ['k_sample', 'k_ins', 'k_coupler']
    
    nuisance_names = [param_names[i] for i in nuisance_indices]
    
    print(f"\n" + "="*60)
    print("NUISANCE PARAMETER INFLUENCE ANALYSIS")
    print("="*60)
    
    # 1. Correlation Analysis
    print("\n1. CORRELATION ANALYSIS")
    print("-" * 40)
    print(f"{'Nuisance Param':<15} {'k_sample':<12} {'k_ins':<12} {'k_coupler':<12}")
    print("-" * 60)
    
    correlations = {}
    for i, nuisance_idx in enumerate(nuisance_indices):
        nuisance_param = samples_flat[:, nuisance_idx]
        corr_row = []
        for k_idx in k_indices:
            k_param = samples_flat[:, k_idx]
            corr = np.corrcoef(nuisance_param, k_param)[0, 1]
            corr_row.append(corr)
        correlations[nuisance_names[i]] = corr_row
        print(f"{nuisance_names[i]:<15} {corr_row[0]:<12.3f} {corr_row[1]:<12.3f} {corr_row[2]:<12.3f}")
    
    # 2. Conditional Variance Analysis
    print("\n2. CONDITIONAL VARIANCE ANALYSIS")
    print("-" * 40)
    print("Variance reduction when nuisance parameter is fixed at its mean")
    print(f"{'Nuisance Param':<15} {'k_sample':<12} {'k_ins':<12} {'k_coupler':<12}")
    print("-" * 60)
    
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
            print(f"{nuisance_names[i]:<15} {var_reduction[0]:<12.1f}% {var_reduction[1]:<12.1f}% {var_reduction[2]:<12.1f}%")
        else:
            conditional_var_reduction[nuisance_names[i]] = [0, 0, 0]
            print(f"{nuisance_names[i]:<15} {'insufficient':<12} {'insufficient':<12} {'insufficient':<12}")
    
    # 3. Create visualization
    plot_nuisance_influence_analysis(correlations, conditional_var_reduction, nuisance_names, k_names)
    
    return {
        'correlations': correlations,
        'conditional_var_reduction': conditional_var_reduction,
        'unconditional_var': unconditional_var
    }

def plot_nuisance_influence_analysis(correlations, conditional_var_reduction, nuisance_names, k_names):
    """
    Create visualizations for nuisance parameter influence analysis.
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Correlation heatmap
    corr_matrix = np.array([correlations[name] for name in nuisance_names])
    im1 = axes[0, 0].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[0, 0].set_title('Correlation between Nuisance and k Parameters')
    axes[0, 0].set_xticks(range(len(k_names)))
    axes[0, 0].set_xticklabels(k_names, rotation=45)
    axes[0, 0].set_yticks(range(len(nuisance_names)))
    axes[0, 0].set_yticklabels(nuisance_names)
    plt.colorbar(im1, ax=axes[0, 0], label='Correlation')
    
    # Add correlation values as text
    for i in range(len(nuisance_names)):
        for j in range(len(k_names)):
            text = axes[0, 0].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
    
    # 2. Variance reduction heatmap
    var_reduction_matrix = np.array([conditional_var_reduction[name] for name in nuisance_names])
    im2 = axes[0, 1].imshow(var_reduction_matrix, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Variance Reduction (%) when Nuisance Parameter Fixed')
    axes[0, 1].set_xticks(range(len(k_names)))
    axes[0, 1].set_xticklabels(k_names, rotation=45)
    axes[0, 1].set_yticks(range(len(nuisance_names)))
    axes[0, 1].set_yticklabels(nuisance_names)
    plt.colorbar(im2, ax=axes[0, 1], label='Variance Reduction (%)')
    
    # Add variance reduction values as text
    for i in range(len(nuisance_names)):
        for j in range(len(k_names)):
            text = axes[0, 1].text(j, i, f'{var_reduction_matrix[i, j]:.1f}%',
                                  ha="center", va="center", color="white", fontsize=8)
    
    # 3. Overall influence score (combining correlation and variance reduction)
    # Normalize both metrics and combine them
    corr_abs = np.abs(corr_matrix)
    var_norm = var_reduction_matrix / 100.0  # Normalize to [0,1]
    
    # Simple combination: average of normalized metrics
    influence_score = (corr_abs + var_norm) / 2
    
    im3 = axes[1, 0].imshow(influence_score, cmap='plasma', aspect='auto')
    axes[1, 0].set_title('Overall Influence Score (0-1)')
    axes[1, 0].set_xticks(range(len(k_names)))
    axes[1, 0].set_xticklabels(k_names, rotation=45)
    axes[1, 0].set_yticks(range(len(nuisance_names)))
    axes[1, 0].set_yticklabels(nuisance_names)
    plt.colorbar(im3, ax=axes[1, 0], label='Influence Score')
    
    # Add influence scores as text
    for i in range(len(nuisance_names)):
        for j in range(len(k_names)):
            text = axes[1, 0].text(j, i, f'{influence_score[i, j]:.2f}',
                                  ha="center", va="center", color="white", fontsize=8)
    
    # 4. Bar plot of average influence scores
    avg_influence = np.mean(influence_score, axis=1)
    bars = axes[1, 1].bar(range(len(nuisance_names)), avg_influence, color='skyblue', alpha=0.7)
    axes[1, 1].set_title('Average Influence Score by Nuisance Parameter')
    axes[1, 1].set_xlabel('Nuisance Parameter')
    axes[1, 1].set_ylabel('Average Influence Score')
    axes[1, 1].set_xticks(range(len(nuisance_names)))
    axes[1, 1].set_xticklabels(nuisance_names, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, avg_influence)):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("nuisance_parameter_influence.png", dpi=300, bbox_inches="tight")
    print("Nuisance parameter influence analysis saved to nuisance_parameter_influence.png")
    plt.show()
    
    # Print summary of most influential parameters
    print("\n3. SUMMARY OF MOST INFLUENTIAL NUISANCE PARAMETERS")
    print("-" * 60)
    sorted_indices = np.argsort(avg_influence)[::-1]  # Sort in descending order
    for i, idx in enumerate(sorted_indices):
        print(f"{i+1}. {nuisance_names[idx]}: {avg_influence[idx]:.3f}")

def plot_posterior_vs_prior(samples_flat, param_names, param_defs):
    """
    Plot posterior distributions of nuisance parameters overlaid on their prior distributions.
    
    Parameters:
    -----------
    samples_flat : np.ndarray
        Flattened samples (n_samples, n_parameters)
    param_names : list
        Names of all parameters
    param_defs : list
        Parameter definitions from config
    """
    # Identify nuisance parameters (first 8)
    nuisance_indices = list(range(8))
    nuisance_names = [param_names[i] for i in nuisance_indices]
    
    print(f"\n" + "="*60)
    print("POSTERIOR VS PRIOR DISTRIBUTIONS")
    print("="*60)
    
    # Create subplots for nuisance parameters
    n_nuisance = len(nuisance_names)
    n_cols = 3
    n_rows = (n_nuisance + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Generate prior samples
    from analysis.config_utils import create_uqpy_distributions
    from UQpy.distributions.collection.JointIndependent import JointIndependent
    
    uqpy_dists = create_uqpy_distributions(param_defs)
    nuisance_prior = JointIndependent(marginals=uqpy_dists[:8])  # First 8 parameters
    
    # Generate many prior samples
    n_prior_samples = 50000
    prior_samples = nuisance_prior.rvs(nsamples=n_prior_samples)
    
    for i, nuisance_idx in enumerate(nuisance_indices):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Get posterior samples for this parameter
        posterior_samples = samples_flat[:, nuisance_idx]
        
        # Get prior samples for this parameter
        prior_param_samples = prior_samples[:, i]
        
        # Create histograms
        ax.hist(prior_param_samples, bins=50, alpha=0.6, label='Prior', 
                color='blue', density=True, edgecolor='black')
        ax.hist(posterior_samples, bins=50, alpha=0.6, label='Posterior', 
                color='red', density=True, edgecolor='black')
        
        ax.set_title(f'{nuisance_names[i]}')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        prior_mean = np.mean(prior_param_samples)
        prior_std = np.std(prior_param_samples)
        post_mean = np.mean(posterior_samples)
        post_std = np.std(posterior_samples)
        
        stats_text = f'Prior: {prior_mean:.2e} ± {prior_std:.2e}\nPost: {post_mean:.2e} ± {post_std:.2e}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_nuisance, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("posterior_vs_prior.png", dpi=300, bbox_inches="tight")
    print("Posterior vs prior distributions saved to posterior_vs_prior.png")
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"{'Parameter':<15} {'Prior Mean':<15} {'Prior Std':<15} {'Post Mean':<15} {'Post Std':<15} {'Change':<10}")
    print("-" * 90)
    
    for i, nuisance_idx in enumerate(nuisance_indices):
        prior_samples_param = prior_samples[:, i]
        posterior_samples_param = samples_flat[:, nuisance_idx]
        
        prior_mean = np.mean(prior_samples_param)
        prior_std = np.std(prior_samples_param)
        post_mean = np.mean(posterior_samples_param)
        post_std = np.std(posterior_samples_param)
        
        # Calculate relative change in mean
        if abs(prior_mean) > 1e-10:
            mean_change = (post_mean - prior_mean) / abs(prior_mean) * 100
        else:
            mean_change = 0
        
        print(f"{nuisance_names[i]:<15} {prior_mean:<15.3e} {prior_std:<15.3e} {post_mean:<15.3e} {post_std:<15.3e} {mean_change:<10.1f}%")

def plot_likelihood_values(samples_full, log_pdf_values, param_names):
    """
    Plot likelihood values to check if MCMC is exploring high-likelihood regions.
    
    Parameters:
    -----------
    samples_full : np.ndarray
        Full parameter samples (n_samples, 11) or (n_samples, n_chains, 11)
    log_pdf_values : np.ndarray
        Log-likelihood values for each sample
    param_names : list
        Names of all parameters
    """
    # Handle different sample formats
    if len(samples_full.shape) == 3:
        # Samples are in format (n_samples, n_chains, 11)
        # Flatten to (n_samples * n_chains, 11) for plotting
        samples_flat = samples_full.reshape(-1, samples_full.shape[2])
        print(f"Likelihood Debug: samples in chain format, flattened shape = {samples_flat.shape}")
    else:
        # Samples are in flat format (n_samples, 11)
        samples_flat = samples_full
        print(f"Likelihood Debug: samples in flat format, shape = {samples_flat.shape}")
    
    # Convert log-likelihood to likelihood (exponentiate)
    likelihood_values = np.exp(log_pdf_values)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Likelihood vs sample index
    axes[0, 0].plot(likelihood_values, alpha=0.6, linewidth=0.5)
    axes[0, 0].set_title('Likelihood Values Over Time')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Likelihood')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Log-likelihood vs sample index
    axes[0, 1].plot(log_pdf_values, alpha=0.6, linewidth=0.5)
    axes[0, 1].set_title('Log-Likelihood Values Over Time')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Log-Likelihood')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Likelihood histogram
    axes[1, 0].hist(likelihood_values, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribution of Likelihood Values')
    axes[1, 0].set_xlabel('Likelihood')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Likelihood vs k_sample (first k parameter, index 8)
    k_sample_idx = 8  # k_sample is at index 8
    scatter = axes[1, 1].scatter(samples_flat[:, k_sample_idx], likelihood_values, 
                                 c=likelihood_values, cmap='viridis', alpha=0.6, s=10)
    axes[1, 1].set_title('Likelihood vs k_sample')
    axes[1, 1].set_xlabel('k_sample')
    axes[1, 1].set_ylabel('Likelihood')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='Likelihood')
    
    plt.tight_layout()
    plt.savefig("likelihood_analysis.png", dpi=300, bbox_inches="tight")
    print("Likelihood analysis plot saved to likelihood_analysis.png")
    plt.show()
    
    # Print statistics
    print(f"\nLikelihood Statistics:")
    print(f"  Mean likelihood: {np.mean(likelihood_values):.2e}")
    print(f"  Std likelihood: {np.std(likelihood_values):.2e}")
    print(f"  Min likelihood: {np.min(likelihood_values):.2e}")
    print(f"  Max likelihood: {np.max(likelihood_values):.2e}")
    print(f"  Mean log-likelihood: {np.mean(log_pdf_values):.2f}")
    print(f"  Std log-likelihood: {np.std(log_pdf_values):.2f}")

def load_mcmc_results():
    """
    Load MCMC results from saved .npz file.
    """
    try:
        # Load from saved .npz file
        data = np.load("mcmc_results.npz")
        samples_full = data['samples_full']
        log_pdf_values = data.get('log_pdf_values', None)
        
        print(f"Loaded {len(samples_full)} accepted samples with {samples_full.shape[1]} parameters")
        if log_pdf_values is not None:
            print(f"Loaded {len(log_pdf_values)} log-likelihood values")
        
        return samples_full, log_pdf_values
        
    except FileNotFoundError:
        print("Could not find mcmc_results.npz. Make sure uqpy_MCMC.py has been run.")
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
        fig.suptitle(title, fontsize=14, y=0.95)
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Corner plot saved to {filename}")
        plt.show()
        
    except ImportError:
        # Fallback using seaborn
        df = pd.DataFrame(data_flat, columns=labels)
        g = sns.pairplot(df, corner=True, diag_kind="kde", 
                        plot_kws={"s": 5, "alpha": 0.4})
        g.fig.suptitle(title, fontsize=14)
        g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.95)
        g.fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Pair plot saved to {filename} (corner library not installed)")
        plt.show()

def plot_parameter_statistics(samples_full, param_names):
    """
    Print and plot parameter statistics for all parameters.
    
    Parameters:
    -----------
    samples_full : np.ndarray
        Full parameter samples (n_samples, 11) or (n_samples, n_chains, 11)
    param_names : list
        Names of all parameters
    """
    # Handle different sample formats
    if len(samples_full.shape) == 3:
        # Samples are in format (n_samples, n_chains, 11)
        # Flatten to (n_samples * n_chains, 11) for statistics
        samples_flat = samples_full.reshape(-1, samples_full.shape[2])
        print(f"Stats Debug: samples in chain format, flattened shape = {samples_flat.shape}")
    else:
        # Samples are in flat format (n_samples, 11)
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
    plt.savefig("parameter_statistics.png", dpi=300, bbox_inches="tight")
    print("Parameter statistics plot saved to parameter_statistics.png")
    plt.show()

def plot_trace_plots(samples_full, param_names, n_walkers=24):
    """
    Plot trace plots to check convergence for all parameters.
    
    Parameters:
    -----------
    samples_full : np.ndarray
        Full parameter samples (n_samples, 11) or (n_samples, n_chains, 11)
    param_names : list
        Names of all parameters
    n_walkers : int
        Number of walkers used in the ensemble MCMC
    """
    # Handle different sample formats
    if len(samples_full.shape) == 3:
        # Samples are in format (n_samples, n_chains, 11)
        # Transpose to (n_chains, n_samples, 11) for easier plotting
        samples_reshaped = samples_full.transpose(1, 0, 2)  # (n_chains, n_samples, 11)
        print(f"Trace Debug: samples in chain format, shape = {samples_full.shape}")
    else:
        # Samples are in flat format (n_samples, 11)
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
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta'][:n_params]
    
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
    plt.savefig("trace_plots.png", dpi=300, bbox_inches="tight")
    print("Trace plots saved to trace_plots.png")
    plt.show()

def main():
    """Main function to load and plot MCMC results."""
    print("Loading MCMC results...")
    
    samples_full, log_pdf_values = load_mcmc_results()
    
    if samples_full is None:
        print("No MCMC results found. Please run uqpy_MCMC.py first.")
        return
    
    # Get parameter names from config
    param_defs = get_param_defs_from_config(config_path="configs/distributions.yaml")
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
    
    # ESS and R-hat for all parameters (using 24 walkers as in updated uqpy_MCMC.py)
    n_walkers = 24
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
    
    # Analyze nuisance parameter influence
    print("\nAnalyzing nuisance parameter influence on k parameters...")
    influence_results = analyze_nuisance_parameter_influence(samples_flat, param_names)
    
    # Plot likelihood values if available
    if log_pdf_values is not None:
        print("\nCreating likelihood analysis plots...")
        plot_likelihood_values(samples_full, log_pdf_values, param_names)
    else:
        print("\nNo log-likelihood values found in mcmc_results.npz")
        print("To include likelihood analysis, modify uqpy_MCMC.py to save log_pdf_values")
    
    # Create trace plots for convergence diagnostics
    print("\nCreating trace plots...")
    plot_trace_plots(samples_full, param_names, n_walkers=n_walkers)
    
    # Create corner plots
    print("\nCreating corner plots...")
    
    # κ parameters corner plot (indices 8, 9, 10)
    k_indices = [8, 9, 10]  # k_sample, k_ins, k_coupler
    k_names = ['k_sample', 'k_ins', 'k_coupler']
    k_samples = samples_full[:, k_indices] if len(samples_full.shape) == 2 else samples_full[:, :, k_indices]
    k_labels = [r"$k_{\text{sample}}$", r"$k_{\text{ins}}$", r"$k_{\text{coupler}}$"]
    create_corner_plot(k_samples, k_labels, "κ Parameters Posterior", "kappa_corner_plot.png")
    
    # Full parameter corner plot for all 11 parameters
    full_labels = [f"${name}$" if name.startswith('k_') else name for name in param_names]
    create_corner_plot(samples_full, full_labels, "All Parameters Posterior", "full_corner_plot.png")
    
    # Plot posterior vs prior distributions for nuisance parameters
    plot_posterior_vs_prior(samples_flat, param_names, param_defs)

    print("\nAll plots completed!")

if __name__ == "__main__":
    main() 