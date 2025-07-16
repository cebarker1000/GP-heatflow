#!/usr/bin/env python3
"""
Plot full parameter MCMC results from test_full_parameter_mcmc.py.
Loads samples and creates comprehensive diagnostics including R-hat, ESS, and corner plots for all 11 parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.signal import correlate

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

def compute_ess_arviz(samples, n_walkers=24):
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
        
        # Create InferenceData with proper structure for all 11 parameters
        param_names = [f"param_{i}" for i in range(chains.shape[2])]
        posterior_dict = {param_names[i]: chains[:, :, i] for i in range(chains.shape[2])}
        
        idata = az.from_dict(posterior=posterior_dict)
        
        # Compute ESS using ArviZ's built-in method
        ess_bulk = az.ess(idata, method="bulk")
        
        ess_values = np.array([ess_bulk[param].values for param in param_names])
        
        print(f"ESS Debug: computed ESS values = {ess_values}")
        
        return ess_values
        
    except ImportError:
        print("ArviZ not available, skipping ESS calculation")
        return np.array([np.nan] * 11)
    except Exception as e:
        print(f"Error computing ESS: {e}")
        return np.array([np.nan] * 11)

def compute_rhat_arviz(samples, n_walkers=24):
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
        
        # Create InferenceData with proper structure for all 11 parameters
        param_names = [f"param_{i}" for i in range(chains.shape[2])]
        posterior_dict = {param_names[i]: chains[:, :, i] for i in range(chains.shape[2])}
        
        idata = az.from_dict(posterior=posterior_dict)
        
        # Compute R-hat using ArviZ's built-in method
        rhat = az.rhat(idata)
        
        rhat_values = np.array([rhat[param].values for param in param_names])
        
        return rhat_values
        
    except ImportError:
        print("ArviZ not available, skipping R-hat calculation")
        return np.array([np.nan] * 11)
    except Exception as e:
        print(f"Error computing R-hat: {e}")
        return np.array([np.nan] * 11)

def plot_likelihood_values(samples_full, log_pdf_values):
    """
    Plot likelihood values to check if MCMC is exploring high-likelihood regions.
    
    Parameters:
    -----------
    samples_full : np.ndarray
        Full parameter samples (n_samples, 11) or (n_samples, n_chains, 11)
    log_pdf_values : np.ndarray
        Log-likelihood values for each sample
    """
    # Handle different sample formats
    if len(samples_full.shape) == 3:
        # Samples are in format (n_samples, n_chains, 11)
        # Flatten to (n_samples * n_chains, 11) for plotting
        samples_flat = samples_full.reshape(-1, 11)
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
    scatter = axes[1, 1].scatter(samples_flat[:, 8], likelihood_values, 
                                 c=likelihood_values, cmap='viridis', alpha=0.6, s=10)
    axes[1, 1].set_title('Likelihood vs k_sample')
    axes[1, 1].set_xlabel('k_sample')
    axes[1, 1].set_ylabel('Likelihood')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='Likelihood')
    
    plt.tight_layout()
    plt.savefig("test_likelihood_analysis.png", dpi=300, bbox_inches="tight")
    print("Likelihood analysis plot saved to test_likelihood_analysis.png")
    plt.show()
    
    # Print statistics
    print(f"\nLikelihood Statistics:")
    print(f"  Mean likelihood: {np.mean(likelihood_values):.2e}")
    print(f"  Std likelihood: {np.std(likelihood_values):.2e}")
    print(f"  Min likelihood: {np.min(likelihood_values):.2e}")
    print(f"  Max likelihood: {np.max(likelihood_values):.2e}")
    print(f"  Mean log-likelihood: {np.mean(log_pdf_values):.2f}")
    print(f"  Std log-likelihood: {np.std(log_pdf_values):.2f}")

def create_prior_posterior_comparison_plot(samples_flat, uqpy_dists, param_names):
    """
    Create a comparison plot showing prior vs posterior distributions for nuisance parameters.
    
    Parameters:
    -----------
    samples_flat : np.ndarray
        Flattened samples (n_samples, n_parameters)
    uqpy_dists : list
        List of UQpy distribution objects for priors
    param_names : list
        Names of the nuisance parameters
    """
    n_nuisance = len(param_names)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(n_nuisance):
        ax = axes[i]
        
        # Plot posterior histogram
        ax.hist(samples_flat[:, i], bins=50, alpha=0.7, density=True, 
                color='blue', label='Posterior', edgecolor='black')
        
        # Plot prior distribution
        prior_dist = uqpy_dists[i]
        
        # Generate prior samples for plotting
        if hasattr(prior_dist, 'rvs'):
            prior_samples = prior_dist.rvs(nsamples=10000)
            ax.hist(prior_samples, bins=50, alpha=0.5, density=True, 
                    color='red', label='Prior', edgecolor='black')
        
        ax.set_title(f'{param_names[i]}')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("test_prior_posterior_comparison.png", dpi=300, bbox_inches="tight")
    print("Prior vs posterior comparison plot saved to test_prior_posterior_comparison.png")
    plt.show()

def load_mcmc_results():
    """
    Load MCMC results from saved .npz file.
    """
    try:
        # Load from saved .npz file
        data = np.load("test_full_mcmc_results.npz")
        samples_full = data['samples_full']
        log_pdf_values = data.get('log_pdf_values', None)
        true_parameters = data.get('true_parameters', None)
        synthetic_data = data.get('synthetic_data', None)
        
        print(f"Loaded {len(samples_full)} accepted samples")
        if log_pdf_values is not None:
            print(f"Loaded {len(log_pdf_values)} log-likelihood values")
        if true_parameters is not None:
            print(f"Loaded true parameters: {true_parameters}")
        if synthetic_data is not None:
            print(f"Loaded synthetic data with shape: {synthetic_data.shape}")
        
        return samples_full, log_pdf_values, true_parameters, synthetic_data
        
    except FileNotFoundError:
        print("Could not find test_full_mcmc_results.npz. Make sure test_full_parameter_mcmc.py has been run.")
        return None, None, None, None

def create_corner_plot_with_truths(data, labels, title, filename, true_values=None):
    """
    Create a corner plot using either corner library or seaborn fallback, with true values overlaid.
    
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
    true_values : np.ndarray, optional
        True parameter values to overlay
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
            title_kwargs={"fontsize": 10},
            truths=true_values,
            truth_color='red',
            quantiles=[0.16, 0.5, 0.84]
        )
        fig.suptitle(title, fontsize=14, y=0.95)
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Corner plot saved to {filename}")
        plt.show()
        
    except ImportError:
        # Fallback using seaborn
        df = pd.DataFrame(data_flat, columns=labels)
        g = sns.pairplot(df, corner=True, diag_kind="kde", 
                        plot_kws={"s": 5, "alpha": 0.4}, height=2)
        
        # Overlay true values if provided
        if true_values is not None:
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if i != j:  # Off-diagonal plots
                        g.axes[i, j].axvline(true_values[j], color='red', linestyle='--', alpha=0.8)
                        g.axes[i, j].axhline(true_values[i], color='red', linestyle='--', alpha=0.8)
                    else:  # Diagonal plots
                        g.axes[i, i].axvline(true_values[i], color='red', linestyle='--', alpha=0.8)
        
        g.fig.suptitle(title, fontsize=14)
        g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.95)
        g.fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Pair plot saved to {filename} (corner library not installed)")
        plt.show()

def plot_parameter_statistics(samples_full, true_parameters=None):
    """
    Print and plot parameter statistics for all 11 parameters.
    Also compare posterior nuisance distributions to their priors.
    
    Parameters:
    -----------
    samples_full : np.ndarray
        Full parameter samples (n_samples, 11) or (n_samples, n_chains, 11)
    true_parameters : np.ndarray, optional
        True parameter values for comparison
    """
    # Handle different sample formats
    if len(samples_full.shape) == 3:
        # Samples are in format (n_samples, n_chains, 11)
        # Flatten to (n_samples * n_chains, 11) for statistics
        samples_flat = samples_full.reshape(-1, 11)
        print(f"Stats Debug: samples in chain format, flattened shape = {samples_flat.shape}")
    else:
        # Samples are in flat format (n_samples, 11)
        samples_flat = samples_full
        print(f"Stats Debug: samples in flat format, shape = {samples_flat.shape}")
    
    # Parameter names (in order as defined in distributions.yaml)
    param_names = [
        "d_sample", "rho_cv_sample", "rho_cv_coupler", "rho_cv_ins",
        "d_coupler", "d_ins_pside", "d_ins_oside", "fwhm",
        "k_sample", "k_ins", "k_coupler"
    ]
    
    print("\n" + "="*80)
    print("FULL PARAMETER STATISTICS")
    print("="*80)
    
    # Print statistics table
    print(f"{'Parameter':<15} {'Posterior Mean':<15} {'Posterior Std':<15} {'True Value':<15} {'Recovery %':<10}")
    print("-" * 80)
    
    for i, name in enumerate(param_names):
        mean_val = samples_flat[:, i].mean()
        std_val = samples_flat[:, i].std()
        
        if true_parameters is not None:
            true_val = true_parameters[i]
            recovery_pct = abs(mean_val - true_val) / abs(true_val) * 100
            print(f"{name:<15} {mean_val:<15.3e} {std_val:<15.3e} {true_val:<15.3e} {recovery_pct:<10.1f}%")
        else:
            print(f"{name:<15} {mean_val:<15.3e} {std_val:<15.3e} {'N/A':<15} {'N/A':<10}")
    
    # Compare posterior nuisance distributions to priors
    print("\n" + "="*80)
    print("NUISANCE PARAMETER PRIOR vs POSTERIOR COMPARISON")
    print("="*80)
    
    # Get prior distributions
    from analysis.config_utils import get_param_defs_from_config, create_uqpy_distributions
    param_defs = get_param_defs_from_config(config_path="configs/distributions.yaml")
    uqpy_dists = create_uqpy_distributions(param_defs)
    
    print(f"{'Parameter':<15} {'Prior Mean':<15} {'Prior Std':<15} {'Posterior Mean':<15} {'Posterior Std':<15} {'Learning':<10}")
    print("-" * 90)
    
    for i in range(8):  # First 8 parameters are nuisance parameters
        name = param_names[i]
        
        # Get prior statistics by sampling from the distribution
        prior_dist = uqpy_dists[i]
        
        # Generate samples from prior to compute statistics
        try:
            prior_samples = prior_dist.rvs(nsamples=10000)
            prior_mean = np.mean(prior_samples)
            prior_std = np.std(prior_samples)
        except:
            prior_mean = np.nan
            prior_std = np.nan
        
        # Get posterior statistics
        post_mean = samples_flat[:, i].mean()
        post_std = samples_flat[:, i].std()
        
        # Check if posterior is learning (std smaller than prior)
        if np.isnan(prior_std):
            learning = "Unknown"
        elif post_std < prior_std * 0.9:
            learning = "Yes"
        elif post_std > prior_std * 1.1:
            learning = "No (wider)"
        else:
            learning = "No (same)"
        
        print(f"{name:<15} {prior_mean:<15.3e} {prior_std:<15.3e} {post_mean:<15.3e} {post_std:<15.3e} {learning:<10}")
    
    # Create summary plot for all parameters
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    # Create box plots for all parameters
    box_data = [samples_flat[:, i] for i in range(11)]
    bp = ax.boxplot(box_data, labels=param_names, patch_artist=True)
    
    # Color code the boxes
    colors = ['lightblue'] * 8 + ['lightgreen'] * 3  # Nuisance params in blue, k params in green
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Overlay true values if available
    if true_parameters is not None:
        for i, true_val in enumerate(true_parameters):
            ax.axhline(true_val, color='red', linestyle='--', alpha=0.7, xmin=(i+0.4)/11, xmax=(i+0.6)/11)
    
    ax.set_title("Full Parameter Distributions")
    ax.set_ylabel("Parameter Value")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("test_parameter_statistics.png", dpi=300, bbox_inches="tight")
    print("Parameter statistics plot saved to test_parameter_statistics.png")
    plt.show()
    
    # Create prior vs posterior comparison plot for nuisance parameters
    create_prior_posterior_comparison_plot(samples_flat, uqpy_dists, param_names[:8])

def plot_trace_plots(samples_full, n_walkers=24):
    """
    Plot trace plots to check convergence for all 11 parameters.
    
    Parameters:
    -----------
    samples_full : np.ndarray
        Full parameter samples (n_samples, 11) or (n_samples, n_chains, 11)
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
        samples_reshaped = samples_full[:nsamples_per_walker * n_walkers].reshape(n_walkers, nsamples_per_walker, 11)
        print(f"Trace Debug: reshaped flat samples, shape = {samples_reshaped.shape}")
    
    # Parameter names
    param_names = [
        "d_sample", "rho_cv_sample", "rho_cv_coupler", "rho_cv_ins",
        "d_coupler", "d_ins_pside", "d_ins_oside", "fwhm",
        "k_sample", "k_ins", "k_coupler"
    ]
    
    # Create subplots for all parameters
    fig, axes = plt.subplots(11, 1, figsize=(15, 25))
    
    for param_idx in range(11):
        ax = axes[param_idx]
        
        # Plot each walker's trace (show first 8 walkers for clarity)
        for walker_idx in range(min(8, n_walkers)):
            trace = samples_reshaped[walker_idx, :, param_idx]
            ax.plot(trace, alpha=0.6, linewidth=0.5)
        
        # Plot mean across walkers
        mean_trace = np.mean(samples_reshaped[:, :, param_idx], axis=0)
        ax.plot(mean_trace, 'k-', linewidth=2, label='Mean across walkers')
        
        ax.set_title(f'Trace Plot: {param_names[param_idx]}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Parameter Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("test_full_trace_plots.png", dpi=300, bbox_inches="tight")
    print("Full parameter trace plots saved to test_full_trace_plots.png")
    plt.show()

def main():
    """Main function to load and plot full MCMC results."""
    print("Loading full MCMC results...")
    
    samples_full, log_pdf_values, true_parameters, synthetic_data = load_mcmc_results()
    
    if samples_full is None:
        print("No MCMC results found. Please run test_full_parameter_mcmc.py first.")
        return
    
    # Compute convergence diagnostics
    print("\n" + "="*80)
    print("CONVERGENCE DIAGNOSTICS")
    print("="*80)
    
    # ESS and R-hat for all 11 parameters (using 24 walkers as in test_full_parameter_mcmc.py)
    n_walkers = 24
    ess_full = compute_ess_arviz(samples_full, n_walkers=n_walkers)
    rhat_full = compute_rhat_arviz(samples_full, n_walkers=n_walkers)
    
    param_names = [
        "d_sample", "rho_cv_sample", "rho_cv_coupler", "rho_cv_ins",
        "d_coupler", "d_ins_pside", "d_ins_oside", "fwhm",
        "k_sample", "k_ins", "k_coupler"
    ]
    
    print("\nAll Parameters:")
    for i, name in enumerate(param_names):
        print(f"  {name:15s}: ESS = {ess_full[i]:.0f}, R-hat = {rhat_full[i]:.3f}")
    
    # Check convergence criteria
    min_ess = float(np.min(ess_full))
    max_rhat = float(np.max(rhat_full))
    
    print(f"\nConvergence Summary:")
    print(f"  Minimum ESS: {min_ess:.0f} (should be > 200)")
    print(f"  Maximum R-hat: {max_rhat:.3f} (should be < 1.01)")
    
    if min_ess < 200:
        print("  ⚠️  WARNING: ESS too low - consider running more samples")
    if max_rhat > 1.01:
        print("  ⚠️  WARNING: R-hat too high - consider longer chains or better mixing")
    if min_ess >= 200 and max_rhat <= 1.01:
        print("  ✅ Convergence looks good!")
    
    # Print statistics and compare priors vs posteriors
    plot_parameter_statistics(samples_full, true_parameters)
    
    # Plot likelihood values if available
    if log_pdf_values is not None:
        print("\nCreating likelihood analysis plots...")
        plot_likelihood_values(samples_full, log_pdf_values)
    else:
        print("\nNo log-likelihood values found in test_full_mcmc_results.npz")
    
    # Create trace plots for convergence diagnostics
    print("\nCreating trace plots...")
    plot_trace_plots(samples_full, n_walkers=n_walkers)
    
    # Create corner plots
    print("\nCreating corner plots...")
    
    # All parameters corner plot
    all_labels = [
        r"$d_{\text{sample}}$", r"$\rho c_v^{\text{sample}}$", r"$\rho c_v^{\text{coupler}}$", r"$\rho c_v^{\text{ins}}$",
        r"$d_{\text{coupler}}$", r"$d_{\text{ins}}^{\text{p}}$", r"$d_{\text{ins}}^{\text{o}}$", r"$\text{FWHM}$",
        r"$k_{\text{sample}}$", r"$k_{\text{ins}}$", r"$k_{\text{coupler}}$"
    ]
    create_corner_plot_with_truths(samples_full, all_labels, "Full Parameter Posterior", "test_full_corner_plot.png", true_parameters)
    
    # κ parameters only corner plot
    k_indices = [8, 9, 10]  # k_sample, k_ins, k_coupler
    k_samples = samples_full[:, k_indices] if len(samples_full.shape) == 2 else samples_full.reshape(-1, 11)[:, k_indices]
    k_labels = [r"$k_{\text{sample}}$", r"$k_{\text{ins}}$", r"$k_{\text{coupler}}$"]
    k_true = true_parameters[k_indices] if true_parameters is not None else None
    create_corner_plot_with_truths(k_samples, k_labels, "κ Parameters Posterior", "test_k_corner_plot.png", k_true)
    
    # Nuisance parameters only corner plot
    nuisance_indices = list(range(8))  # First 8 parameters
    nuisance_samples = samples_full[:, nuisance_indices] if len(samples_full.shape) == 2 else samples_full.reshape(-1, 11)[:, nuisance_indices]
    nuisance_labels = [
        r"$d_{\text{sample}}$", r"$\rho c_v^{\text{sample}}$", r"$\rho c_v^{\text{coupler}}$", r"$\rho c_v^{\text{ins}}$",
        r"$d_{\text{coupler}}$", r"$d_{\text{ins}}^{\text{p}}$", r"$d_{\text{ins}}^{\text{o}}$", r"$\text{FWHM}$"
    ]
    nuisance_true = true_parameters[nuisance_indices] if true_parameters is not None else None
    create_corner_plot_with_truths(nuisance_samples, nuisance_labels, "Nuisance Parameters Posterior", "test_nuisance_corner_plot.png", nuisance_true)
    
    print("\nAll plots completed!")
    print("\nNew feature: Prior vs Posterior comparison for nuisance parameters")
    print("This shows whether the MCMC is learning about nuisance parameters or just exploring the prior space.")

if __name__ == "__main__":
    main() 