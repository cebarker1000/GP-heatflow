#!/usr/bin/env python3
"""
Plot MCMC results from saved outputs.
Loads samples and creates corner plots for κ parameters only.
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

def compute_ess_arviz(samples, n_walkers=8):
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
        
        # Create InferenceData with proper structure
        posterior_dict = {
            "k_sample": chains[:, :, 0],
            "k_ins": chains[:, :, 1], 
            "k_coupler": chains[:, :, 2]
        }
        
        idata = az.from_dict(posterior=posterior_dict)
        
        # Compute ESS using ArviZ's built-in method
        ess_bulk = az.ess(idata, method="bulk")
        
        ess_values = np.array([
            ess_bulk["k_sample"].values,
            ess_bulk["k_ins"].values,
            ess_bulk["k_coupler"].values
        ])
        
        print(f"ESS Debug: computed ESS values = {ess_values}")
        
        return ess_values
        
    except ImportError:
        print("ArviZ not available, skipping ESS calculation")
        return np.array([np.nan, np.nan, np.nan])
    except Exception as e:
        print(f"Error computing ESS: {e}")
        return np.array([np.nan, np.nan, np.nan])

def compute_rhat_arviz(samples, n_walkers=8):
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
        
        # Create InferenceData with proper structure
        posterior_dict = {
            "k_sample": chains[:, :, 0],
            "k_ins": chains[:, :, 1],
            "k_coupler": chains[:, :, 2]
        }
        
        idata = az.from_dict(posterior=posterior_dict)
        
        # Compute R-hat using ArviZ's built-in method
        rhat = az.rhat(idata)
        
        rhat_values = np.array([
            rhat["k_sample"].values,
            rhat["k_ins"].values,
            rhat["k_coupler"].values
        ])
        
        return rhat_values
        
    except ImportError:
        print("ArviZ not available, skipping R-hat calculation")
        return np.array([np.nan, np.nan, np.nan])
    except Exception as e:
        print(f"Error computing R-hat: {e}")
        return np.array([np.nan, np.nan, np.nan])

def plot_likelihood_values(samples_kappa, log_pdf_values):
    """
    Plot likelihood values to check if MCMC is exploring high-likelihood regions.
    
    Parameters:
    -----------
    samples_kappa : np.ndarray
        κ parameter samples (n_samples, 3) or (n_samples, n_chains, 3)
    log_pdf_values : np.ndarray
        Log-likelihood values for each sample
    """
    # Handle different sample formats
    if len(samples_kappa.shape) == 3:
        # Samples are in format (n_samples, n_chains, 3)
        # Flatten to (n_samples * n_chains, 3) for plotting
        samples_flat = samples_kappa.reshape(-1, 3)
        print(f"Likelihood Debug: samples in chain format, flattened shape = {samples_flat.shape}")
    else:
        # Samples are in flat format (n_samples, 3)
        samples_flat = samples_kappa
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
    
    # Plot 4: Likelihood vs k_sample (first parameter)
    scatter = axes[1, 1].scatter(samples_flat[:, 0], likelihood_values, 
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
        data = np.load("test_full_mcmc_results.npz")
        samples_kappa = data['samples_kappa']
        log_pdf_values = data.get('log_pdf_values', None)
        
        print(f"Loaded {len(samples_kappa)} accepted κ samples")
        if log_pdf_values is not None:
            print(f"Loaded {len(log_pdf_values)} log-likelihood values")
        
        return samples_kappa, log_pdf_values
        
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

def plot_parameter_statistics(samples_kappa):
    """
    Print and plot parameter statistics for κ parameters only.
    
    Parameters:
    -----------
    samples_kappa : np.ndarray
        κ parameter samples (n_samples, 3) or (n_samples, n_chains, 3)
    """
    # Handle different sample formats
    if len(samples_kappa.shape) == 3:
        # Samples are in format (n_samples, n_chains, 3)
        # Flatten to (n_samples * n_chains, 3) for statistics
        samples_flat = samples_kappa.reshape(-1, 3)
        print(f"Stats Debug: samples in chain format, flattened shape = {samples_flat.shape}")
    else:
        # Samples are in flat format (n_samples, 3)
        samples_flat = samples_kappa
        print(f"Stats Debug: samples in flat format, shape = {samples_flat.shape}")
    
    # κ parameter statistics
    k_names = ["k_sample", "k_ins", "k_coupler"]
    print("\n" + "="*50)
    print("κ PARAMETER STATISTICS")
    print("="*50)
    for i, name in enumerate(k_names):
        mean_val = samples_flat[:, i].mean()
        std_val = samples_flat[:, i].std()
        print(f"{name:12s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Create summary plot for κ parameters only
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # κ parameters
    ax.boxplot([samples_flat[:, i] for i in range(3)], labels=k_names)
    ax.set_title("κ Parameter Distributions")
    ax.set_ylabel("Thermal Conductivity (W/m/K)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("parameter_statistics.png", dpi=300, bbox_inches="tight")
    print("Parameter statistics plot saved to parameter_statistics.png")
    plt.show()

def plot_trace_plots(samples_kappa, n_walkers=8):
    """
    Plot trace plots to check convergence for κ parameters only.
    
    Parameters:
    -----------
    samples_kappa : np.ndarray
        κ parameter samples (n_samples, 3) or (n_samples, n_chains, 3)
    n_walkers : int
        Number of walkers used in the ensemble MCMC
    """
    # Handle different sample formats
    if len(samples_kappa.shape) == 3:
        # Samples are in format (n_samples, n_chains, 3)
        # Transpose to (n_chains, n_samples, 3) for easier plotting
        samples_reshaped = samples_kappa.transpose(1, 0, 2)  # (n_chains, n_samples, 3)
        print(f"Trace Debug: samples in chain format, shape = {samples_kappa.shape}")
    else:
        # Samples are in flat format (n_samples, 3)
        # Reshape to separate walkers
        nsamples_per_walker = len(samples_kappa) // n_walkers
        samples_reshaped = samples_kappa[:nsamples_per_walker * n_walkers].reshape(n_walkers, nsamples_per_walker, 3)
        print(f"Trace Debug: reshaped flat samples, shape = {samples_reshaped.shape}")
    
    # κ parameters trace plots
    k_names = ["k_sample", "k_ins", "k_coupler"]
    colors = ['blue', 'red', 'green']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for param_idx in range(3):
        ax = axes[param_idx]
        
        # Plot each walker's trace (show first 8 walkers for clarity)
        for walker_idx in range(min(8, n_walkers)):
            trace = samples_reshaped[walker_idx, :, param_idx]
            ax.plot(trace, alpha=0.6, linewidth=0.5, color=colors[param_idx])
        
        # Plot mean across walkers
        mean_trace = np.mean(samples_reshaped[:, :, param_idx], axis=0)
        ax.plot(mean_trace, 'k-', linewidth=2, label='Mean across walkers')
        
        ax.set_title(f'κ Trace Plot: {k_names[param_idx]}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Parameter Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("kappa_trace_plots.png", dpi=300, bbox_inches="tight")
    print("κ trace plots saved to kappa_trace_plots.png")
    plt.show()

def main():
    """Main function to load and plot MCMC results."""
    print("Loading MCMC results...")
    
    samples_kappa, log_pdf_values = load_mcmc_results()
    
    if samples_kappa is None:
        print("No MCMC results found. Please run uqpy_MCMC.py first.")
        return
    
    # Compute convergence diagnostics
    print("\n" + "="*50)
    print("CONVERGENCE DIAGNOSTICS")
    print("="*50)
    
    # ESS and R-hat for κ parameters (using 8 walkers as in uqpy_MCMC.py)
    n_walkers = 8
    ess_kappa = compute_ess_arviz(samples_kappa, n_walkers=n_walkers)
    rhat_kappa = compute_rhat_arviz(samples_kappa, n_walkers=n_walkers)
    
    k_names = ["k_sample", "k_ins", "k_coupler"]
    print("\nκ Parameters:")
    for i, name in enumerate(k_names):
        print(f"  {name:12s}: ESS = {ess_kappa[i]:.0f}, R-hat = {rhat_kappa[i]:.3f}")
    
    # Check convergence criteria
    min_ess = float(np.min(ess_kappa))
    max_rhat = float(np.max(rhat_kappa))
    
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
    plot_parameter_statistics(samples_kappa)
    
    # Plot likelihood values if available
    if log_pdf_values is not None:
        print("\nCreating likelihood analysis plots...")
        plot_likelihood_values(samples_kappa, log_pdf_values)
    else:
        print("\nNo log-likelihood values found in mcmc_results.npz")
        print("To include likelihood analysis, modify uqpy_MCMC.py to save log_pdf_values")
    
    # Create trace plots for convergence diagnostics
    print("\nCreating trace plots...")
    plot_trace_plots(samples_kappa, n_walkers=n_walkers)
    
    # Create corner plots
    print("\nCreating corner plots...")
    
    # κ parameters corner plot
    k_labels = [r"$k_{\text{sample}}$", r"$k_{\text{ins}}$", r"$k_{\text{coupler}}$"]
    create_corner_plot(samples_kappa, k_labels, "κ Parameters Posterior", "kappa_corner_plot.png")
    
    print("\nAll plots completed!")

if __name__ == "__main__":
    main() 