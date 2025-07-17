#!/usr/bin/env python3
"""
Plot MCMC results from log-space and real-space sampling.
Loads samples from both approaches and creates comparison plots.
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

def load_mcmc_results_logspace():
    """
    Load MCMC results from log-space and real-space saved files.
    """
    results = {}
    
    # Try to load log-space results
    try:
        data_log = np.load("mcmc_results_logspace.npz")
        results['logspace'] = {
            'samples': data_log['samples_logspace'],
            'log_pdf_values': data_log.get('log_pdf_values', None),
            'param_names': data_log.get('param_names', None),
            'sampling_space': data_log.get('sampling_space', 'logspace')
        }
        print(f"Loaded log-space results: {len(results['logspace']['samples'])} samples")
    except FileNotFoundError:
        print("Could not find mcmc_results_logspace.npz")
        results['logspace'] = None
    
    # Try to load real-space results
    try:
        data_real = np.load("mcmc_results_realspace.npz")
        results['realspace'] = {
            'samples': data_real['samples_realspace'],
            'log_pdf_values': data_real.get('log_pdf_values', None),
            'param_names': data_real.get('param_names', None),
            'sampling_space': data_real.get('sampling_space', 'realspace')
        }
        print(f"Loaded real-space results: {len(results['realspace']['samples'])} samples")
    except FileNotFoundError:
        print("Could not find mcmc_results_realspace.npz")
        results['realspace'] = None
    
    # Try to load original MCMC results for comparison
    try:
        data_orig = np.load("mcmc_results.npz")
        results['original'] = {
            'samples': data_orig['samples_full'],
            'log_pdf_values': data_orig.get('log_pdf_values', None),
            'param_names': None,  # Will be set from config
            'sampling_space': 'original'
        }
        print(f"Loaded original MCMC results: {len(results['original']['samples'])} samples")
    except FileNotFoundError:
        print("Could not find mcmc_results.npz")
        results['original'] = None
    
    return results

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

def plot_parameter_statistics_comparison(results, param_names):
    """
    Print and plot parameter statistics comparing different sampling approaches.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results from different sampling approaches
    param_names : list
        Names of all parameters
    """
    print("\n" + "="*80)
    print("PARAMETER STATISTICS COMPARISON")
    print("="*80)
    
    # Get parameter names from config if not provided
    if param_names is None:
        param_defs = get_param_defs_from_config()
        param_names = [param_def['name'] for param_def in param_defs]
    
    # Compare statistics across different approaches
    approaches = []
    means = []
    stds = []
    
    for approach_name, result in results.items():
        if result is not None:
            approaches.append(approach_name)
            
            # Handle different sample formats
            samples = result['samples']
            if len(samples.shape) == 3:
                samples_flat = samples.reshape(-1, samples.shape[2])
            else:
                samples_flat = samples
            
            # Compute statistics
            means.append([samples_flat[:, i].mean() for i in range(len(param_names))])
            stds.append([samples_flat[:, i].std() for i in range(len(param_names))])
    
    # Print comparison table
    for i, param_name in enumerate(param_names):
        print(f"\n{param_name}:")
        print(f"{'Approach':<15} {'Mean':<15} {'Std':<15}")
        print("-" * 50)
        for j, approach in enumerate(approaches):
            print(f"{approach:<15} {means[j][i]:<15.3e} {stds[j][i]:<15.3e}")
    
    # Create comparison plot for k parameters
    k_indices = [8, 9, 10]  # k_sample, k_ins, k_coupler
    k_names = ['k_sample', 'k_ins', 'k_coupler']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (k_idx, k_name) in enumerate(zip(k_indices, k_names)):
        ax = axes[i]
        
        # Plot distributions for each approach
        for j, approach in enumerate(approaches):
            samples = results[approach]['samples']
            if len(samples.shape) == 3:
                samples_flat = samples.reshape(-1, samples.shape[2])
            else:
                samples_flat = samples
            
            k_values = samples_flat[:, k_idx]
            
            # Create histogram
            ax.hist(k_values, bins=50, alpha=0.6, label=approach, density=True)
        
        ax.set_xlabel(k_name)
        ax.set_ylabel('Density')
        ax.set_title(f'Posterior Distribution: {k_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("parameter_comparison.png", dpi=300, bbox_inches="tight")
    print("Parameter comparison plot saved to parameter_comparison.png")
    plt.show()

def plot_trace_plots_comparison(results, param_names, n_walkers=24):
    """
    Plot trace plots comparing different sampling approaches.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results from different sampling approaches
    param_names : list
        Names of all parameters
    n_walkers : int
        Number of walkers used in the ensemble MCMC
    """
    # Get parameter names from config if not provided
    if param_names is None:
        param_defs = get_param_defs_from_config()
        param_names = [param_def['name'] for param_def in param_defs]
    
    # Focus on k parameters for trace plot comparison
    k_indices = [8, 9, 10]  # k_sample, k_ins, k_coupler
    k_names = ['k_sample', 'k_ins', 'k_coupler']
    
    fig, axes = plt.subplots(len(k_names), len(results), figsize=(5*len(results), 4*len(k_names)))
    
    if len(k_names) == 1:
        axes = axes.reshape(1, -1)
    if len(results) == 1:
        axes = axes.reshape(-1, 1)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (k_idx, k_name) in enumerate(zip(k_indices, k_names)):
        for j, (approach_name, result) in enumerate(results.items()):
            if result is not None:
                ax = axes[i, j]
                
                samples = result['samples']
                if len(samples.shape) == 3:
                    # Samples are in format (n_samples, n_chains, n_dimensions)
                    # Transpose to (n_chains, n_samples, n_dimensions) for easier plotting
                    samples_reshaped = samples.transpose(1, 0, 2)
                else:
                    # Samples are in flat format (n_samples, n_dimensions)
                    # Reshape to separate walkers
                    nsamples_per_walker = len(samples) // n_walkers
                    samples_reshaped = samples[:nsamples_per_walker * n_walkers].reshape(n_walkers, nsamples_per_walker, samples.shape[1])
                
                # Plot each walker's trace (show first 4 walkers for clarity)
                for walker_idx in range(min(4, n_walkers)):
                    trace = samples_reshaped[walker_idx, :, k_idx]
                    ax.plot(trace, alpha=0.6, linewidth=0.5, color=colors[j])
                
                # Plot mean across walkers
                mean_trace = np.mean(samples_reshaped[:, :, k_idx], axis=0)
                ax.plot(mean_trace, 'k-', linewidth=2, label='Mean across walkers')
                
                ax.set_title(f'{approach_name}: {k_name}')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Parameter Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("trace_plots_comparison.png", dpi=300, bbox_inches="tight")
    print("Trace plots comparison saved to trace_plots_comparison.png")
    plt.show()

def plot_ess_comparison(results, param_names, n_walkers=24):
    """
    Plot ESS comparison between different sampling approaches.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results from different sampling approaches
    param_names : list
        Names of all parameters
    n_walkers : int
        Number of walkers used in the ensemble MCMC
    """
    # Get parameter names from config if not provided
    if param_names is None:
        param_defs = get_param_defs_from_config()
        param_names = [param_def['name'] for param_def in param_defs]
    
    print("\n" + "="*80)
    print("EFFECTIVE SAMPLE SIZE (ESS) COMPARISON")
    print("="*80)
    
    approaches = []
    ess_values = []
    
    for approach_name, result in results.items():
        if result is not None:
            approaches.append(approach_name)
            ess = compute_ess_arviz(result['samples'], param_names, n_walkers=n_walkers)
            ess_values.append(ess)
            
            print(f"\n{approach_name.upper()} ESS:")
            for i, name in enumerate(param_names):
                print(f"  {name:<15}: {ess[i]:.0f}")
    
    # Create ESS comparison plot
    if len(approaches) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        x = np.arange(len(param_names))
        width = 0.8 / len(approaches)
        
        for i, approach in enumerate(approaches):
            ax.bar(x + i * width, ess_values[i], width, label=approach, alpha=0.8)
        
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Effective Sample Size (ESS)')
        ax.set_title('ESS Comparison Across Sampling Approaches')
        ax.set_xticks(x + width * (len(approaches) - 1) / 2)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig("ess_comparison.png", dpi=300, bbox_inches="tight")
        print("ESS comparison plot saved to ess_comparison.png")
        plt.show()

def main():
    """Main function to load and plot MCMC results comparison."""
    print("Loading MCMC results for comparison...")
    
    # Load all available results
    results = load_mcmc_results_logspace()
    
    if not any(results.values()):
        print("No MCMC results found. Please run uqpy_MCMC_logspace.py first.")
        return
    
    # Get parameter names from config
    param_defs = get_param_defs_from_config()
    param_names = [param_def['name'] for param_def in param_defs]
    
    print(f"Parameter names: {param_names}")
    
    # Print summary of available results
    print("\n" + "="*80)
    print("AVAILABLE RESULTS SUMMARY")
    print("="*80)
    for approach, result in results.items():
        if result is not None:
            samples = result['samples']
            if len(samples.shape) == 3:
                total_samples = samples.shape[0] * samples.shape[1]
                n_walkers = samples.shape[1]
            else:
                total_samples = len(samples)
                n_walkers = 24  # Default assumption
            
            print(f"{approach}: {total_samples} total samples, {n_walkers} walkers")
        else:
            print(f"{approach}: Not available")
    
    # Compute convergence diagnostics for each approach
    print("\n" + "="*80)
    print("CONVERGENCE DIAGNOSTICS COMPARISON")
    print("="*80)
    
    for approach_name, result in results.items():
        if result is not None:
            print(f"\n{approach_name.upper()}:")
            
            # ESS and R-hat
            ess = compute_ess_arviz(result['samples'], param_names, n_walkers=24)
            rhat = compute_rhat_arviz(result['samples'], param_names, n_walkers=24)
            
            print(f"  Minimum ESS: {np.min(ess):.0f}")
            print(f"  Maximum R-hat: {np.max(rhat):.3f}")
            
            # Check convergence criteria
            if np.min(ess) < 200:
                print("  ⚠️  WARNING: ESS too low - consider running more samples")
            if np.max(rhat) > 1.01:
                print("  ⚠️  WARNING: R-hat too high - consider longer chains or better mixing")
            if np.min(ess) >= 200 and np.max(rhat) <= 1.01:
                print("  ✅ Convergence looks good!")
    
    # Plot parameter statistics comparison
    print("\nCreating parameter statistics comparison...")
    plot_parameter_statistics_comparison(results, param_names)
    
    # Plot trace plots comparison
    print("\nCreating trace plots comparison...")
    plot_trace_plots_comparison(results, param_names, n_walkers=24)
    
    # Plot ESS comparison
    print("\nCreating ESS comparison...")
    plot_ess_comparison(results, param_names, n_walkers=24)
    
    # Create corner plots for each approach
    print("\nCreating corner plots...")
    
    for approach_name, result in results.items():
        if result is not None:
            samples = result['samples']
            
            # κ parameters corner plot (indices 8, 9, 10)
            k_indices = [8, 9, 10]  # k_sample, k_ins, k_coupler
            k_names = ['k_sample', 'k_ins', 'k_coupler']
            
            if len(samples.shape) == 3:
                k_samples = samples[:, :, k_indices]
            else:
                k_samples = samples[:, k_indices]
            
            if approach_name == 'logspace':
                k_labels = [r"$\log(k_{\text{sample}})$", r"$\log(k_{\text{ins}})$", r"$\log(k_{\text{coupler}})$"]
                title = f"Log-space Posterior - κ Parameters"
                filename = f"k_corner_plot_{approach_name}.png"
            else:
                k_labels = [r"$k_{\text{sample}}$", r"$k_{\text{ins}}$", r"$k_{\text{coupler}}$"]
                title = f"{approach_name.capitalize()} Posterior - κ Parameters"
                filename = f"k_corner_plot_{approach_name}.png"
            
            create_corner_plot(k_samples, k_labels, title, filename)
    
    print("\nAll comparison plots completed!")

if __name__ == "__main__":
    main() 