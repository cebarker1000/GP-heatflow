#!/usr/bin/env python3
"""
Minimal script to plot histograms for specific parameters from Edmund MCMC results.
Usage: python plot_parameter_histogram.py <parameter_name> [x_min] [x_max]
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from analysis.config_utils import get_param_defs_from_config

def plot_parameter_histogram(parameter_name, x_min=None, x_max=None, bins=50):
    """
    Plot histogram for a specific parameter from Edmund MCMC results.
    
    Parameters:
    -----------
    parameter_name : str
        Name of the parameter to plot (e.g., 'k_sample', 'd_sample', etc.)
    x_min : float, optional
        Minimum x value for histogram range
    x_max : float, optional
        Maximum x value for histogram range
    bins : int
        Number of histogram bins
    """
    
    # Load MCMC results
    try:
        data = np.load('mcmc_results_edmund.npz')
        samples_full = data['samples_full']
        print(f"Loaded MCMC results with shape: {samples_full.shape}")
    except FileNotFoundError:
        print("Error: mcmc_results_edmund.npz not found. Please run uqpy_MCMC_edmund.py first.")
        return
    
    # Get parameter names from Edmund config
    param_defs = get_param_defs_from_config(config_path="configs/distributions_edmund.yaml")
    param_names = [param_def['name'] for param_def in param_defs]
    
    print(f"Available parameters: {param_names}")
    
    # Find parameter index
    try:
        param_idx = param_names.index(parameter_name)
    except ValueError:
        print(f"Error: Parameter '{parameter_name}' not found.")
        print(f"Available parameters: {param_names}")
        return
    
    # Extract parameter samples (handle both 2D and 3D formats)
    if len(samples_full.shape) == 3:
        # 3D format: (n_samples, n_chains, n_dimensions)
        param_samples = samples_full[:, :, param_idx].flatten()
        print(f"Extracted {len(param_samples)} samples from 3D format")
    else:
        # 2D format: (n_samples, n_dimensions)
        param_samples = samples_full[:, param_idx]
        print(f"Extracted {len(param_samples)} samples from 2D format")
    
    # Filter out any infinite or NaN values
    finite_mask = np.isfinite(param_samples)
    param_samples = param_samples[finite_mask]
    
    if len(param_samples) == 0:
        print("Error: No finite values found for this parameter.")
        return
    
    print(f"Parameter: {parameter_name}")
    print(f"Sample statistics:")
    print(f"  Mean: {np.mean(param_samples):.6e}")
    print(f"  Std:  {np.std(param_samples):.6e}")
    print(f"  Min:  {np.min(param_samples):.6e}")
    print(f"  Max:  {np.max(param_samples):.6e}")
    
    # Set x range if not provided
    if x_min is None:
        x_min = np.min(param_samples)
    if x_max is None:
        x_max = np.max(param_samples)
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    
    # Filter samples within the specified range
    range_mask = (param_samples >= x_min) & (param_samples <= x_max)
    filtered_samples = param_samples[range_mask]
    
    if len(filtered_samples) == 0:
        print(f"Error: No samples found in range [{x_min}, {x_max}]")
        return
    
    plt.hist(filtered_samples, bins=bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add vertical line for mean
    mean_val = np.mean(filtered_samples)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_val:.6e}')
    
    # Add vertical line for median
    median_val = np.median(filtered_samples)
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                label=f'Median: {median_val:.6e}')
    
    plt.xlabel(f'{parameter_name}')
    plt.ylabel('Frequency')
    plt.title(f'Edmund MCMC: {parameter_name} Distribution\n'
              f'Range: [{x_min:.6e}, {x_max:.6e}] | '
              f'Samples: {len(filtered_samples)}/{len(param_samples)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis for scientific notation if needed
    if x_max / x_min > 1000:
        plt.gca().ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()
    
    print(f"Histogram plotted for {parameter_name}")
    print(f"Range: [{x_min:.6e}, {x_max:.6e}]")
    print(f"Samples in range: {len(filtered_samples)}/{len(param_samples)}")

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python plot_parameter_histogram.py <parameter_name> [x_min] [x_max]")
        print("\nAvailable parameters:")
        param_defs = get_param_defs_from_config(config_path="configs/distributions_edmund.yaml")
        param_names = [param_def['name'] for param_def in param_defs]
        for name in param_names:
            print(f"  {name}")
        return
    
    parameter_name = sys.argv[1]
    
    # Parse optional x range
    x_min = None
    x_max = None
    
    if len(sys.argv) >= 3:
        try:
            x_min = float(sys.argv[2])
        except ValueError:
            print(f"Error: Invalid x_min value: {sys.argv[2]}")
            return
    
    if len(sys.argv) >= 4:
        try:
            x_max = float(sys.argv[3])
        except ValueError:
            print(f"Error: Invalid x_max value: {sys.argv[3]}")
            return
    
    # Plot the histogram
    plot_parameter_histogram(parameter_name, x_min, x_max)

if __name__ == "__main__":
    main() 