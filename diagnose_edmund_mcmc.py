#!/usr/bin/env python3
"""
Diagnostic script to analyze Edmund MCMC results and understand the k_ins issue.
"""

import numpy as np
import matplotlib.pyplot as plt
from analysis.config_utils import get_param_defs_from_config

def diagnose_edmund_mcmc():
    """Analyze Edmund MCMC results to understand the k_ins issue."""
    
    print("=" * 60)
    print("EDMUND MCMC DIAGNOSTIC")
    print("=" * 60)
    
    # Load MCMC results
    try:
        data = np.load('mcmc_results_edmund.npz')
        samples_full = data['samples_full']
        print(f"Loaded MCMC results with shape: {samples_full.shape}")
    except FileNotFoundError:
        print("Error: mcmc_results_edmund.npz not found")
        return
    
    # Get parameter definitions
    param_defs = get_param_defs_from_config(config_path="configs/distributions_edmund.yaml")
    param_names = [param_def['name'] for param_def in param_defs]
    
    print(f"\nParameter names: {param_names}")
    print(f"Number of parameters: {len(param_names)}")
    
    # Handle different sample formats
    if len(samples_full.shape) == 3:
        # 3D format: (n_samples, n_chains, n_dimensions)
        samples_flat = samples_full.reshape(-1, samples_full.shape[2])
        print(f"Flattened samples shape: {samples_flat.shape}")
    else:
        # 2D format: (n_samples, n_dimensions)
        samples_flat = samples_full
        print(f"Samples already flat, shape: {samples_flat.shape}")
    
    # Analyze each parameter
    print("\n" + "=" * 60)
    print("PARAMETER ANALYSIS")
    print("=" * 60)
    
    for i, name in enumerate(param_names):
        values = samples_flat[:, i]
        
        print(f"\n{name} (index {i}):")
        print(f"  Range: {values.min():.6e} to {values.max():.6e}")
        print(f"  Mean: {values.mean():.6e}")
        print(f"  Std: {values.std():.6e}")
        print(f"  Median: {np.median(values):.6e}")
        
        # Check if values are reasonable
        if name.startswith('k_'):
            if values.max() > 1000:
                print(f"  ⚠️  WARNING: {name} has very large values (>1000)")
            if values.min() < 0.1:
                print(f"  ⚠️  WARNING: {name} has very small values (<0.1)")
        
        # Check for parameter-specific issues
        if name == 'k_ins':
            print(f"  k_ins prior range: 1 to 30 W/(m·K)")
            if values.max() > 30:
                print(f"  ❌ ERROR: k_ins exceeds prior upper bound!")
            if values.min() < 1:
                print(f"  ❌ ERROR: k_ins below prior lower bound!")
    
    # Find k_ins index
    k_ins_idx = param_names.index('k_ins')
    k_sample_idx = param_names.index('k_sample')
    
    print(f"\n" + "=" * 60)
    print("THERMAL CONDUCTIVITY ANALYSIS")
    print("=" * 60)
    
    k_ins_values = samples_flat[:, k_ins_idx]
    k_sample_values = samples_flat[:, k_sample_idx]
    
    print(f"k_sample (index {k_sample_idx}):")
    print(f"  Range: {k_sample_values.min():.2f} to {k_sample_values.max():.2f} W/(m·K)")
    print(f"  Mean: {k_sample_values.mean():.2f} W/(m·K)")
    print(f"  Prior range: 25 to 100 W/(m·K)")
    
    print(f"\nk_ins (index {k_ins_idx}):")
    print(f"  Range: {k_ins_values.min():.2f} to {k_ins_values.max():.2f} W/(m·K)")
    print(f"  Mean: {k_ins_values.mean():.2f} W/(m·K)")
    print(f"  Prior range: 1 to 30 W/(m·K)")
    
    # Check if k_ins is being pulled to the lower bound
    k_ins_percentiles = np.percentile(k_ins_values, [5, 25, 50, 75, 95])
    print(f"  Percentiles: {k_ins_percentiles}")
    
    # Check correlation between k parameters
    correlation = np.corrcoef(k_sample_values, k_ins_values)[0, 1]
    print(f"\nCorrelation between k_sample and k_ins: {correlation:.3f}")
    
    # Create diagnostic plots
    print(f"\n" + "=" * 60)
    print("CREATING DIAGNOSTIC PLOTS")
    print("=" * 60)
    
    # Plot 1: k_ins distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(k_ins_values, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(1, color='red', linestyle='--', label='Prior lower bound')
    plt.axvline(30, color='red', linestyle='--', label='Prior upper bound')
    plt.xlabel('k_ins (W/(m·K))')
    plt.ylabel('Frequency')
    plt.title('k_ins Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: k_sample vs k_ins scatter
    plt.subplot(1, 3, 2)
    plt.scatter(k_sample_values, k_ins_values, alpha=0.5, s=1)
    plt.xlabel('k_sample (W/(m·K))')
    plt.ylabel('k_ins (W/(m·K))')
    plt.title('k_sample vs k_ins')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: All parameters box plot
    plt.subplot(1, 3, 3)
    box_data = [samples_flat[:, i] for i in range(len(param_names))]
    plt.boxplot(box_data, labels=param_names)
    plt.xticks(rotation=45)
    plt.title('All Parameters')
    plt.ylabel('Parameter Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("edmund_mcmc_diagnostic.png", dpi=300, bbox_inches='tight')
    print("Diagnostic plot saved to edmund_mcmc_diagnostic.png")
    plt.show()
    
    # Check for potential issues
    print(f"\n" + "=" * 60)
    print("POTENTIAL ISSUES")
    print("=" * 60)
    
    issues_found = []
    
    # Check if k_ins is at the lower bound
    if k_ins_values.mean() < 2.0:
        issues_found.append("k_ins is being pulled to the lower bound")
    
    # Check if k_ins has very low variance
    if k_ins_values.std() < 0.1:
        issues_found.append("k_ins has very low variance (may be stuck)")
    
    # Check if k_ins is outside prior bounds
    if k_ins_values.max() > 30 or k_ins_values.min() < 1:
        issues_found.append("k_ins values outside prior bounds")
    
    # Check for any parameter with extremely large values
    for i, name in enumerate(param_names):
        values = samples_flat[:, i]
        if values.max() > 1e6:
            issues_found.append(f"{name} has extremely large values (>1e6)")
    
    if issues_found:
        print("Issues found:")
        for issue in issues_found:
            print(f"  ❌ {issue}")
    else:
        print("No obvious issues detected")
    
    # Suggestions for fixing the issue
    print(f"\n" + "=" * 60)
    print("SUGGESTIONS")
    print("=" * 60)
    
    print("If k_ins is being pulled to the lower bound:")
    print("1. Check if the surrogate model is correctly trained for low k_ins values")
    print("2. Verify that the experimental data is properly normalized")
    print("3. Check if the likelihood function is correctly implemented")
    print("4. Consider if the prior range for k_ins is appropriate")
    print("5. Check if there are parameter correlations causing issues")
    print("6. Verify that the surrogate model predictions are reasonable for low k_ins")

if __name__ == "__main__":
    diagnose_edmund_mcmc() 