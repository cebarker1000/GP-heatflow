import sys
import numpy as np
import pandas as pd
from UQpy.sampling.stratified_sampling import LatinHypercubeSampling
from analysis.uq_wrapper import (run_single_simulation, run_batch_simulations, save_batch_results, 
                       load_batch_results, build_fpca_model, save_fpca_model, 
                       recast_training_data_to_fpca)
from analysis.config_utils import load_all_from_config, create_uqpy_distributions
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta


# Load configuration from YAML file
param_defs, PARAM_MAPPING, sampling_config, output_config = load_all_from_config()

# Create UQpy distributions for each parameter
distributions = create_uqpy_distributions(param_defs)

# Get sampling parameters from config
n_samples = sampling_config.get('n_samples', 200)

# Latin Hypercube Sampling
lhs = LatinHypercubeSampling(distributions=distributions, nsamples=n_samples)
samples = lhs.samples

# Note: UQpy Lognormal distribution already returns values in the correct space
# No exponentiation needed for lognormal parameters

print("LHS samples (physical space):")
print(samples)

header = ",".join([p["name"] for p in param_defs])
samples_file = output_config.get('samples_file', 'outputs/initial_train_set.csv')
np.savetxt(samples_file, samples, delimiter=",", header=header, comments='')


def plot_parameter_distributions(samples, param_defs, output_dir="outputs"):
    """
    Create histograms of the sampled parameter distributions.
    
    Args:
        samples: Array of shape (n_samples, n_params) containing the sampled values
        param_defs: List of parameter definition dictionaries
        output_dir: Directory to save the plot
    """
    n_params = len(param_defs)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for i, (param_def, ax) in enumerate(zip(param_defs, axes_flat)):
        param_name = param_def["name"]
        param_values = samples[:, i]
        
        # Filter out infinite values for plotting
        finite_values = param_values[np.isfinite(param_values)]
        if len(finite_values) == 0:
            ax.text(0.5, 0.5, 'All values infinite', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{param_name}\n({param_def["type"]}) - ALL INFINITE', fontsize=10)
            continue
        
        # Create histogram
        ax.hist(finite_values, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add theoretical distribution if possible
        if param_def["type"] == "lognormal":
            # Generate theoretical lognormal distribution
            mu = np.log(param_def["center"])
            sigma = param_def["sigma_log"]
            x_theoretical = np.linspace(finite_values.min(), finite_values.max(), 100)
            y_theoretical = (1 / (x_theoretical * sigma * np.sqrt(2 * np.pi))) * \
                           np.exp(-0.5 * ((np.log(x_theoretical) - mu) / sigma) ** 2)
            # Scale to match histogram
            y_theoretical = y_theoretical * len(finite_values) * (finite_values.max() - finite_values.min()) / 30
            ax.plot(x_theoretical, y_theoretical, 'r-', linewidth=2, label='Theoretical')
            
        elif param_def["type"] == "uniform":
            # Add uniform distribution line
            low, high = param_def["low"], param_def["high"]
            height = len(finite_values) / 30  # Approximate histogram height
            ax.plot([low, high], [height, height], 'r-', linewidth=2, label='Theoretical')
        
        # Add statistics
        mean_val = np.mean(finite_values)
        std_val = np.std(finite_values)
        ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2e}')
        
        # Formatting
        ax.set_title(f'{param_name}\n({param_def["type"]})', fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Use scientific notation for x-axis if values are very small or large
        if mean_val < 1e-3 or mean_val > 1e6:
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Hide unused subplots
    for i in range(n_params, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = f"{output_dir}/parameter_distributions.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Parameter distribution plot saved to: {plot_file}")
    
    # Also create a summary statistics table
    print("\nParameter Distribution Summary:")
    print("-" * 80)
    print(f"{'Parameter':<15} {'Type':<12} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-" * 80)
    
    for i, param_def in enumerate(param_defs):
        param_values = samples[:, i]
        finite_values = param_values[np.isfinite(param_values)]
        
        if len(finite_values) == 0:
            print(f"{param_def['name']:<15} {param_def['type']:<12} {'ALL INFINITE':<15} {'ALL INFINITE':<15} {'ALL INFINITE':<15} {'ALL INFINITE':<15}")
        else:
            mean_val = np.mean(finite_values)
            std_val = np.std(finite_values)
            min_val = np.min(finite_values)
            max_val = np.max(finite_values)
            
            print(f"{param_def['name']:<15} {param_def['type']:<12} {mean_val:<15.2e} {std_val:<15.2e} {min_val:<15.2e} {max_val:<15.2e}")
    
    plt.show()


def plot_parameter_correlations(samples, param_defs, output_dir="outputs"):
    """
    Create correlation matrix plot of the sampled parameters.
    
    Args:
        samples: Array of shape (n_samples, n_params) containing the sampled values
        param_defs: List of parameter definition dictionaries
        output_dir: Directory to save the plot
    """
    import seaborn as sns
    
    # Create correlation matrix
    param_names = [p["name"] for p in param_defs]
    df = pd.DataFrame(samples, columns=param_names)
    corr_matrix = df.corr()
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Upper triangle mask
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Parameter Correlation Matrix (LHS Sampling)', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Save the plot
    correlation_plot_file = f"{output_dir}/parameter_correlations.png"
    plt.savefig(correlation_plot_file, dpi=300, bbox_inches='tight')
    print(f"Parameter correlation plot saved to: {correlation_plot_file}")
    plt.show()


# Create parameter distribution plots
print("\n" + "="*60)
print("CREATING PARAMETER DISTRIBUTION PLOTS")
print("="*60)
distribution_plot_file = output_config.get('distribution_plot_file', 'outputs/parameter_distributions.png')
plot_parameter_distributions(samples, param_defs, output_dir="outputs")

# Create parameter correlation plot
print("\n" + "="*60)
print("CREATING PARAMETER CORRELATION PLOT")
print("="*60)
plot_parameter_correlations(samples, param_defs, output_dir="outputs")

if __name__ == "__main__":
    
    # Run batch simulations
    print("\nRunning batch simulations...")
    print(f"Running {len(samples)} simulations...")
    
    # Timing variables
    start_time = time.time()
    simulation_times = []
    
    # Add progress callback with timing
    def progress_callback(current, total):
        if current == 0:
            # First simulation starting
            print(f"Starting simulation {current + 1}/{total}...")
        else:
            # Calculate time for previous simulation
            sim_time = time.time() - start_time - sum(simulation_times)
            simulation_times.append(sim_time)
            
            # Calculate average time per simulation
            avg_time = np.mean(simulation_times)
            
            # Calculate remaining simulations and predicted time
            remaining_sims = total - current
            predicted_remaining = avg_time * remaining_sims
            
            # Format times nicely
            sim_time_str = str(timedelta(seconds=int(sim_time)))
            avg_time_str = str(timedelta(seconds=int(avg_time)))
            remaining_str = str(timedelta(seconds=int(predicted_remaining)))
            
            print(f"Simulation {current}/{total} completed in {sim_time_str} (avg: {avg_time_str})")
            print(f"Starting simulation {current + 1}/{total}... (est. remaining: {remaining_str})")
    
    results = run_batch_simulations(samples, param_defs, PARAM_MAPPING, suppress_print=True, progress_callback=progress_callback)
    
    # Final timing summary
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    avg_time = np.mean(simulation_times) if simulation_times else 0
    avg_time_str = str(timedelta(seconds=int(avg_time)))
    
    print(f"\nAll simulations completed!")
    print(f"Total time: {total_time_str}")
    print(f"Average time per simulation: {avg_time_str}")
    
    # Count successful vs failed simulations
    successful = sum(1 for r in results if 'error' not in r)
    failed = len(results) - successful
    print(f"\nSimulation Summary:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        # Save results
        results_file = output_config.get('results_file', 'outputs/uq_batch_results.npz')
        save_batch_results(results, param_defs, output_file=results_file)
        
        # Load and verify the saved data
        print("\nVerifying saved data...")
        loaded_data = load_batch_results(results_file)
        print(f"Loaded data keys: {list(loaded_data.keys())}")
        print(f"Oside curves shape: {loaded_data['oside_curves'].shape}")
        print(f"Parameters shape: {loaded_data['parameters'].shape}")
        print(f"Parameter names: {loaded_data['parameter_names']}")
        
        # Show some statistics about the oside curves
        oside_curves = loaded_data['oside_curves']
        valid_curves = oside_curves[~np.isnan(oside_curves).any(axis=1)]
        print(f"Valid oside curves: {len(valid_curves)}")
        if len(valid_curves) > 0:
            print(f"Oside curve statistics:")
            print(f"  Mean max value: {np.mean(np.max(valid_curves, axis=1)):.3f}")
            print(f"  Mean min value: {np.mean(np.min(valid_curves, axis=1)):.3f}")
            print(f"  Mean range: {np.mean(np.max(valid_curves, axis=1) - np.min(valid_curves, axis=1)):.3f}")
        
        # Step 2: Build FPCA model and recast training data
        print("\n" + "="*60)
        print("STEP 2: BUILDING FPCA MODEL AND RECASTING DATA")
        print("="*60)
        
        # Build FPCA model
        print("\nBuilding FPCA model...")
        fpca_model = build_fpca_model(
            input_file=results_file,
            min_components=4,
            variance_threshold=0.99
        )
        
        # Save FPCA model
        print("\nSaving FPCA model...")
        fpca_model_file = output_config.get('fpca_model_file', 'outputs/fpca_model.npz')
        save_fpca_model(fpca_model, fpca_model_file)
        
        # Recast training data to FPCA space
        print("\nRecasting training data to FPCA space...")
        fpca_training_file = output_config.get('fpca_training_file', 'outputs/training_data_fpca.npz')
        recast_data = recast_training_data_to_fpca(
            input_file=results_file,
            fpca_model=fpca_model,
            output_file=fpca_training_file
        )
        
        print(f"\nFPCA Analysis Summary:")
        print(f"  Number of components: {fpca_model['n_components']}")
        print(f"  Explained variance: {fpca_model['cumulative_variance'][-1]:.4f}")
        print(f"  Training data shape: {recast_data['parameters'].shape}")
        print(f"  FPCA scores shape: {recast_data['fpca_scores'].shape}")
        
        # Show FPCA score statistics
        fpca_scores = recast_data['fpca_scores']
        print(f"\nFPCA Score Statistics:")
        for i in range(fpca_model['n_components']):
            scores_i = fpca_scores[:, i]
            print(f"  PC{i+1}: mean={np.mean(scores_i):.4f}, std={np.std(scores_i):.4f}")
        
    else:
        print("No successful simulations to save!")
