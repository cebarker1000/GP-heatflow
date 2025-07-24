import sys
import numpy as np
import pandas as pd
import argparse
import yaml
import os
from UQpy.sampling.stratified_sampling import LatinHypercubeSampling
from analysis.uq_wrapper import (run_single_simulation, run_batch_simulations, save_batch_results, 
                       load_batch_results, build_fpca_model, save_fpca_model, 
                       recast_training_data_to_fpca)
from analysis.config_utils import load_all_from_config, create_uqpy_distributions
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta


def plot_experimental_data_verification(config_path, output_dir="outputs"):
    """
    Load and plot the experimental data curves from the heating file specified in the config.
    This allows verification that the correct heating file is being used.
    """
    print("\n" + "="*60)
    print("VERIFYING EXPERIMENTAL DATA FILE")
    print("="*60)
    
    # Load the config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get the heating file path
    heating_file = config['heating']['file']
    print(f"Heating file specified in config: {heating_file}")
    
    # Check if file exists
    if not os.path.exists(heating_file):
        print(f"ERROR: Heating file not found: {heating_file}")
        return False
    
    # Load the experimental data
    try:
        df = pd.read_csv(heating_file)
        print(f"Successfully loaded experimental data from: {heating_file}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for required columns
        required_columns = ['time']
        if 'oside' in df.columns:
            required_columns.append('oside')
        if 'temp' in df.columns:
            required_columns.append('temp')
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"WARNING: Missing columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
        
        # Convert time to numeric and sort
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df = df.sort_values('time').dropna(subset=['time'])
        
        # Create the plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Experimental Data Verification\nFile: {heating_file}', fontsize=14)
        
        # Plot time vs temp (if available)
        if 'temp' in df.columns:
            df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
            df_temp = df.dropna(subset=['temp'])
            
            axes[0].plot(df_temp['time'] * 1e6, df_temp['temp'], 'b-', linewidth=2, label='temp')
            axes[0].set_xlabel('Time (μs)')
            axes[0].set_ylabel('Temperature (K)')
            axes[0].set_title('Temperature vs Time')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            print(f"Temperature data: {len(df_temp)} points")
            print(f"Time range: {df_temp['time'].min()*1e6:.3f} to {df_temp['time'].max()*1e6:.3f} μs")
            print(f"Temperature range: {df_temp['temp'].min():.2f} to {df_temp['temp'].max():.2f} K")
        else:
            axes[0].text(0.5, 0.5, 'No temp column found', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Temperature vs Time (No Data)')
        
        # Plot time vs oside (if available)
        if 'oside' in df.columns:
            df['oside'] = pd.to_numeric(df['oside'], errors='coerce')
            df_oside = df.dropna(subset=['oside'])
            
            axes[1].plot(df_oside['time'] * 1e6, df_oside['oside'], 'r-', linewidth=2, label='oside')
            axes[1].set_xlabel('Time (μs)')
            axes[1].set_ylabel('Temperature (K)')
            axes[1].set_title('Oside vs Time')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            
            print(f"Oside data: {len(df_oside)} points")
            print(f"Time range: {df_oside['time'].min()*1e6:.3f} to {df_oside['time'].max()*1e6:.3f} μs")
            print(f"Oside range: {df_oside['oside'].min():.2f} to {df_oside['oside'].max():.2f} K")
        else:
            axes[1].text(0.5, 0.5, 'No oside column found', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Oside vs Time (No Data)')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_dir, 'experimental_data_verification.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Experimental data verification plot saved to: {plot_file}")
        
        # Show the plot
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"ERROR loading experimental data: {e}")
        return False


def plot_test_simulation_verification(sample, param_defs, param_mapping, config_path, output_dir="outputs"):
    """
    Run a single test simulation and compare the oside curve with experimental data.
    This allows verification that the simulation setup is working correctly.
    """
    print("\n" + "="*60)
    print("RUNNING TEST SIMULATION VERIFICATION")
    print("="*60)
    
    # Load the config file to get experimental data path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get experimental data path - try multiple possible locations
    experimental_file = None
    possible_paths = [
        config.get('output', {}).get('analysis', {}).get('experimental_data_file'),
        config.get('output', {}).get('experimental_data_file'),
        "data/experimental/geballe/geballe_80GPa_1.csv",  # Common fallback
        "data/experimental/geballe/geballe_80GPa_2.csv",  # Common fallback
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            experimental_file = path
            break
    
    if not experimental_file:
        print("WARNING: No experimental data file found. Tried paths:")
        for path in possible_paths:
            if path:
                print(f"  - {path} {'(EXISTS)' if os.path.exists(path) else '(NOT FOUND)'}")
        print("Skipping experimental data comparison.")
    else:
        print(f"Found experimental data file: {experimental_file}")
    
    print(f"Running test simulation with first sample...")
    print(f"Sample parameters:")
    for i, param in enumerate(param_defs):
        print(f"  {param['name']}: {sample[0, i]:.3e}")
    
    # Run a single simulation
    try:
        test_result = run_single_simulation(
            sample=sample[0],
            param_defs=param_defs,
            param_mapping=param_mapping,
            simulation_index=0,
            config_path=config_path,
            suppress_print=True
        )
        
        if 'error' in test_result:
            print(f"ERROR: Test simulation failed: {test_result['error']}")
            return False
        
        if 'watcher_data' not in test_result or 'oside' not in test_result['watcher_data']:
            print("ERROR: Test simulation did not return valid oside data")
            return False
        
        # Extract oside data
        oside_data = test_result['watcher_data']['oside']
        sim_time = np.array(oside_data['time'])
        sim_oside_normalized = np.array(oside_data['normalized'])
        
        print(f"Test simulation completed successfully!")
        print(f"Simulation time range: {sim_time[0]*1e6:.3f} to {sim_time[-1]*1e6:.3f} μs")
        print(f"Oside normalized range: {sim_oside_normalized.min():.6f} to {sim_oside_normalized.max():.6f}")
        
        # Load experimental data for comparison if available
        exp_time = None
        exp_oside = None
        
        if experimental_file and os.path.exists(experimental_file):
            try:
                print(f"Loading experimental data from: {experimental_file}")
                exp_df = pd.read_csv(experimental_file)
                print(f"Experimental data shape: {exp_df.shape}")
                print(f"Experimental data columns: {list(exp_df.columns)}")
                
                # Check for oside and temp columns
                if 'oside' in exp_df.columns and 'temp' in exp_df.columns:
                    # Convert to numeric and sort
                    exp_df['time'] = pd.to_numeric(exp_df['time'], errors='coerce')
                    exp_df['oside'] = pd.to_numeric(exp_df['oside'], errors='coerce')
                    exp_df['temp'] = pd.to_numeric(exp_df['temp'], errors='coerce')
                    exp_df = exp_df.dropna(subset=['time', 'oside', 'temp']).sort_values('time')
                    
                    print(f"After cleaning - data shape: {exp_df.shape}")
                    print(f"Time range: {exp_df['time'].min()*1e6:.3f} to {exp_df['time'].max()*1e6:.3f} μs")
                    print(f"Oside range: {exp_df['oside'].min():.2f} to {exp_df['oside'].max():.2f} K")
                    print(f"Temp range: {exp_df['temp'].min():.2f} to {exp_df['temp'].max():.2f} K")
                    
                    exp_time = exp_df['time'].values
                    exp_oside_raw = exp_df['oside'].values
                    exp_temp_raw = exp_df['temp'].values
                    
                    # Calculate pside excursion (max temp - min temp)
                    pside_excursion = exp_temp_raw.max() - exp_temp_raw.min()
                    print(f"Pside excursion: {pside_excursion:.2f} K")
                    
                    # Normalize experimental oside data using pside excursion
                    exp_oside = (exp_oside_raw - exp_oside_raw.min()) / pside_excursion
                    
                    print(f"Experimental data loaded successfully!")
                    print(f"Experimental time range: {exp_time[0]*1e6:.3f} to {exp_time[-1]*1e6:.3f} μs")
                    print(f"Experimental oside normalized range: {exp_oside.min():.6f} to {exp_oside.max():.6f}")
                else:
                    missing_cols = []
                    if 'oside' not in exp_df.columns:
                        missing_cols.append('oside')
                    if 'temp' not in exp_df.columns:
                        missing_cols.append('temp')
                    print(f"WARNING: Missing required columns: {missing_cols}. Available columns: {list(exp_df.columns)}")
                    print("First few rows of experimental data:")
                    print(exp_df.head())
                    
            except Exception as e:
                print(f"WARNING: Failed to load experimental data: {e}")
                import traceback
                traceback.print_exc()
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Test Simulation Verification\nFirst Sample vs Experimental Data', fontsize=14)
        
        # Plot 1: Raw simulation data
        axes[0].plot(sim_time * 1e6, sim_oside_normalized, 'b-', linewidth=2, label='Simulation (normalized)')
        axes[0].set_xlabel('Time (μs)')
        axes[0].set_ylabel('Normalized Temperature')
        axes[0].set_title('Simulation Oside Curve')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Comparison with experimental data
        axes[1].plot(sim_time * 1e6, sim_oside_normalized, 'b-', linewidth=2, label='Simulation (normalized)')
        
        if exp_time is not None and exp_oside is not None:
            axes[1].plot(exp_time * 1e6, exp_oside, 'r--', linewidth=2, label='Experimental (normalized)')
            
            # Calculate overlap statistics
            # Interpolate simulation data to experimental time points for comparison
            from scipy.interpolate import interp1d
            try:
                sim_interp = interp1d(sim_time, sim_oside_normalized, bounds_error=False, fill_value='extrapolate')
                sim_at_exp_times = sim_interp(exp_time)
                
                # Calculate statistics on overlapping time range
                valid_mask = ~np.isnan(sim_at_exp_times)
                if np.sum(valid_mask) > 0:
                    exp_valid = exp_oside[valid_mask]
                    sim_valid = sim_at_exp_times[valid_mask]
                    
                    rmse = np.sqrt(np.mean((sim_valid - exp_valid)**2))
                    max_diff = np.max(np.abs(sim_valid - exp_valid))
                    mean_diff = np.mean(sim_valid - exp_valid)
                    
                    stats_text = f'Comparison Statistics:\nRMSE: {rmse:.6f}\nMax |Diff|: {max_diff:.6f}\nMean Diff: {mean_diff:.6f}'
                    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
                    
                    print(f"\nComparison Statistics:")
                    print(f"  RMSE: {rmse:.6f}")
                    print(f"  Max |Diff|: {max_diff:.6f}")
                    print(f"  Mean Diff: {mean_diff:.6f}")
            except Exception as e:
                print(f"WARNING: Could not calculate comparison statistics: {e}")
        
        axes[1].set_xlabel('Time (μs)')
        axes[1].set_ylabel('Normalized Temperature')
        axes[1].set_title('Simulation vs Experimental Comparison')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_dir, 'test_simulation_verification.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Test simulation verification plot saved to: {plot_file}")
        
        # Show the plot
        plt.show()
        
        # Ask user if they want to proceed
        print("\n" + "="*60)
        print("TEST SIMULATION VERIFICATION COMPLETE")
        print("="*60)
        print("Please review the plot above to ensure:")
        print("1. The simulation oside curve looks reasonable")
        print("2. The timing and shape are appropriate")
        print("3. The comparison with experimental data (if available) is reasonable")
        
        response = input("\nDo you want to proceed with the full batch of simulations? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            print("Proceeding with full batch simulations...")
            return True
        else:
            print("Aborting batch simulations. Please check your configuration and try again.")
            return False
            
    except Exception as e:
        print(f"ERROR running test simulation: {e}")
        return False


def main():
    """Main function to generate training data with configurable input files."""
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Generate training data for surrogate models using Latin Hypercube Sampling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default Geballe configuration
  python generate_training_data.py
  
  # Use Edmund configuration
  python generate_training_data.py --distributions configs/distributions_edmund.yaml --config configs/edmund.yaml
  
  # Use custom output directory (overrides config file paths)
  python generate_training_data.py --distributions configs/distributions_edmund.yaml --config configs/edmund.yaml --output-dir outputs/my_custom_dir
        """
    )
    
    parser.add_argument('--distributions', type=str, default='configs/distributions.yaml',
                       help='Path to the distributions YAML file (default: configs/distributions.yaml)')
    parser.add_argument('--config', type=str, default='configs/config_5_materials.yaml',
                       help='Path to the simulation configuration YAML file (default: configs/config_5_materials.yaml)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results (overrides config file setting)')
    parser.add_argument('--rebuild-fpca-from', type=str, default=None,
                       help='Path to a uq_batch_results.npz file to rebuild the FPCA model from, skipping simulations.')
    parser.add_argument('--skip-test', action='store_true',
                       help='Skip the test simulation verification step')
    
    args = parser.parse_args()
    
    print(f"Using distributions file: {args.distributions}")
    print(f"Using simulation config file: {args.config}")
    print(f"Output directory: {args.output_dir}")
    
    # Verify the experimental data file being used
    if not args.rebuild_fpca_from:  # Only verify if we're actually running simulations
        success = plot_experimental_data_verification(args.config, args.output_dir)
        if not success:
            print("WARNING: Failed to verify experimental data. Proceeding anyway...")
    
    # Load configuration from specified YAML file
    param_defs, PARAM_MAPPING, sampling_config, output_config = load_all_from_config(args.distributions)
    
    # Create UQpy distributions for each parameter
    distributions = create_uqpy_distributions(param_defs)
    
    # Get number of samples from config (no command-line override)
    n_samples = sampling_config.get('n_samples', 200)
    print(f"Number of samples from config: {n_samples}")
    
    print(f"\nGenerating {n_samples} samples using Latin Hypercube Sampling...")
    
    # Latin Hypercube Sampling
    lhs = LatinHypercubeSampling(distributions=distributions, nsamples=n_samples)
    samples = lhs.samples
    
    # Note: UQpy Lognormal distribution already returns values in the correct space
    # No exponentiation needed for lognormal parameters
    
    print("LHS samples (physical space):")
    print(samples)
    
    # After generating LHS samples, augment with extreme boundary points
    # ------------------------------------------------------------------
    def _get_param_bounds(p):
        """Return (low, high) bounds in physical space for a parameter definition."""
        if p["type"] == "uniform":
            return p["low"], p["high"]
        elif p["type"] == "normal":
            c, s = p["center"], p["sigma"]
            return c - 3 * s, c + 3 * s  # ±3σ captures 99.7 % of the mass
        elif p["type"] == "lognormal":
            mu = np.log(p["center"])
            sigma = p["sigma_log"]
            return np.exp(mu - 3 * sigma), np.exp(mu + 3 * sigma)
        else:
            raise ValueError(f"Unknown parameter type: {p['type']}")

    def _augment_with_boundaries(base_samples, p_defs, n_per_face: int = 3, random_state: int = 42):
        """Augment base_samples with extreme points.

        For each parameter we create `n_per_face` samples at the low bound and
        `n_per_face` at the high bound.  For diversity the remaining coordinates
        are copied from random LHS samples rather than the global mean.
        """
        rng = np.random.default_rng(random_state)
        aug = []
        for idx, p in enumerate(p_defs):
            low, high = _get_param_bounds(p)
            for _ in range(n_per_face):
                template = base_samples[rng.integers(len(base_samples))].copy()
                low_samp = template.copy();  low_samp[idx] = low
                high_samp = template.copy(); high_samp[idx] = high
                aug.extend([low_samp, high_samp])
        return np.vstack([base_samples, np.array(aug)])

    samples = _augment_with_boundaries(samples, param_defs, n_per_face=3, random_state=42)
    print(f"Augmented samples with boundary points -> new total: {len(samples)}")
    
    # Determine output paths - use distributions file paths, but allow command-line override
    if args.output_dir != 'outputs':  # If user specified a custom output directory
        # Override all paths to use the custom directory
        samples_file = f"{args.output_dir}/initial_train_set.csv"
        results_file = f"{args.output_dir}/uq_batch_results.npz"
        fpca_model_file = f"{args.output_dir}/fpca_model.npz"
        fpca_training_file = f"{args.output_dir}/training_data_fpca.npz"
        distribution_plot_file = f"{args.output_dir}/parameter_distributions.png"
        correlation_plot_file = f"{args.output_dir}/parameter_correlations.png"
        print(f"Using custom output directory: {args.output_dir}")
    else:
        # Use paths from distributions file
        samples_file = output_config.get('samples_file', 'outputs/initial_train_set.csv')
        results_file = output_config.get('results_file', 'outputs/uq_batch_results.npz')
        fpca_model_file = output_config.get('fpca_model_file', 'outputs/fpca_model.npz')
        fpca_training_file = output_config.get('fpca_training_file', 'outputs/training_data_fpca.npz')
        distribution_plot_file = output_config.get('distribution_plot_file', 'outputs/parameter_distributions.png')
        correlation_plot_file = output_config.get('correlation_plot_file', 'outputs/parameter_correlations.png')
        print(f"Using output paths from distributions file")
    
    # Ensure output directory exists
    import os
    output_dir = os.path.dirname(samples_file)
    os.makedirs(output_dir, exist_ok=True)
    
    header = ",".join([p["name"] for p in param_defs])
    np.savetxt(samples_file, samples, delimiter=",", header=header, comments='')
    print(f"Sample parameters saved to: {samples_file}")
    
    # Create parameter distribution plots
    print("\n" + "="*60)
    print("CREATING PARAMETER DISTRIBUTION PLOTS")
    print("="*60)
    plot_parameter_distributions(samples, param_defs, output_dir=os.path.dirname(distribution_plot_file))
    
    # Create parameter correlation plot
    print("\n" + "="*60)
    print("CREATING PARAMETER CORRELATION PLOT")
    print("="*60)
    plot_parameter_correlations(samples, param_defs, output_dir=os.path.dirname(correlation_plot_file))
    
    # This block will be skipped if --rebuild-fpca-from is used
    if not args.rebuild_fpca_from:
        # Run test simulation verification (unless skipped)
        if not args.skip_test:
            print("\n" + "="*60)
            print("STEP 1: TEST SIMULATION VERIFICATION")
            print("="*60)
            test_success = plot_test_simulation_verification(
                samples, param_defs, PARAM_MAPPING, args.config, args.output_dir
            )
            if not test_success:
                print("Test simulation verification failed. Exiting.")
                return
        
        # Run batch simulations
        print("\n" + "="*60)
        print("STEP 2: RUNNING BATCH SIMULATIONS")
        print("="*60)
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
        
        # Pass the simulation config file to the batch simulation function
        results = run_batch_simulations(
            samples, param_defs, PARAM_MAPPING, 
            suppress_print=True, 
            progress_callback=progress_callback,
            config_path=args.config  # Pass the simulation config file
        )
        
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
            save_batch_results(results, param_defs, output_file=results_file)
        else:
            print("No successful simulations to save!")
            return # Exit if no results
    
    # If rebuilding, override the results_file path to the user-provided one
    if args.rebuild_fpca_from:
        print(f"\n--rebuild-fpca-from specified. Loading results from: {args.rebuild_fpca_from}")
        print("Skipping batch simulations and using existing data.")
        results_file = args.rebuild_fpca_from
        
    # Step 3: Build FPCA model and recast training data
    print("\n" + "="*60)
    print("STEP 3: BUILDING FPCA MODEL AND RECASTING DATA")
    print("="*60)
    
    # Load and verify the saved data
    print("\nVerifying saved data...")
    batch_data = load_batch_results(results_file)
    
    print(f"Loaded data keys: {list(batch_data.keys())}")
    print(f"Oside curves shape: {batch_data['oside_curves'].shape}")
    print(f"Parameters shape: {batch_data['parameters'].shape}")
    print(f"Parameter names: {batch_data['parameter_names']}")
    
    # Count valid curves
    valid_curves = np.sum(~np.isnan(batch_data['oside_curves']).any(axis=1))
    print(f"Valid oside curves: {valid_curves}")
    
    # Basic statistics
    valid_data = batch_data['oside_curves'][~np.isnan(batch_data['oside_curves']).any(axis=1)]
    if len(valid_data) > 0:
        print(f"Oside curve statistics:")
        print(f"  Mean max value: {np.mean(np.max(valid_data, axis=1)):.3f}")
        print(f"  Mean min value: {np.mean(np.min(valid_data, axis=1)):.3f}")
        print(f"  Mean range: {np.mean(np.max(valid_data, axis=1) - np.min(valid_data, axis=1)):.3f}")
    
    # Build FPCA model
    print("\nBuilding FPCA model...")
    fpca_model = build_fpca_model(batch_data, results_file)
    
    # Save FPCA model
    print("\nSaving FPCA model...")
    save_fpca_model(fpca_model, fpca_model_file)
    
    # Recast training data to FPCA space
    print("\nRecasting training data to FPCA space...")
    recast_training_data_to_fpca(batch_data, fpca_model, fpca_training_file)
    
    # Print summary
    print("\n" + "="*60)
    print("FPCA ANALYSIS SUMMARY")
    print("="*60)
    print(f"Number of components: {fpca_model['n_components']}")
    print(f"Explained variance: {fpca_model['explained_variance']:.4f}")
    print(f"Training data shape: {batch_data['parameters'].shape}")
    print(f"FPCA scores shape: {fpca_model['training_scores'].shape}")
    
    # Print FPCA score statistics
    print("\nFPCA Score Statistics:")
    for i in range(fpca_model['n_components']):
        scores = fpca_model['training_scores'][:, i]
        print(f"PC{i+1}: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
    
    print(f"\nTraining data generation completed successfully!")
    print(f"Results saved to: {results_file}")
    print(f"FPCA model saved to: {fpca_model_file}")
    print(f"FPCA training data saved to: {fpca_training_file}")


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


if __name__ == "__main__":
    main()
