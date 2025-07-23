#!/usr/bin/env python3
"""
Script to compare surrogate model predictions with full FEM simulations.
Uses parameter values from Edmund config to test surrogate accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import argparse
from analysis.config_utils import get_param_defs_from_config, get_param_mapping_from_config
from analysis.uq_wrapper import run_single_simulation
from train_surrogate_models import FullSurrogateModel
import time

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare surrogate vs FEM predictions')
    parser.add_argument('--surrogate-path', 
                       default="outputs/edmund1/full_surrogate_model_int_ins_match.pkl",
                       help='Path to surrogate model file (default: outputs/edmund1/full_surrogate_model_int_ins_match.pkl)')
    parser.add_argument('--config-path',
                       default="configs/edmund.yaml", 
                       help='Path to simulation config file (default: configs/edmund.yaml)')
    parser.add_argument('--experimental-data',
                       default="data/experimental/edmund_71Gpa_run1.csv",
                       help='Path to experimental data file (default: data/experimental/edmund_71Gpa_run1.csv)')
    return parser.parse_args()

def load_edmund_config_params(config_path="configs/edmund.yaml"):
    """
    Load parameter values directly from Edmund config file.
    Returns the parameter values in the order expected by the surrogate.
    """
    # Load parameter definitions and mapping for parameter order
    param_defs = get_param_defs_from_config("configs/distributions_edmund.yaml")
    param_mapping = get_param_mapping_from_config("configs/distributions_edmund.yaml")
    
    # Extract parameter names in order
    param_names = [param_def['name'] for param_def in param_defs]
    print(f"Parameter names: {param_names}")
    
    # Load Edmund config file directly
    with open(config_path, 'r') as f:
        edmund_config = yaml.safe_load(f)
    
    # Extract parameter values from Edmund config
    param_values = []
    for param_def in param_defs:
        param_name = param_def['name']
        
        if param_name == 'd_sample':
            value = float(edmund_config['mats']['sample']['z'])
        elif param_name == 'rho_cv_sample':
            value = float(edmund_config['mats']['sample']['rho_cv'])
        elif param_name == 'rho_cv_ins':
            value = float(edmund_config['mats']['p_ins']['rho_cv'])  # Use p_ins as reference
        elif param_name == 'd_ins_pside':
            value = float(edmund_config['mats']['p_ins']['z'])
        elif param_name == 'd_ins_oside':
            value = float(edmund_config['mats']['o_ins']['z'])
        elif param_name == 'fwhm':
            value = float(edmund_config['heating']['fwhm'])
        elif param_name == 'k_sample':
            value = float(edmund_config['mats']['sample']['k'])
        elif param_name == 'k_ins':
            value = float(edmund_config['mats']['p_ins']['k'])  # Use p_ins as reference
        elif param_name == 'k_int':
            value = float(edmund_config['mats']['p_int']['k'])  # Use p_int as reference
        else:
            print(f"WARNING: Unknown parameter {param_name}, using midpoint of distribution")
            # Fallback to distribution midpoint
            if param_def['type'] == 'normal':
                value = param_def['center']
            elif param_def['type'] == 'uniform':
                value = (param_def['low'] + param_def['high']) / 2
            elif param_def['type'] == 'lognormal':
                value = param_def['center']
        
        param_values.append(value)
    
    param_values = np.array(param_values)
    
    print(f"Parameter values from Edmund config:")
    for name, value in zip(param_names, param_values):
        print(f"  {name}: {value:.3e}")
    
    return param_values, param_names, param_defs, param_mapping

def run_fem_simulation(param_values, param_names, param_defs, param_mapping, config_path="configs/edmund.yaml"):
    """
    Run a single FEM simulation with the given parameters.
    """
    print("\nRunning FEM simulation...")
    start_time = time.time()
    
    try:
        result = run_single_simulation(
            sample=param_values,
            param_defs=param_defs,
            param_mapping=param_mapping,
            simulation_index=0,
            config_path=config_path
        )
        
        fem_time = time.time() - start_time
        print(f"FEM simulation completed in {fem_time:.2f} seconds")
        
        if 'watcher_data' in result and 'oside' in result['watcher_data']:
            fem_curve = result['watcher_data']['oside']['normalized']
            fem_time_array = result['watcher_data']['oside']['time']
            return fem_curve, fem_time_array, result
        else:
            print("ERROR: No watcher data found in FEM result")
            return None, None, result
            
    except Exception as e:
        print(f"ERROR in FEM simulation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def run_surrogate_prediction(param_values, param_names, surrogate_path):
    """
    Run surrogate model prediction with the given parameters.
    """
    print("\nRunning surrogate prediction...")
    start_time = time.time()
    
    try:
        # Load surrogate model
        surrogate = FullSurrogateModel.load_model(surrogate_path)
        
        # Check if parameter names match
        surr_param_names = surrogate.parameter_names
        print(f"Surrogate parameter names: {surr_param_names}")
        print(f"Input parameter names: {param_names}")
        
        if len(surr_param_names) != len(param_names):
            print(f"WARNING: Parameter count mismatch! Surrogate: {len(surr_param_names)}, Input: {len(param_names)}")
        
        # Reshape parameters for surrogate (n_samples, n_params)
        X = param_values.reshape(1, -1)
        
        # Predict temperature curves along with predictive uncertainty
        curves, fpca_coeffs, fpca_uncertainties, curve_uncertainties = surrogate.predict_temperature_curves(X)
        
        surrogate_time = time.time() - start_time
        print(f"Surrogate prediction completed in {surrogate_time:.4f} seconds")

        # Flatten uncertainty array just in case
        curve_uncert = np.asarray(curve_uncertainties).flatten()

        # Print basic uncertainty statistics
        print("GP predictive 1σ (surrogate) statistics:")
        print(f"  min:  {curve_uncert.min():.3e}")
        print(f"  mean: {curve_uncert.mean():.3e}")
        print(f"  max:  {curve_uncert.max():.3e}")
        
        # Get the predicted curve
        surrogate_curve = curves[0]
        curve_uncert = curve_uncertainties[0]
        
        # The surrogate model now has the correct time grid
        surrogate_time_array = surrogate.time_grid
        
        return surrogate_curve, curve_uncert, surrogate_time_array, surrogate
        
    except Exception as e:
        print(f"ERROR in surrogate prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def load_experimental_data(data_path="data/experimental/edmund_71Gpa_run1.csv"):
    """
    Load experimental data for comparison.
    """
    try:
        data = pd.read_csv(data_path)
        oside_data = data['oside'].values
        exp_time = data['time'].values
        
        # Normalize experimental data the same way as in MCMC
        y_obs = (oside_data - oside_data[0]) / (data['temp'].max() - data['temp'].min())
        
        return y_obs, exp_time
    except Exception as e:
        print(f"ERROR loading experimental data: {e}")
        return None, None

def plot_comparison(fem_curve, fem_time, surrogate_curve, curve_uncert, surrogate_time,
                   exp_curve, exp_time, param_names, param_values):
    """
    Create comparison plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Direct comparison of curves
    ax1 = axes[0, 0]
    if fem_curve is not None:
        ax1.plot(fem_time * 1e6, fem_curve, 'b-', linewidth=2, label='FEM')
    if surrogate_curve is not None:
        ax1.plot(surrogate_time * 1e6, surrogate_curve, 'r--', linewidth=2, label='Surrogate')
        if curve_uncert is not None:
            ax1.fill_between(surrogate_time * 1e6,
                             surrogate_curve - curve_uncert,
                             surrogate_curve + curve_uncert,
                             color='red', alpha=0.2, label='Surrogate ±1σ', zorder=1)
    if exp_curve is not None:
        ax1.plot(exp_time * 1e6, exp_curve, 'k-', linewidth=1, alpha=0.7, label='Experimental')
    
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Normalized Temperature')
    ax1.set_title('Temperature Curves Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference between FEM and Surrogate
    ax2 = axes[0, 1]
    if fem_curve is not None and surrogate_curve is not None:
        # Interpolate surrogate to FEM time grid for comparison
        from scipy.interpolate import interp1d
        interp_func = interp1d(surrogate_time, surrogate_curve, kind='linear', 
                              bounds_error=False, fill_value=(surrogate_curve[0], surrogate_curve[-1]))
        surrogate_interp = interp_func(fem_time)
        
        difference = fem_curve - surrogate_interp
        ax2.plot(fem_time * 1e6, difference, 'g-', linewidth=2)
        ax2.set_xlabel('Time (μs)')
        ax2.set_ylabel('FEM - Surrogate')
        ax2.set_title('Difference (FEM - Surrogate)')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        rmse = np.sqrt(np.mean(difference**2))
        max_diff = np.max(np.abs(difference))
        ax2.text(0.05, 0.95, f'RMSE: {rmse:.6f}\nMax |Diff|: {max_diff:.6f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3: Parameter values
    ax3 = axes[1, 0]
    param_names_short = [name.replace('_', '\n') for name in param_names]
    bars = ax3.bar(range(len(param_values)), param_values)
    ax3.set_xticks(range(len(param_values)))
    ax3.set_xticklabels(param_names_short, rotation=45, ha='right')
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('Parameter Values Used')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, param_values)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2e}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Time domain comparison (if experimental data available)
    ax4 = axes[1, 1]
    if exp_curve is not None:
        # Interpolate both models to experimental time grid
        from scipy.interpolate import interp1d
        
        if fem_curve is not None:
            fem_interp_func = interp1d(fem_time, fem_curve, kind='linear', 
                                     bounds_error=False, fill_value=(fem_curve[0], fem_curve[-1]))
            fem_interp = fem_interp_func(exp_time)
            ax4.plot(exp_time * 1e6, fem_interp, 'b-', linewidth=2, label='FEM')
        
        if surrogate_curve is not None:
            surr_interp_func = interp1d(surrogate_time, surrogate_curve, kind='linear', 
                                      bounds_error=False, fill_value=(surrogate_curve[0], surrogate_curve[-1]))
            surr_interp = surr_interp_func(exp_time)
            ax4.plot(exp_time * 1e6, surr_interp, 'r--', linewidth=2, label='Surrogate')

            # Interpolate uncertainty to experimental grid and plot band
            if curve_uncert is not None:
                uncert_interp_func = interp1d(surrogate_time, curve_uncert, kind='linear',
                                              bounds_error=False, fill_value=(curve_uncert[0], curve_uncert[-1]))
                uncert_interp = uncert_interp_func(exp_time)
                ax4.fill_between(exp_time * 1e6,
                                 surr_interp - uncert_interp,
                                 surr_interp + uncert_interp,
                                 color='red', alpha=0.2, zorder=1)
        
        ax4.plot(exp_time * 1e6, exp_curve, 'k-', linewidth=1, alpha=0.7, label='Experimental')
        ax4.set_xlabel('Time (μs)')
        ax4.set_ylabel('Normalized Temperature')
        ax4.set_title('Comparison on Experimental Time Grid')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Calculate RMSE vs experimental
        if fem_curve is not None:
            fem_rmse = np.sqrt(np.mean((fem_interp - exp_curve)**2))
            ax4.text(0.05, 0.95, f'FEM RMSE vs Exp: {fem_rmse:.6f}', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        if surrogate_curve is not None:
            surr_rmse = np.sqrt(np.mean((surr_interp - exp_curve)**2))
            ax4.text(0.05, 0.85, f'Surrogate RMSE vs Exp: {surr_rmse:.6f}', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("surrogate_vs_fem_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """
    Main function to run the comparison.
    """
    args = parse_arguments()
    
    print("=" * 60)
    print("SURROGATE vs FEM COMPARISON")
    print("=" * 60)
    print(f"Using surrogate model: {args.surrogate_path}")
    print(f"Using config file: {args.config_path}")
    print(f"Using experimental data: {args.experimental_data}")
    
    # Load parameter values from Edmund config
    param_values, param_names, param_defs, param_mapping = load_edmund_config_params(args.config_path)
    
    # Run FEM simulation
    fem_curve, fem_time, fem_result = run_fem_simulation(param_values, param_names, param_defs, param_mapping, args.config_path)
    
    # Run surrogate prediction
    surrogate_curve, curve_uncert, surrogate_time, surrogate = run_surrogate_prediction(param_values, param_names, args.surrogate_path)
    
    # Load experimental data
    exp_curve, exp_time = load_experimental_data(args.experimental_data)
    
    # Create comparison plots
    if fem_curve is not None or surrogate_curve is not None:
        fig = plot_comparison(fem_curve, fem_time, surrogate_curve, curve_uncert, surrogate_time,
                             exp_curve, exp_time, param_names, param_values)
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        
        if fem_curve is not None and surrogate_curve is not None:
            # Interpolate for comparison
            from scipy.interpolate import interp1d
            interp_func = interp1d(surrogate_time, surrogate_curve, kind='linear', 
                                  bounds_error=False, fill_value=(surrogate_curve[0], surrogate_curve[-1]))
            surrogate_interp = interp_func(fem_time)
            
            difference = fem_curve - surrogate_interp
            rmse = np.sqrt(np.mean(difference**2))
            max_diff = np.max(np.abs(difference))
            mean_diff = np.mean(difference)
            std_diff = np.std(difference)
            
            print(f"FEM vs Surrogate Statistics:")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  Max absolute difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            print(f"  Std difference: {std_diff:.6f}")
        
        if exp_curve is not None:
            print(f"\nExperimental Data Statistics:")
            print(f"  Time range: {exp_time[0]:.3e} to {exp_time[-1]:.3e} seconds")
            print(f"  Data points: {len(exp_curve)}")
            print(f"  Temperature range: {exp_curve.min():.6f} to {exp_curve.max():.6f}")
        
        print(f"\nPlot saved as: surrogate_vs_fem_comparison.png")
        
    else:
        print("ERROR: Neither FEM nor surrogate produced valid results!")
    
    print("\nComparison complete!")

if __name__ == "__main__":
    main() 