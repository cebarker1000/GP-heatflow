#!/usr/bin/env python3
"""
Plot surrogate model sensitivity to thermal conductivity parameters.
Creates three subplots showing how the surrogate output varies with each k parameter
while keeping the other two fixed at their midpoint values.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from train_surrogate_models import FullSurrogateModel
from analysis.config_utils import get_fixed_params_from_config, get_param_defs_from_config
import warnings

warnings.filterwarnings('ignore')

def load_experimental_data():
    """Load and preprocess experimental data."""
    data = pd.read_csv("data/experimental/geballe_heat_data.csv")
    oside_data = data['oside'].values
    y_obs = (oside_data - oside_data[0]) / (data['temp'].max() - data['temp'].iloc[0])
    exp_time = data['time'].values
    return y_obs, exp_time

def interpolate_to_surrogate_grid(exp_data, exp_time):
    """Interpolate experimental data to surrogate time grid."""
    sim_t_final = 7.5e-6  # seconds
    sim_num_steps = 50
    surrogate_time_grid = np.linspace(0, sim_t_final, sim_num_steps)
    interp_func = interp1d(exp_time, exp_data, kind='linear', 
                           bounds_error=False, fill_value=(exp_data[0], exp_data[-1]))
    interpolated_data = interp_func(surrogate_time_grid)
    return interpolated_data, surrogate_time_grid

def create_sensitivity_plots():
    """Create sensitivity plots for each thermal conductivity parameter."""
    
    print("Loading surrogate model...")
    surrogate = FullSurrogateModel.load_model("outputs/full_surrogate_model.pkl")
    
    print("Loading experimental data...")
    y_obs, exp_time = load_experimental_data()
    y_obs_interp, time_grid = interpolate_to_surrogate_grid(y_obs, exp_time)
    
    # Get fixed parameters (nuisance parameters)
    PARAMS_FIXED = get_fixed_params_from_config()
    
    # Get parameter definitions from config file
    param_defs = get_param_defs_from_config()
    
    # Extract k parameter ranges from config
    k_ranges = {}
    for param_def in param_defs:
        if param_def['name'].startswith('k_'):
            if param_def['type'] == 'uniform':
                k_ranges[param_def['name']] = (param_def['low'], param_def['high'])
            else:
                print(f"Warning: {param_def['name']} is not uniform, skipping sensitivity analysis")
    
    print(f"Parameter ranges from config: {k_ranges}")
    
    # Calculate midpoint values for k parameters
    k_midpoints = {'k_sample': 3.3, 'k_ins': 10, 'k_coupler': 350}
    '''
    for param_name, (low, high) in k_ranges.items():
        k_midpoints[param_name] = (low + high) / 2
    '''
    
    print(f"Midpoint values: {k_midpoints}")
    
    # Create figure with subplots for each k parameter
    n_k_params = len(k_ranges)
    fig, axes = plt.subplots(n_k_params, 1, figsize=(12, 4*n_k_params))
    if n_k_params == 1:
        axes = [axes]
    
    for i, (param_name, (k_min, k_max)) in enumerate(k_ranges.items()):
        print(f"Generating sensitivity plot for {param_name}...")
        
        # Create parameter values to test
        k_values = np.linspace(k_min, k_max, 10)
        
        # Create colormap for this parameter (viridis shows clear progression)
        colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))
        
        # Plot experimental data
        axes[i].plot(time_grid * 1e6, y_obs_interp, 'k-', linewidth=2, 
                    label='Experimental', alpha=0.8)
        
        # Test each parameter value
        for j, k_val in enumerate(k_values):
            # Create parameter vector with current k value and midpoints for others
            params = np.zeros(len(k_ranges))
            for k_idx, (k_param_name, _) in enumerate(k_ranges.items()):
                if k_param_name == param_name:
                    params[k_idx] = k_val
                else:
                    params[k_idx] = k_midpoints[k_param_name]
            
            # Build full parameter vector
            params_full = np.hstack([PARAMS_FIXED, params])
            
            # Get surrogate prediction
            try:
                y_pred, _, _, _ = surrogate.predict_temperature_curves(params_full.reshape(1, -1))
                y_pred = y_pred[0]  # Remove batch dimension
                
                # Plot prediction with color from colormap
                axes[i].plot(time_grid * 1e6, y_pred, color=colors[j], 
                           alpha=0.8, linewidth=1.5, 
                           label=f'{param_name}={k_val:.1f}' if j in [0, 4, 9] else "")
                
            except Exception as e:
                print(f"Error predicting for {param_name}={k_val}: {e}")
        
        # Customize subplot
        axes[i].set_xlabel('Time (μs)')
        axes[i].set_ylabel('Normalized Temperature')
        axes[i].set_title(f'Surrogate Sensitivity: {param_name}')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('surrogate_sensitivity_plots.png', dpi=300, bbox_inches='tight')
    print("Sensitivity plots saved to surrogate_sensitivity_plots.png")
    plt.show()
    
    # Create summary statistics
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*60)
    
    for param_name, (k_min, k_max) in k_ranges.items():
        print(f"\n{param_name} ({k_min} to {k_max}):")
        
        # Test extreme values
        for k_val in [k_min, (k_min + k_max)/2, k_max]:
            params = np.zeros(len(k_ranges))
            for k_idx, (k_param_name, _) in enumerate(k_ranges.items()):
                if k_param_name == param_name:
                    params[k_idx] = k_val
                else:
                    params[k_idx] = k_midpoints[k_param_name]
            
            params_full = np.hstack([PARAMS_FIXED, params])
            
            try:
                y_pred, _, _, _ = surrogate.predict_temperature_curves(params_full.reshape(1, -1))
                y_pred = y_pred[0]
                
                # Calculate RMSE compared to experimental data
                rmse = np.sqrt(np.mean((y_pred - y_obs_interp)**2))
                max_temp = np.max(y_pred)
                min_temp = np.min(y_pred)
                temp_range = max_temp - min_temp
                
                print(f"  {param_name}={k_val:.1f}: RMSE={rmse:.4f}, "
                      f"Range=[{min_temp:.4f}, {max_temp:.4f}], ΔT={temp_range:.4f}")
                
            except Exception as e:
                print(f"  {param_name}={k_val:.1f}: ERROR - {e}")

def main():
    """Main function to create sensitivity plots."""
    print("Creating surrogate sensitivity plots...")
    create_sensitivity_plots()
    print("Done!")

if __name__ == "__main__":
    main() 