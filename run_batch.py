"""
Batch processing script for heat flow simulations.
Supports running multiple simulations with different parameter configurations.
"""

import os
import yaml
import numpy as np
from typing import List, Dict, Any, Optional
from simulation_engine import OptimizedSimulationEngine
import matplotlib.pyplot as plt
import pandas as pd
import analysis_utils as au


def run_batch_simulations(configs: List[Dict[str, Any]], 
                         suppress_print: bool = True,
                         progress_callback=None) -> List[Dict[str, Any]]:
    """
    Run a batch of simulations using minimal run mode.
    
    Parameters:
    -----------
    configs : List[Dict[str, Any]]
        List of configuration dictionaries for each simulation
    suppress_print : bool, optional
        Whether to suppress print output during simulations
    progress_callback : callable, optional
        Callback function for progress reporting (called with current index, total)
    
    Returns:
    --------
    List[Dict[str, Any]]
        List of results for each simulation
    """
    results = []
    
    for i, config in enumerate(configs):
        if progress_callback:
            progress_callback(i, len(configs))
        
        try:
            # Create temporary mesh and output folders (not used but required by engine)
            mesh_folder = f"temp_mesh_{i}"
            output_folder = f"temp_output_{i}"
            
            # Create simulation engine
            engine = OptimizedSimulationEngine(config, mesh_folder, output_folder)
            
            # Run minimal simulation
            result = engine.run_minimal(suppress_print=suppress_print)
            
            # Add simulation index to result
            result['simulation_index'] = i
            
            results.append(result)
            
        except Exception as e:
            print(f"Simulation {i} failed: {e}")
            results.append({
                'simulation_index': i,
                'error': str(e),
                'config': config
            })
    
    return results


def extract_qois_from_batch_results(results: List[Dict[str, Any]], 
                                   qoi_names: Optional[List[str]] = None) -> Dict[str, List]:
    """
    Extract quantities of interest from batch results.
    
    Parameters:
    -----------
    results : List[Dict[str, Any]]
        Results from batch simulations
    qoi_names : List[str], optional
        Names of QoIs to extract. If None, extracts all available.
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary with QoI names as keys and arrays of values as values
    """
    if not results:
        return {}
    
    # Get QoI names from first successful result
    first_result = next((r for r in results if 'watcher_data' in r), None)
    if not first_result:
        return {}
    
    available_qois = list(first_result['watcher_data'].keys())
    if qoi_names is None:
        qoi_names = available_qois
    
    # Validate QoI names
    invalid_qois = [name for name in qoi_names if name not in available_qois]
    if invalid_qois:
        raise ValueError(f"Invalid QoI names: {invalid_qois}. Available: {available_qois}")
    
    # Extract QoIs
    qois = {}
    for qoi_name in qoi_names:
        qoi_values = []
        for result in results:
            if 'watcher_data' in result and qoi_name in result['watcher_data']:
                # Extract normalized temperature curve
                normalized_curve = result['watcher_data'][qoi_name]['normalized']
                qoi_values.append(normalized_curve)
            else:
                # Handle failed simulations
                qoi_values.append(None)
        
        qois[qoi_name] = qoi_values
    
    return qois


def save_batch_results(results: List[Dict[str, Any]], 
                      output_file: str = "batch_results.npz"):
    """
    Save batch results to a compressed numpy file.
    
    Parameters:
    -----------
    results : List[Dict[str, Any]]
        Results from batch simulations
    output_file : str
        Output file path
    """
    # Extract QoIs
    qois = extract_qois_from_batch_results(results)
    
    # Prepare data for saving
    save_data = {}
    
    # Save QoI data
    for qoi_name, qoi_values in qois.items():
        # Convert list of arrays to 2D array (simulations x time_steps)
        valid_arrays = [arr for arr in qoi_values if arr is not None]
        if valid_arrays:
            # Pad arrays to same length if needed
            max_length = max(arr.shape[0] for arr in valid_arrays)
            padded_arrays = []
            for arr in qoi_values:
                if arr is not None:
                    if arr.shape[0] < max_length:
                        # Pad with last value
                        padded = np.pad(arr, (0, max_length - arr.shape[0]), 
                                      mode='edge')
                    else:
                        padded = arr
                    padded_arrays.append(padded)
                else:
                    # Failed simulation - fill with NaN
                    padded_arrays.append(np.full(max_length, np.nan))
            
            save_data[f'qoi_{qoi_name}'] = np.array(padded_arrays)
    
    # Save timing information
    timing_data = []
    for result in results:
        if 'timing' in result:
            timing_data.append([
                result['timing'].get('total_loop_time', np.nan),
                result['timing'].get('avg_step_time', np.nan),
                result['timing'].get('num_steps', np.nan)
            ])
        else:
            timing_data.append([np.nan, np.nan, np.nan])
    
    save_data['timing'] = np.array(timing_data)
    
    # Save simulation indices
    save_data['simulation_indices'] = np.array([r.get('simulation_index', i) 
                                               for i, r in enumerate(results)])
    
    # Save to file
    np.savez_compressed(output_file, **save_data)
    print(f"Batch results saved to {output_file}")


def load_batch_results(input_file: str = "batch_results.npz") -> Dict[str, np.ndarray]:
    """
    Load batch results from a compressed numpy file.
    
    Parameters:
    -----------
    input_file : str
        Input file path
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary with loaded data
    """
    data = np.load(input_file)
    return {key: data[key] for key in data.keys()}


# Example usage functions for different parameter sampling strategies

def create_parameter_variations(base_config: Dict[str, Any], 
                               parameter_ranges: Dict[str, tuple],
                               num_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Create parameter variations using uniform sampling.
    
    Parameters:
    -----------
    base_config : Dict[str, Any]
        Base configuration dictionary
    parameter_ranges : Dict[str, tuple]
        Dictionary mapping parameter paths to (min, max) ranges
    num_samples : int
        Number of parameter samples to generate
    
    Returns:
    --------
    List[Dict[str, Any]]
        List of modified configuration dictionaries
    """
    configs = []
    
    for i in range(num_samples):
        # Deep copy the base config
        config = yaml.safe_load(yaml.safe_dump(base_config))
        
        # Sample parameters
        for param_path, (min_val, max_val) in parameter_ranges.items():
            # Generate random value
            value = np.random.uniform(min_val, max_val)
            
            # Navigate nested dictionary structure
            keys = param_path.split('.')
            current = config
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value
        
        configs.append(config)
    
    return configs


def example_batch_run():
    """Example of how to use the batch processing functionality."""
    # Load base configuration
    with open('cfgs/geballe_no_diamond.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Define parameter ranges for sampling
    parameter_ranges = {
        'mats.sample.z': (4.0e-6, 4.0e-6),      # thermal conductivity
    }
    
    # Create parameter variations
    configs = create_parameter_variations(base_config, parameter_ranges, num_samples=1)
    
    # Run batch simulations
    results = run_batch_simulations(configs, suppress_print=False)
    
    # Extract QoIs
    qois = extract_qois_from_batch_results(results)
    
    # Save results
    save_batch_results(results, "example_batch_results.npz")
    
    print(f"Completed {len(results)} simulations")
    print(f"Extracted QoIs: {list(qois.keys())}")

    # --- Plotting section using analysis_utils ---
    data = np.load("example_batch_results.npz", allow_pickle=True)

    # Get raw watcher data for pside and oside
    results_first = results[0]
    pside_raw = results_first['watcher_data']['pside']['raw'] if 'pside' in results_first['watcher_data'] else None
    oside_raw = results_first['watcher_data']['oside']['raw'] if 'oside' in results_first['watcher_data'] else None
    sim_time = results_first['watcher_data']['pside']['time'] if pside_raw is not None else None

    if pside_raw is not None and oside_raw is not None and sim_time is not None:
        pside_excursion = np.nanmax(pside_raw) - np.nanmin(pside_raw)
        pside_normed = (pside_raw - pside_raw[0]) / pside_excursion if pside_excursion > 0 else pside_raw * 0
        oside_normed = (oside_raw - oside_raw[0]) / pside_excursion if pside_excursion > 0 else oside_raw * 0
    else:
        pside_normed = pside_raw
        oside_normed = oside_raw

    # Load experimental data if available
    exp_file = base_config.get('heating', {}).get('file', None)
    if exp_file and os.path.exists(exp_file):
        df_exp = pd.read_csv(exp_file)
        exp_pside_normed = (df_exp['temp'] - df_exp['temp'].iloc[0]) / (df_exp['temp'].max() - df_exp['temp'].min())
        if 'oside' in df_exp.columns:
            ic_temp = base_config['heating']['ic_temp']
            oside_initial = df_exp['oside'].iloc[0]
            exp_oside_shifted = df_exp['oside'] - oside_initial + ic_temp
            exp_oside_normed = (exp_oside_shifted - exp_oside_shifted.iloc[0]) / (df_exp['temp'].max() - df_exp['temp'].min())
        else:
            exp_oside_normed = exp_pside_normed.copy()
        exp_time = df_exp['time']
    else:
        exp_pside_normed = None
        exp_oside_normed = None
        exp_time = None

    # Plot using analysis_utils
    if pside_normed is not None and sim_time is not None:
        plt.figure(figsize=(8, 5))
        au.plot_temperature_curves(
            sim_time=sim_time,
            sim_pside=pside_normed,
            sim_oside=oside_normed,
            exp_pside=exp_pside_normed,
            exp_oside=exp_oside_normed,
            exp_time=exp_time,
            show_plot=True
        )
    else:
        print("[WARNING] No simulation data to plot.")
    # --- End plotting section ---

    return results


if __name__ == "__main__":
    example_batch_run()
