"""
Optimized heat flow simulation runner with full YAML configuration.
"""

import os
import yaml
import argparse
import time
from typing import Dict, Any
import pandas as pd
import numpy as np
from simulation_engine import OptimizedSimulationEngine
import analysis_utils as au


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['heating', 'mats', 'timing']
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required configuration section: {section}")
    
    return cfg


def setup_paths(cfg: Dict[str, Any], mesh_folder: str = None, output_folder: str = None) -> tuple:
    """Setup mesh and output folders based on configuration."""
    # Determine mesh folder
    if mesh_folder is None:
        mesh_folder = cfg.get('io', {}).get('mesh_path', 'meshes/default')
    
    # Determine output folder
    if output_folder is None:
        sim_name = cfg.get('simulation_name', 'default_simulation')
        output_folder = f'outputs/{sim_name}'
    
    # Create directories
    os.makedirs(mesh_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    return mesh_folder, output_folder


def run_simulation(cfg: Dict[str, Any], mesh_folder: str, output_folder: str,
                  rebuild_mesh: bool = False, visualize_mesh: bool = False,
                  suppress_print: bool = False, plot_results: bool = False, 
                  config_path: str = None) -> Dict[str, Any]:
    """
    Run the heat flow simulation using the optimized engine.
    
    Parameters:
    -----------
    cfg : dict
        Configuration dictionary loaded from YAML
    mesh_folder : str
        Path to the folder containing mesh.msh and mesh_cfg.yaml
    output_folder : str
        Where to save simulation outputs
    rebuild_mesh : bool, optional
        Whether to rebuild the mesh and update material tags
    visualize_mesh : bool, optional
        Whether to visualize the mesh (overrides config setting)
    suppress_print : bool, optional
        If True, suppress all print output
    plot_results : bool, optional
        Whether to plot temperature curves at the end of the simulation
    
    Returns:
    --------
    dict
        Simulation results including timing information
    """
    
    # Create simulation engine
    engine = OptimizedSimulationEngine(cfg, mesh_folder, output_folder, config_path)
    
    # Run simulation
    results = engine.run(
        rebuild_mesh=rebuild_mesh,
        visualize_mesh=visualize_mesh,
        suppress_print=suppress_print
    )
    
    # Save configuration used for this run
    config_save_path = os.path.join(output_folder, 'used_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    
    # Plot results if requested
    if plot_results:
        plot_temperature_curves(cfg, output_folder)
    
    return results


def plot_temperature_curves(cfg, output_folder):
    """
    Plot temperature curves comparing simulation and experimental data.
    
    Parameters:
    -----------
    cfg : dict
        Configuration dictionary
    output_folder : str
        Path to the output folder
    """
    try:
        # Load simulation watcher data
        watcher_csv_path = os.path.join(output_folder, 'watcher_points.csv')
        if not os.path.exists(watcher_csv_path):
            print(f"Warning: Watcher data file not found at {watcher_csv_path}")
            return
        
        df_sim = pd.read_csv(watcher_csv_path)
        
        # Load experimental data
        exp_file = cfg['heating']['file']
        if not os.path.exists(exp_file):
            print(f"Warning: Experimental data file not found at {exp_file}")
            return
        
        df_exp = pd.read_csv(exp_file)
        
        # Get watcher point names from config
        watcher_points = cfg['output']['watcher_points']['points']
        sim_columns = list(watcher_points.keys())
        
        if len(sim_columns) < 2:
            print("Warning: Need at least 2 watcher points for comparison plot")
            return
        
        # Use first two watcher points for comparison (assuming pside and oside)
        pside_col = sim_columns[0]
        oside_col = sim_columns[1]
        
        # Normalize simulation data using pside temperature excursion for both
        pside_excursion = df_sim[pside_col].max() - df_sim[pside_col].min()
        sim_pside_normed = (df_sim[pside_col] - df_sim[pside_col].iloc[0]) / pside_excursion
        sim_oside_normed = (df_sim[oside_col] - df_sim[oside_col].iloc[0]) / pside_excursion
        
        # Normalize experimental data
        exp_pside_normed = (df_exp['temp'] - df_exp['temp'].iloc[0]) / (df_exp['temp'].max() - df_exp['temp'].min())
        
        # Handle experimental oside data if available
        if 'oside' in df_exp.columns:
            # Downshift experimental oside to start from ic_temp and normalize
            ic_temp = cfg['heating']['ic_temp']
            oside_initial = df_exp['oside'].iloc[0]
            exp_oside_shifted = df_exp['oside'] - oside_initial + ic_temp
            exp_oside_normed = (exp_oside_shifted - exp_oside_shifted.iloc[0]) / (df_exp['temp'].max() - df_exp['temp'].min())
        else:
            # If no oside data, use pside data for both (common in some experiments)
            exp_oside_normed = exp_pside_normed.copy()
        
        # Plot normalized temperature curves
        plot_path = os.path.join(output_folder, 'temperature_curves.png')
        au.plot_temperature_curves(
            sim_time=df_sim['time'],
            sim_pside=sim_pside_normed,
            sim_oside=sim_oside_normed,
            exp_pside=exp_pside_normed,
            exp_oside=exp_oside_normed,
            exp_time=df_exp['time'],
            save_path=plot_path,
            show_plot=True
        )
        
        # Calculate RMSE for oside data
        oside_rmse = au.calculate_rmse(
            exp_time=df_exp['time'],
            exp_data=exp_oside_normed,
            sim_time=df_sim['time'],
            sim_data=sim_oside_normed
        )
        
        print(f"\n--- RMSE Analysis ---")
        print(f"O-side RMSE: {oside_rmse:.4f}")
        print("-------------------\n")
        
    except Exception as e:
        print(f"Error plotting results: {e}")


def print_timing_summary(results: Dict[str, Any]):
    """Print timing summary from simulation results."""
    timing = results.get('timing', {})
    
    print("\n" + "="*50)
    print("SIMULATION TIMING SUMMARY")
    print("="*50)
    print(f"Total simulation time: {timing.get('total_loop_time', 0):.2f} seconds")
    print(f"Average time per step: {timing.get('avg_step_time', 0):.4f} seconds")
    print(f"Number of time steps: {results.get('num_steps', 0)}")
    
    if timing.get('total_loop_time', 0) > 0:
        steps_per_second = results.get('num_steps', 0) / timing.get('total_loop_time', 1)
        print(f"Simulation speed: {steps_per_second:.2f} steps/second")
    
    print("="*50)


def main():
    """Main entry point for the optimized simulation runner."""
    parser = argparse.ArgumentParser(
        description='Optimized heat flow simulation runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulation
  python run_optimized.py --config cfgs/geballe_no_diamond.yaml --mesh-folder meshes/geballe_no_diamond --output-folder outputs/geballe_no_diamond

  # Rebuild mesh and plot results
  python run_optimized.py --config cfgs/geballe_no_diamond.yaml --mesh-folder meshes/geballe_no_diamond --output-folder outputs/geballe_no_diamond --rebuild-mesh --plot

  # Suppress output and visualize mesh
  python run_optimized.py --config cfgs/geballe_no_diamond.yaml --mesh-folder meshes/geballe_no_diamond --output-folder outputs/geballe_no_diamond --visualize-mesh --suppress-print
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to the YAML configuration file')
    parser.add_argument('--mesh-folder', type=str, required=True,
                       help='Path to the mesh folder')
    parser.add_argument('--output-folder', type=str, required=True,
                       help='Path to the output folder')
    parser.add_argument('--rebuild-mesh', action='store_true',
                       help='Rebuild the mesh')
    parser.add_argument('--visualize-mesh', action='store_true',
                       help='Visualize the mesh')
    parser.add_argument('--suppress-print', action='store_true',
                       help='Suppress print output')
    parser.add_argument('--plot', action='store_true',
                       help='Plot temperature curves at the end of simulation')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    try:
        # Load configuration
        cfg = load_config(args.config)
        
        # Use command line arguments directly since they are required
        mesh_folder = args.mesh_folder
        output_folder = args.output_folder
        
        print(f"Starting simulation with configuration: {args.config}")
        print(f"Mesh folder: {mesh_folder}")
        print(f"Output folder: {output_folder}")
        
        # Run simulation
        results = run_simulation(
            cfg=cfg,
            mesh_folder=mesh_folder,
            output_folder=output_folder,
            rebuild_mesh=args.rebuild_mesh,
            visualize_mesh=args.visualize_mesh,
            suppress_print=args.suppress_print,
            plot_results=args.plot,
            config_path=args.config
        )
        
        # Print results
        if not args.suppress_print:
            print_timing_summary(results)
            print(f"\nSimulation completed successfully!")
            print(f"Results saved to: {output_folder}")
            if args.plot:
                print(f"Temperature curves plot: {os.path.join(output_folder, 'temperature_curves.png')}")
        
    except Exception as e:
        print(f"Error running simulation: {e}")
        raise


if __name__ == '__main__':
    main() 