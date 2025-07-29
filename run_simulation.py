"""
Optimized heat flow simulation runner with full YAML configuration.
"""

import os
import sys
import yaml
import argparse
import time
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Baseline helper (same rules as core.simulation_engine)
# ------------------------------------------------------------


def _compute_baseline(times: np.ndarray, temps: np.ndarray, cfg: Dict[str, Any]):
    """Compute baseline according to cfg['baseline'] settings."""
    baseline_cfg = cfg.get('baseline', {})
    if not baseline_cfg.get('use_average', False):
        return float(temps[0])

    t_window = float(baseline_cfg.get('time_window', 0.0))
    mask = times <= t_window
    if mask.any():
        return float(np.mean(temps[mask]))
    return float(temps[0])

from core.simulation_engine import OptimizedSimulationEngine, suppress_output
from analysis import analysis_utils as au


def get_default_paths():
    """Get default paths from environment variables or config."""
    return {
        'mesh_dir': os.getenv('V2HEATFLOW_MESH_DIR', 'data/meshes'),
        'output_dir': os.getenv('V2HEATFLOW_OUTPUT_DIR', 'outputs'),
        'experimental_data': os.getenv('V2HEATFLOW_EXPERIMENTAL_DATA', None)
    }



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


def run_simulation(cfg: Dict[str, Any], output_dir: str,
                  rebuild_mesh: bool = False, suppress_output_flag: bool = False,
                  no_plots: bool = False, no_xdmf: bool = False, mesh_vis: bool = False,
                  mesh_dir: Optional[str] = None, experimental_data: Optional[str] = None,
                  config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the heat flow simulation using the optimized engine.
    
    Parameters:
    -----------
    cfg : dict
        Configuration dictionary loaded from YAML
    output_dir : str
        Where to save simulation outputs
    rebuild_mesh : bool, optional
        Whether to rebuild the mesh and update material tags
    suppress_output_flag : bool, optional
        If True, suppress all print output
    no_plots : bool, optional
        If True, skip plotting temperature curves
    no_xdmf : bool, optional
        If True, skip XDMF file creation
    mesh_vis : bool, optional
        Whether to visualize the mesh
    mesh_dir : str, optional
        Override mesh directory path
    experimental_data : str, optional
        Override experimental data file path
    config_path : str, optional
        Path to the configuration file
    
    Returns:
    --------
    dict
        Simulation results including timing information
    """
    
    # Get default paths
    defaults = get_default_paths()
    
    # Determine mesh directory
    if mesh_dir is None:
        mesh_dir = cfg.get('io', {}).get('mesh_path', defaults['mesh_dir'])
    
    # Override experimental data path if provided
    if experimental_data is not None:
        cfg['output']['analysis']['experimental_data_file'] = experimental_data
    
    # Disable XDMF if requested
    if no_xdmf:
        cfg['output']['xdmf']['enabled'] = False
    
    # Create simulation engine
    engine = OptimizedSimulationEngine(cfg, mesh_dir, output_dir, config_path)
    
    # Run simulation
    results = engine.run(
        rebuild_mesh=rebuild_mesh,
        visualize_mesh=mesh_vis,
        suppress_print=suppress_output_flag
    )
    
    # Save configuration used for this run
    config_save_path = os.path.join(output_dir, 'used_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    
    # Plot results if requested
    if not no_plots:
        with suppress_output(suppress_output_flag):
            plot_temperature_curves(cfg, output_dir)
    
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
        
        # Load experimental data - check for processed data first
        processed_data_path = os.path.join(output_folder, 'processed_experimental_data.csv')
        if os.path.exists(processed_data_path):
            print(f"Loading processed experimental data from: {processed_data_path}")
            df_exp = pd.read_csv(processed_data_path)
        else:
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
        
        # Compute baselines using the same rules as the simulation
        times_sim = df_sim['time'].values
        pside_baseline_sim = _compute_baseline(times_sim, df_sim[pside_col].values, cfg)
        oside_baseline_sim = _compute_baseline(times_sim, df_sim[oside_col].values, cfg)

        pside_excursion = (df_sim[pside_col] - pside_baseline_sim).max() - (df_sim[pside_col] - pside_baseline_sim).min()
        if pside_excursion == 0:
            print("Warning: pside excursion is zero; normalization skipped")
            return

        sim_pside_normed = (df_sim[pside_col] - pside_baseline_sim) / pside_excursion
        sim_oside_normed = (df_sim[oside_col] - oside_baseline_sim) / pside_excursion
        
        # Check if smoothing was applied and handle experimental data accordingly
        smoothing_enabled = cfg.get('heating', {}).get('smoothing', {}).get('enabled', False)
        
        if smoothing_enabled and 'temp_raw' in df_exp.columns:
            # Compute baseline on smoothed data for consistency
            times_exp = df_exp['time'].values
            pside_baseline_exp = _compute_baseline(times_exp, df_exp['temp'].values, cfg)

            smoothed_excursion = (df_exp['temp'] - pside_baseline_exp).max() - (df_exp['temp'] - pside_baseline_exp).min()
            exp_pside_normed = (df_exp['temp'] - pside_baseline_exp) / smoothed_excursion
            exp_pside_raw_normed = (df_exp['temp_raw'] - pside_baseline_exp) / smoothed_excursion
        else:
            times_exp = df_exp['time'].values
            pside_baseline_exp = _compute_baseline(times_exp, df_exp['temp'].values, cfg)
            exp_temp_for_norm = df_exp['temp']
            exp_temp_raw = None
            exp_excursion = (exp_temp_for_norm - pside_baseline_exp).max() - (exp_temp_for_norm - pside_baseline_exp).min()
            exp_pside_normed = (df_exp['temp'] - pside_baseline_exp) / exp_excursion
            exp_pside_raw_normed = None
        
        # Handle experimental oside data if available
        if 'oside' in df_exp.columns:
            oside_baseline_exp = _compute_baseline(times_exp, df_exp['oside'].values, cfg)
            exp_oside_normed = (df_exp['oside'] - oside_baseline_exp) / exp_excursion
        else:
            # If no oside data, use pside data for both (common in some experiments)
            exp_oside_normed = exp_pside_normed.copy()
        
        # Plot normalized temperature curves with enhanced experimental data display
        plot_path = os.path.join(output_folder, 'temperature_curves.png')
        
        # Create single plot showing simulation and experimental data
        plt.figure(figsize=(12, 8))
        
        # Plot simulation curves
        plt.plot(df_sim['time'], sim_pside_normed, 'b-', linewidth=2, label='Sim P-side')
        plt.plot(df_sim['time'], sim_oside_normed, 'r-', linewidth=2, label='Sim O-side')
        
        # Plot experimental data
        if exp_pside_raw_normed is not None:
            # When smoothing is enabled, show both raw and smoothed
            plt.scatter(df_exp['time'], exp_pside_normed, color='blue', marker='o', s=40, label='Exp P-side (smoothed)')
            plt.scatter(df_exp['time'], exp_pside_raw_normed, color='lightblue', marker='x', s=30, 
                       alpha=0.7, label='Exp P-side (raw)')
        else:
            # If no smoothing, just show the regular pside data
            plt.scatter(df_exp['time'], exp_pside_normed, color='blue', marker='o', s=40, label='Exp P-side')
        
        plt.scatter(df_exp['time'], exp_oside_normed, color='red', marker='o', s=40, label='Exp O-side')
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Normalized Temperature', fontsize=12)
        plt.title('Temperature: Simulation vs Experiment', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Temperature curves plot saved to: {plot_path}")
        plt.show()
        
        # Create residual plot for oside data
        residual_plot_path = os.path.join(output_folder, 'residual_plot.png')
        au.plot_residuals(
            exp_time=df_exp['time'],
            exp_data=exp_oside_normed,
            sim_time=df_sim['time'],
            sim_data=sim_oside_normed,
            save_path=residual_plot_path,
            show_plot=True
        )
        
        # Calculate RMSE for oside data
        oside_rmse = au.calculate_rmse(
            exp_time=df_exp['time'],
            exp_data=exp_oside_normed,
            sim_time=df_sim['time'],
            sim_data=sim_oside_normed
        )
        
        # Calculate residual variance (sensor noise) for oside data
        oside_residual_stats = au.calculate_residual_variance(
            exp_time=df_exp['time'],
            exp_data=exp_oside_normed,
            sim_time=df_sim['time'],
            sim_data=sim_oside_normed
        )
        
        print(f"\n--- Analysis Results ---")
        print(f"O-side RMSE: {oside_rmse:.4f}")
        print(f"Residual Variance (sensor noise): {oside_residual_stats['variance']:.6f}")
        print(f"Residual Std Dev (sensor noise): {oside_residual_stats['std']:.6f}")
        if smoothing_enabled:
            print(f"Savitzky-Golay smoothing: Enabled")
        print("------------------------\n")
        
        # Save residual statistics to a file for further analysis
        residual_stats_path = os.path.join(output_folder, 'residual_statistics.txt')
        with open(residual_stats_path, 'w') as f:
            f.write("Residual Analysis Results\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"O-side RMSE: {oside_rmse:.6f}\n")
            f.write(f"Residual Variance (sensor noise): {oside_residual_stats['variance']:.8f}\n")
            f.write(f"Residual Standard Deviation (sensor noise): {oside_residual_stats['std']:.8f}\n")
            f.write(f"Number of data points: {len(oside_residual_stats['residuals'])}\n")
            f.write(f"Residual range: [{oside_residual_stats['residuals'].min():.6f}, {oside_residual_stats['residuals'].max():.6f}]\n")
            if smoothing_enabled:
                f.write(f"Savitzky-Golay smoothing: Enabled\n")
        
        print(f"Residual statistics saved to: {residual_stats_path}")
        
    except Exception as e:
        print(f"Error plotting results: {e}")
        import traceback
        traceback.print_exc()


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
    parser.add_argument('--suppress-output', action='store_true',
                       help='Suppress all output')
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
        
        # Use suppress_output context manager for complete silence
        with suppress_output(args.suppress_output):
            print(f"Starting simulation with configuration: {args.config}")
            print(f"Mesh folder: {mesh_folder}")
            print(f"Output folder: {output_folder}")
            
            # Run simulation
            results = run_simulation(
                cfg=cfg,
                output_dir=output_folder,
                rebuild_mesh=args.rebuild_mesh,
                suppress_output_flag=args.suppress_output,
                no_plots=not args.plot,  # Invert the plot flag
                no_xdmf=False,  # Default to creating XDMF files
                mesh_vis=args.visualize_mesh,
                mesh_dir=args.mesh_folder,
                experimental_data=None,  # Use config file setting
                config_path=args.config
            )
            
            # Print results
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