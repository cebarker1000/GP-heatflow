#!/usr/bin/env python3
"""
Example script demonstrating the optimized heat flow simulation with plotting.
"""

import os
import sys
import yaml
from run_optimized import run_simulation

def main():
    """Run an example simulation with plotting enabled."""
    
    # Configuration file
    config_file = 'cfgs/geballe_no_diamond.yaml'
    
    # Check if config exists
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found: {config_file}")
        print("Please ensure the configuration file exists.")
        return 1
    
    # Setup paths
    mesh_folder = 'meshes/geballe_no_diamond'
    output_folder = 'outputs/geballe_no_diamond'
    
    # Create directories
    os.makedirs(mesh_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    print("="*60)
    print("OPTIMIZED HEAT FLOW SIMULATION EXAMPLE")
    print("="*60)
    print(f"Configuration: {config_file}")
    print(f"Mesh folder: {mesh_folder}")
    print(f"Output folder: {output_folder}")
    print("="*60)
    
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        
        print("\nConfiguration loaded successfully!")
        print(f"Materials: {list(cfg['mats'].keys())}")
        print(f"Timing: {cfg['timing']['num_steps']} steps, {cfg['timing']['t_final']} s")
        print(f"Heating: {cfg['heating']['file']}")
        
        # Run simulation with plotting enabled
        print("\nStarting simulation with plotting enabled...")
        results = run_simulation(
            cfg=cfg,
            mesh_folder=mesh_folder,
            output_folder=output_folder,
            rebuild_mesh=True,  # Rebuild mesh for this example
            visualize_mesh=False,
            suppress_print=False,
            plot_results=True  # Enable plotting
        )
        
        # Print results
        print("\n" + "="*60)
        print("SIMULATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        timing = results.get('timing', {})
        print(f"Total simulation time: {timing.get('total_time', 0):.2f} seconds")
        print(f"Average time per step: {timing.get('avg_step_time', 0):.4f} seconds")
        
        # Check for output files
        watcher_file = os.path.join(output_folder, 'watcher_points.csv')
        plot_file = os.path.join(output_folder, 'temperature_curves.png')
        config_file_out = os.path.join(output_folder, 'used_config.yaml')
        
        print(f"\nOutput files:")
        print(f"  Watcher data: {watcher_file}")
        print(f"  Temperature plot: {plot_file}")
        print(f"  Used config: {config_file_out}")
        
        if os.path.exists(watcher_file):
            print(f"  ✓ Watcher data saved")
        if os.path.exists(plot_file):
            print(f"  ✓ Temperature plot saved")
        if os.path.exists(config_file_out):
            print(f"  ✓ Configuration saved")
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. View the temperature plot: temperature_curves.png")
        print("2. Analyze watcher data: watcher_points.csv")
        print("3. Modify configuration: cfgs/geballe_no_diamond.yaml")
        print("4. Run with different parameters:")
        print("   python run_optimized.py --config cfgs/geballe_no_diamond.yaml \\")
        print("     --mesh-folder meshes/geballe_no_diamond \\")
        print("     --output-folder outputs/geballe_no_diamond \\")
        print("     --plot")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\nError running simulation: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure all required files exist")
        print("2. Check that experimental data file is present")
        print("3. Verify DOLFINx environment is activated")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 