#!/usr/bin/env python3
"""
Test script to demonstrate Savitzky-Golay smoothing functionality.
This script loads experimental data, applies smoothing, and creates comparison plots.
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def load_and_smooth_data(data_file, window_length=11, polyorder=3):
    """
    Load experimental data and apply Savitzky-Golay smoothing.
    
    Parameters:
    -----------
    data_file : str
        Path to the experimental data CSV file
    window_length : int
        Window length for Savitzky-Golay filter (must be odd)
    polyorder : int
        Polynomial order for Savitzky-Golay filter
        
    Returns:
    --------
    tuple
        (time, temp_raw, temp_smoothed, oside) - experimental data arrays
    """
    print(f"Loading data from: {data_file}")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Sort by time and convert to numeric
    df = (df.sort_values('time')
            .assign(
                time=pd.to_numeric(df['time'], errors='coerce'),
                temp=pd.to_numeric(df['temp'], errors='coerce')
            )
            .dropna(subset=['time', 'temp'])
            .reset_index(drop=True))
    
    # Ensure window_length is odd and not larger than data length
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(df):
        window_length = len(df) if len(df) % 2 == 1 else len(df) - 1
    
    # Apply Savitzky-Golay filter
    temp_raw = np.array(df['temp'].values)
    temp_smoothed = savgol_filter(temp_raw, window_length, polyorder)
    
    # Get oside data if available
    oside = np.array(df['oside'].values) if 'oside' in df.columns else None
    
    print(f"Applied Savitzky-Golay smoothing: window={window_length}, polyorder={polyorder}")
    print(f"Data points: {len(temp_raw)}")
    print(f"Raw temp range: {np.min(temp_raw):.2f} to {np.max(temp_raw):.2f} K")
    print(f"Smoothed temp range: {np.min(temp_smoothed):.2f} to {np.max(temp_smoothed):.2f} K")
    
    return np.array(df['time'].values), temp_raw, temp_smoothed, oside

def plot_smoothing_comparison(time, temp_raw, temp_smoothed, oside=None, save_path=None):
    """
    Create comparison plots showing raw vs smoothed data.
    
    Parameters:
    -----------
    time : np.ndarray
        Time points
    temp_raw : np.ndarray
        Raw temperature data
    temp_smoothed : np.ndarray
        Smoothed temperature data
    oside : np.ndarray, optional
        O-side temperature data
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Temperature comparison
    axes[0].plot(time * 1e6, temp_raw, 'ko-', markersize=4, alpha=0.7, label='Raw data')
    axes[0].plot(time * 1e6, temp_smoothed, 'r-', linewidth=2, label='Savitzky-Golay smoothed')
    axes[0].set_xlabel('Time (μs)')
    axes[0].set_ylabel('Temperature (K)')
    axes[0].set_title('Experimental Heating Data: Raw vs Smoothed')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: O-side data if available
    if oside is not None:
        axes[1].plot(time * 1e6, oside, 'b-', linewidth=2, label='O-side data')
        axes[1].set_xlabel('Time (μs)')
        axes[1].set_ylabel('Temperature (K)')
        axes[1].set_title('O-side Temperature Data')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # Show difference between raw and smoothed
        diff = temp_smoothed - temp_raw
        axes[1].plot(time * 1e6, diff, 'g-', linewidth=2, label='Difference (smoothed - raw)')
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Time (μs)')
        axes[1].set_ylabel('Temperature Difference (K)')
        axes[1].set_title('Smoothing Effect: Smoothed - Raw')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def test_different_parameters(data_file):
    """
    Test different Savitzky-Golay parameters and show their effects.
    
    Parameters:
    -----------
    data_file : str
        Path to the experimental data file
    """
    print("\n" + "="*60)
    print("TESTING DIFFERENT SAVITZKY-GOLAY PARAMETERS")
    print("="*60)
    
    # Load raw data
    df = pd.read_csv(data_file)
    df = (df.sort_values('time')
            .assign(
                time=pd.to_numeric(df['time'], errors='coerce'),
                temp=pd.to_numeric(df['temp'], errors='coerce')
            )
            .dropna(subset=['time', 'temp'])
            .reset_index(drop=True))
    
    time = np.array(df['time'].values)
    temp_raw = np.array(df['temp'].values)
    
    # Test different parameters
    parameters = [
        (5, 2, 'Minimal smoothing'),
        (11, 3, 'Moderate smoothing (recommended)'),
        (21, 3, 'Heavy smoothing'),
        (11, 2, 'Lower polynomial order'),
        (11, 4, 'Higher polynomial order')
    ]
    
    fig, axes = plt.subplots(len(parameters), 1, figsize=(12, 3*len(parameters)))
    if len(parameters) == 1:
        axes = [axes]
    
    for i, (window, poly, title) in enumerate(parameters):
        # Ensure window is odd and valid
        if window % 2 == 0:
            window += 1
        if window > len(temp_raw):
            window = len(temp_raw) if len(temp_raw) % 2 == 1 else len(temp_raw) - 1
        
        # Apply smoothing
        temp_smoothed = savgol_filter(temp_raw, window, poly)
        
        # Plot
        axes[i].plot(time * 1e6, temp_raw, 'ko-', markersize=2, alpha=0.5, label='Raw data')
        axes[i].plot(time * 1e6, temp_smoothed, 'r-', linewidth=2, label=f'Smoothed (w={window}, p={poly})')
        axes[i].set_title(f'{title} - Window={window}, Polyorder={poly}')
        axes[i].set_xlabel('Time (μs)')
        axes[i].set_ylabel('Temperature (K)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('savgol_parameter_comparison.png', dpi=300, bbox_inches='tight')
    print("Parameter comparison plot saved to: savgol_parameter_comparison.png")
    plt.show()

def main():
    """Main function to test Savitzky-Golay smoothing."""
    print("="*60)
    print("SAVITZKY-GOLAY SMOOTHING TEST")
    print("="*60)
    
    # Test with both data files
    data_files = [
        "data/experimental/geballe_heat_data.csv",
        "data/experimental/edmund_71Gpa_run1.csv"
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"\nTesting with: {data_file}")
            
            try:
                # Test basic smoothing
                time, temp_raw, temp_smoothed, oside = load_and_smooth_data(
                    data_file, window_length=11, polyorder=3
                )
                
                # Create comparison plot
                plot_name = os.path.splitext(os.path.basename(data_file))[0]
                plot_path = f"savgol_smoothing_{plot_name}.png"
                plot_smoothing_comparison(time, temp_raw, temp_smoothed, oside, plot_path)
                
                # Test different parameters
                test_different_parameters(data_file)
                
            except Exception as e:
                print(f"Error processing {data_file}: {e}")
        else:
            print(f"Data file not found: {data_file}")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print("\nTo use Savitzky-Golay smoothing in simulations:")
    print("1. Add smoothing configuration to your YAML config file:")
    print("   heating:")
    print("     smoothing:")
    print("       enabled: true")
    print("       window_length: 11")
    print("       polyorder: 3")
    print("2. Run your simulation as usual")
    print("3. The output plots will show both raw and smoothed experimental data")

if __name__ == '__main__':
    main() 