import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_temperature_curves(sim_time, sim_pside, sim_oside, exp_pside, exp_oside, 
                          exp_time=None, save_path=None, show_plot=True):
    """
    Plot temperature curves comparing simulation and experimental data.
    
    Parameters:
    -----------
    sim_time : pd.Series
        Simulation time data
    sim_pside : pd.Series
        Simulation pside temperature data
    sim_oside : pd.Series
        Simulation oside temperature data
    exp_pside : pd.Series
        Experimental pside temperature data
    exp_oside : pd.Series
        Experimental oside temperature data
    exp_time : pd.Series, optional
        Experimental time data. If None, will use the same time points as exp_pside/exp_oside
    save_path : str, optional
        Path to save the plot. If None, plot is not saved
    show_plot : bool, default True
        Whether to display the plot
    """
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot simulation curves
    plt.plot(sim_time, sim_pside, 'b-', linewidth=2, label='Sim P-side')
    plt.plot(sim_time, sim_oside, 'r-', linewidth=2, label='Sim O-side')
    
    # Plot experimental points
    if exp_time is not None:
        plt.scatter(exp_time, exp_pside, color='blue', marker='o', s=40, label='Exp P-side')
        plt.scatter(exp_time, exp_oside, color='red', marker='o', s=40, label='Exp O-side')
    else:
        # If no separate exp_time, assume exp_pside and exp_oside have the same time index
        plt.scatter(exp_pside.index, exp_pside, color='blue', marker='o', s=40, label='Exp P-side')
        plt.scatter(exp_oside.index, exp_oside, color='red', marker='o', s=40, label='Exp O-side')
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Temperature (K)', fontsize=12)
    plt.title('Temperature: Simulation vs Experiment', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temperature curves plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def calculate_rmse(exp_time, exp_data, sim_time, sim_data):
    """
    Calculate RMSE between experimental and simulation data at experimental time points.
    
    Parameters:
    -----------
    exp_time : pd.Series or array-like
        Experimental time points
    exp_data : pd.Series or array-like
        Experimental data values
    sim_time : pd.Series or array-like
        Simulation time points
    sim_data : pd.Series or array-like
        Simulation data values
    
    Returns:
    --------
    float
        RMSE value between the two datasets
    """
    
    # Interpolate simulation data at experimental time points
    sim_data_at_exp_times = np.interp(exp_time, sim_time, sim_data)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((sim_data_at_exp_times - exp_data)**2))
    
    return rmse
