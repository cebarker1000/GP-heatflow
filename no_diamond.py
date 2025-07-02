import run_no_diamond as run
import yaml
import os
import pandas as pd
import numpy as np
import analysis_utils as au
from dolfinx import fem

sim_name = 'geballe_no_diamond_read_flux'

# Load the configuration
with open(f'cfgs/{sim_name}.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Extract parameters from config (no diamond or gasket)
r_sample = float(cfg['mats']['p_sample']['r'])
z_sample = float(cfg['mats']['p_sample']['z'])
z_ins_pside = float(cfg['mats']['p_ins']['z'])
z_ins_oside = float(cfg['mats']['o_ins']['z'])
z_coupler = float(cfg['mats']['p_coupler']['z'])

# Calculate mesh boundaries (no diamond or gasket)
mesh_zmin = -(z_sample/2) - z_ins_pside - z_coupler
mesh_zmax = (z_sample/2) + z_ins_oside + z_coupler

# Calculate coupler boundary positions (no diamond)
bnd_p_ins_end = mesh_zmin + z_ins_pside
bnd_o_ins_start = mesh_zmax - z_ins_oside

pside_coupler_z = bnd_p_ins_end + z_coupler/2  # Middle of pside coupler
oside_coupler_z = bnd_o_ins_start - z_coupler/2  # Middle of oside coupler


# Define watcher points at the center of each coupler (r=0 for centerline)
watcher_points = {
    'pside': (pside_coupler_z, 0.0),  # (z, r) coordinates
    'oside': (oside_coupler_z, 0.0)   # (z, r) coordinates
}

# Run the simulation
run.run_simulation(
    cfg=cfg,
    mesh_folder=f'meshes/{sim_name}',
    rebuild_mesh=True,
    visualize_mesh=False,
    output_folder=f'outputs/{sim_name}',
    watcher_points=watcher_points,
    write_xdmf=True,  # No XDMF output as requested
    suppress_print=False
)

print(f"Simulation completed! Check outputs/{sim_name}/ for results.")

# Load simulation watcher data
watcher_csv_path = f'outputs/{sim_name}/watcher_points.csv'
if not os.path.exists(watcher_csv_path):
    print(f"Warning: Watcher data file not found at {watcher_csv_path}")
else:
    df_sim = pd.read_csv(watcher_csv_path)
    
    # Load experimental data
    df_exp = pd.read_csv('experimental_data/geballe_heat_data.csv')
    
    # Normalize simulation data
    sim_pside_normed = (df_sim['pside'] - df_sim['pside'].iloc[0]) / (df_sim['pside'].max() - df_sim['pside'].min())
    sim_oside_normed = (df_sim['oside'] - df_sim['oside'].iloc[0]) / (df_sim['pside'].max() - df_sim['pside'].min())
    
    # Normalize experimental data
    exp_pside_normed = (df_exp['temp'] - df_exp['temp'].iloc[0]) / (df_exp['temp'].max() - df_exp['temp'].min())
    
    # Downshift experimental oside to start from ic_temp and normalize
    ic_temp = cfg['heating']['ic_temp']
    oside_initial = df_exp['oside'].iloc[0]
    exp_oside_shifted = df_exp['oside'] - oside_initial + ic_temp
    exp_oside_normed = (exp_oside_shifted - exp_oside_shifted.iloc[0]) / (df_exp['temp'].max() - df_exp['temp'].min())
    
    # Plot normalized temperature curves using analysis_utils
    au.plot_temperature_curves(
        sim_time=df_sim['time'],
        sim_pside=sim_pside_normed,
        sim_oside=sim_oside_normed,
        exp_pside=exp_pside_normed,
        exp_oside=exp_oside_normed,
        exp_time=df_exp['time'],
        save_path=f'outputs/{sim_name}/temperature_curves.png',
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






