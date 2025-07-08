#!/usr/bin/env python3
"""
Compare surrogate GP model predictions to actual simulations for random parameter sets.
Draws 10 random parameter sets, predicts with surrogate, runs actual simulation, projects actual curve to FPCA, and overlays both.
"""

import sys
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from analysis.uq_wrapper import (
    run_single_simulation,
    project_curve_to_fpca,
    reconstruct_curve_from_fpca,
    load_fpca_model,
    load_recast_training_data,
)
from train_surrogate_models import FullSurrogateModel, get_parameter_ranges
import warnings

import sys

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Number of test runs
N_TEST = 15

OUTPUT_DIR = "outputs"

# -----------------------------------------------------------------------------
# LOAD MODELS & TRAINING DATA
# -----------------------------------------------------------------------------
surrogate = FullSurrogateModel.load_model(f"{OUTPUT_DIR}/full_surrogate_model.pkl")
fpca_model = surrogate.fpca_model
param_ranges = get_parameter_ranges()
param_names = list(param_ranges.keys())

# Load original training parameters to measure distance-to-training
try:
    recast_data = load_recast_training_data(f"{OUTPUT_DIR}/training_data_fpca.npz")
    X_train = recast_data["parameters"]
    X_train_scaled = surrogate.scaler.transform(X_train)
    nn_model = NearestNeighbors(n_neighbors=1).fit(X_train_scaled)
except Exception as e:
    print("[WARNING] Could not load training parameters for distance diagnostics:", e)
    nn_model = None

# Draw random parameter sets
low = [param_ranges[name][0] for name in param_names]
high = [param_ranges[name][1] for name in param_names]
test_samples = np.random.uniform(low=low, high=high, size=(N_TEST, len(param_names)))

# Prepare containers for extended diagnostics
results = []
diagnostics = []  # list of dicts – one per successful simulation

# Default param_defs and param_mapping (copied from create_initial_train_set.py)
param_defs = [
    {"name": "d_sample", "type": "lognormal", "center": 1.84e-6, "sigma_log": 0.079},
    {"name": "rho_cv_sample", "type": "lognormal", "center": 2764828, "sigma_log": 0.079},
    {"name": "rho_cv_coupler", "type": "lognormal", "center": 3445520, "sigma_log": 0.079},
    {"name": "rho_cv_ins", "type": "lognormal", "center": 2764828, "sigma_log": 0.079},
    {"name": "d_coupler", "type": "lognormal", "center": 6.2e-8, "sigma_log": 0.204},
    {"name": "d_ins_oside", "type": "lognormal", "center": 3.2e-6, "sigma_log": 0.001},
    {"name": "d_ins_pside", "type": "lognormal", "center": 6.3e-6, "sigma_log": 0.001},
    {"name": "fwhm", "type": "lognormal", "center": 12e-6, "sigma_log": 0.041},
    {"name": "k_sample", "type": "uniform", "low": 2.8, "high": 4.8},
    {"name": "k_ins", "type": "uniform", "low": 7, "high": 13.0},
    {"name": "k_coupler", "type": "uniform", "low": 300, "high": 400},
]
param_mapping = {
    "d_sample": [("mats", "sample", "z")],
    "rho_cv_sample": [("mats", "sample", "rho_cv")],
    "rho_cv_coupler": [("mats", "p_coupler", "rho_cv"), ("mats", "o_coupler", "rho_cv")],
    "rho_cv_ins": [("mats", "p_ins", "rho_cv"), ("mats", "o_ins", "rho_cv")],
    "d_coupler": [("mats", "p_coupler", "z"), ("mats", "o_coupler", "z")],
    "d_ins_oside": [("mats", "o_ins", "z")],
    "d_ins_pside": [("mats", "p_ins", "z")],
    "fwhm": [("heating", "fwhm")],
    "k_sample": [("mats", "sample", "k")],
    "k_ins": [("mats", "p_ins", "k"), ("mats", "o_ins", "k")],
    "k_coupler": [("mats", "p_coupler", "k"), ("mats", "o_coupler", "k")],
}

for i, params in enumerate(test_samples):
    print(f"\nTest {i+1}/{N_TEST}")
    # Surrogate prediction
    surrogate_curve, surrogate_coeffs, _ = surrogate.predict_temperature_curves(params)
    surrogate_curve = surrogate_curve[0]  # shape (n_timepoints,)
    surrogate_coeffs = surrogate_coeffs[0]

    # Actual simulation
    sim_result = run_single_simulation(
        sample=params,
        param_defs=param_defs,
        param_mapping=param_mapping,
        simulation_index=i,
        config_path="configs/config_5_materials.yaml",
        suppress_print=True
    )
    if 'error' in sim_result:
        print(f"Simulation failed: {sim_result['error']}")
        results.append({'params': params, 'surrogate_curve': surrogate_curve, 'sim_curve': None, 'surrogate_coeffs': surrogate_coeffs, 'sim_coeffs': None})
        continue
    if 'watcher_data' not in sim_result or 'oside' not in sim_result['watcher_data']:
        print("Simulation did not return valid oside curve.")
        results.append({'params': params, 'surrogate_curve': surrogate_curve, 'sim_curve': None, 'surrogate_coeffs': surrogate_coeffs, 'sim_coeffs': None})
        continue
    sim_curve = sim_result['watcher_data']['oside']['normalized']
    # Project actual curve to FPCA
    sim_coeffs = project_curve_to_fpca(sim_curve, fpca_model)
    # Reconstruct from FPCA (should be nearly identical to sim_curve)
    sim_curve_reconstructed = reconstruct_curve_from_fpca(sim_coeffs, fpca_model)

    # ---------------------------------------------------------------------
    # DIAGNOSTIC METRICS
    # ---------------------------------------------------------------------
    # FPCA truncation error (simulation vs. reconstruction with identical PCs)
    fpca_trunc_err = sim_curve - sim_curve_reconstructed
    rmse_fpca_trunc = np.sqrt(np.mean(fpca_trunc_err ** 2))

    # Coefficient errors & GP predictive std
    coeff_err_vec = surrogate_coeffs - sim_coeffs
    coeff_err_l2 = np.linalg.norm(coeff_err_vec)
    coeff_err_abs = np.abs(coeff_err_vec)

    # GP posterior std for each component
    X_scaled = surrogate.scaler.transform(params.reshape(1, -1))
    gp_stds = [gp.predict(X_scaled, return_std=True)[1][0] for gp in surrogate.gps]

    # GP-induced curve error (with same truncation)
    gp_curve_err = surrogate_curve - sim_curve_reconstructed
    rmse_gp_curve = np.sqrt(np.mean(gp_curve_err ** 2))

    # Total curve error
    total_curve_err = surrogate_curve - sim_curve
    rmse_total = np.sqrt(np.mean(total_curve_err ** 2))

    # Distance to nearest training point (scaled parameter space)
    if nn_model is not None:
        dist, _ = nn_model.kneighbors(X_scaled)
        nearest_dist = dist[0][0]
    else:
        nearest_dist = np.nan

    diagnostics.append({
        "rmse_fpca_trunc": rmse_fpca_trunc,
        "rmse_gp_curve": rmse_gp_curve,
        "rmse_total": rmse_total,
        "coeff_err_l2": coeff_err_l2,
        **{f"coeff_err_pc{i+1}": coeff_err_abs[i] for i in range(len(coeff_err_vec))},
        **{f"gp_std_pc{i+1}": gp_stds[i] for i in range(len(gp_stds))},
        "nearest_dist": nearest_dist,
    })

    results.append({
        'params': params,
        'surrogate_curve': surrogate_curve,
        'sim_curve': sim_curve,
        'surrogate_coeffs': surrogate_coeffs,
        'sim_coeffs': sim_coeffs,
        'sim_curve_reconstructed': sim_curve_reconstructed
    })

# Plot overlays
for i, res in enumerate(results):
    plt.figure(figsize=(10, 6))
    if res['sim_curve'] is not None:
        plt.plot(res['sim_curve'], label='Actual Simulation', color='black', linewidth=2)
        plt.plot(res['sim_curve_reconstructed'], '--', label='Sim FPCA Reconstruction', color='gray', linewidth=1)
    plt.plot(res['surrogate_curve'], label='Surrogate Prediction', color='tab:blue', linewidth=2)
    plt.title(f'Test {i+1}: Surrogate vs Simulation')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outputs/surrogate_vs_simulation_{i+1}.png', dpi=200)
    plt.close()
    print(f"Saved overlay plot for test {i+1} to outputs/surrogate_vs_simulation_{i+1}.png")

# -----------------------------------------------------------------------------
# SAVE & VISUALISE DIAGNOSTICS
# -----------------------------------------------------------------------------

if diagnostics:
    diag_df = pd.DataFrame(diagnostics)
    diag_csv_path = os.path.join(OUTPUT_DIR, "surrogate_diagnostics.csv")
    diag_df.to_csv(diag_csv_path, index=False)
    print(f"Saved diagnostic metrics to {diag_csv_path}")

    # Scatter: GP std vs. coefficient absolute error per PC
    num_pcs = len(surrogate.gps)
    for pc in range(num_pcs):
        plt.figure(figsize=(6, 4))
        plt.scatter(diag_df[f"gp_std_pc{pc+1}"], diag_df[f"coeff_err_pc{pc+1}"], c='tab:blue')
        plt.xlabel("GP posterior σ (PC%d)" % (pc + 1))
        plt.ylabel("|Coeff error| (PC%d)" % (pc + 1))
        plt.title(f"PC{pc+1}: Prediction uncert. vs. abs. error")
        plt.grid(alpha=0.3)
        fname = os.path.join(OUTPUT_DIR, f"gp_std_vs_coeff_err_pc{pc+1}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved scatter plot {fname}")

    # Scatter: distance to nearest training vs total curve RMSE
    if diag_df["nearest_dist"].notna().any():
        plt.figure(figsize=(6, 4))
        plt.scatter(diag_df["nearest_dist"], diag_df["rmse_total"], c='tab:green')
        plt.xlabel("Distance to nearest training point (scaled)")
        plt.ylabel("Curve RMSE (total)")
        plt.title("Extrapolation vs. error")
        plt.grid(alpha=0.3)
        fname = os.path.join(OUTPUT_DIR, "dist_vs_curve_rmse.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved scatter plot {fname}")


print("\nAll overlays and diagnostics complete. Check the outputs/ directory for results.") 