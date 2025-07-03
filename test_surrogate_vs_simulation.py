#!/usr/bin/env python3
"""
Compare surrogate GP model predictions to actual simulations for random parameter sets.
Draws 10 random parameter sets, predicts with surrogate, runs actual simulation, projects actual curve to FPCA, and overlays both.
"""

import numpy as np
import matplotlib.pyplot as plt
from uq_wrapper import (
    run_single_simulation, project_curve_to_fpca, reconstruct_curve_from_fpca, load_fpca_model
)
from create_full_surrogate_model import FullSurrogateModel, get_parameter_ranges
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')

# Number of test runs
N_TEST = 10

# Load surrogate model
surrogate = FullSurrogateModel.load_model('outputs/full_surrogate_model.pkl')
fpca_model = surrogate.fpca_model
param_ranges = get_parameter_ranges()
param_names = list(param_ranges.keys())

# Draw random parameter sets
low = [param_ranges[name][0] for name in param_names]
high = [param_ranges[name][1] for name in param_names]
test_samples = np.random.uniform(low=low, high=high, size=(N_TEST, len(param_names)))

# Store results
results = []

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
        config_path="cfgs/geballe_no_diamond.yaml",
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

print("\nAll overlays complete. Check the outputs/ directory for results.") 