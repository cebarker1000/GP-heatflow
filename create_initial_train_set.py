import numpy as np
from UQpy.sampling.stratified_sampling import LatinHypercubeSampling
from UQpy.distributions.collection import Uniform, Normal
from uq_wrapper import run_single_simulation, run_batch_simulations, save_batch_results, load_batch_results
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta

# Parameter definitions from create_initial_train_set.py
param_defs = [
    {"name": "d_sample", 
        "type": "lognormal", 
        "center": 1.84e-6, 
        "sigma_log": 0.079},
    {"name": "rho_cv_sample", 
        "type": "lognormal", 
        "center": 2764828, 
        "sigma_log": 0.079},
    {"name": "rho_cv_coupler", 
        "type": "lognormal", 
        "center": 3445520, 
        "sigma_log": 0.079},
    {"name": "rho_cv_ins", 
        "type": "lognormal", 
        "center": 2764828, 
        "sigma_log": 0.079},
    {"name": "d_coupler", 
        "type": "lognormal", 
        "center": 6.2e-8, 
        "sigma_log": 0.204},
    {"name": "d_ins_oside", 
        "type": "lognormal", 
        "center": 3.2e-6, 
        "sigma_log": 0.001},
    {"name": "d_ins_pside", 
        "type": "lognormal", 
        "center": 6.3e-6, 
        "sigma_log": 0.001},
    {"name" : "fwhm",
        "type": "lognormal",
        "center": 12e-6,
        "sigma_log": 0.041},
    {"name" : "k_sample",
        "type": "uniform",
        "low": 2.8,
        "high": 4.8},
    {"name" : "k_ins",
        "type": "uniform",
        "low": 7,
        "high": 13.0},
    {"name" : "k_coupler",
        "type": "uniform",
        "low": 300,
        "high": 400},
]

# Parameter mapping to config structure - each param can map to multiple locations
PARAM_MAPPING = {
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

# Create UQpy distributions for each parameter
distributions = []
for p in param_defs:
    if p["type"] == "lognormal":
        mu = np.log(p["center"])
        sigma = p["sigma_log"]
        distributions.append(Normal(loc=mu, scale=sigma))
    elif p["type"] == "normal":
        distributions.append(Normal(loc=p["center"], scale=p["sigma"]))
    elif p["type"] == "uniform":
        distributions.append(Uniform(loc=p["low"], scale=p["high"] - p["low"]))
    else:
        raise ValueError(f"Unknown type: {p['type']}")

# Number of samples
n_samples = 200

# Latin Hypercube Sampling
lhs = LatinHypercubeSampling(distributions=distributions, nsamples=n_samples)
samples = lhs.samples

# Exponentiate lognormal parameters
for i, p in enumerate(param_defs):
    if p["type"] == "lognormal":
        samples[:, i] = np.exp(samples[:, i])

print("LHS samples (physical space):")
print(samples)

header = ",".join([p["name"] for p in param_defs])
np.savetxt("outputs/initial_train_set.csv", samples, delimiter=",", header=header, comments='')

# Example usage for running simulations
if __name__ == "__main__":
    
    # Run batch simulations
    print("\nRunning batch simulations...")
    print(f"Running {len(samples)} simulations...")
    
    # Timing variables
    start_time = time.time()
    simulation_times = []
    
    # Add progress callback with timing
    def progress_callback(current, total):
        if current == 0:
            # First simulation starting
            print(f"Starting simulation {current + 1}/{total}...")
        else:
            # Calculate time for previous simulation
            sim_time = time.time() - start_time - sum(simulation_times)
            simulation_times.append(sim_time)
            
            # Calculate average time per simulation
            avg_time = np.mean(simulation_times)
            
            # Calculate remaining simulations and predicted time
            remaining_sims = total - current
            predicted_remaining = avg_time * remaining_sims
            
            # Format times nicely
            sim_time_str = str(timedelta(seconds=int(sim_time)))
            avg_time_str = str(timedelta(seconds=int(avg_time)))
            remaining_str = str(timedelta(seconds=int(predicted_remaining)))
            
            print(f"Simulation {current}/{total} completed in {sim_time_str} (avg: {avg_time_str})")
            print(f"Starting simulation {current + 1}/{total}... (est. remaining: {remaining_str})")
    
    results = run_batch_simulations(samples, param_defs, PARAM_MAPPING, suppress_print=True, progress_callback=progress_callback)
    
    # Final timing summary
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    avg_time = np.mean(simulation_times) if simulation_times else 0
    avg_time_str = str(timedelta(seconds=int(avg_time)))
    
    print(f"\nAll simulations completed!")
    print(f"Total time: {total_time_str}")
    print(f"Average time per simulation: {avg_time_str}")
    
    # Count successful vs failed simulations
    successful = sum(1 for r in results if 'error' not in r)
    failed = len(results) - successful
    print(f"\nSimulation Summary:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        # Save results
        save_batch_results(results, param_defs, output_file="outputs/uq_batch_results.npz")
        
        # Load and verify the saved data
        print("\nVerifying saved data...")
        loaded_data = load_batch_results("outputs/uq_batch_results.npz")
        print(f"Loaded data keys: {list(loaded_data.keys())}")
        print(f"Oside curves shape: {loaded_data['oside_curves'].shape}")
        print(f"Parameters shape: {loaded_data['parameters'].shape}")
        print(f"Parameter names: {loaded_data['parameter_names']}")
        
        # Show some statistics about the oside curves
        oside_curves = loaded_data['oside_curves']
        valid_curves = oside_curves[~np.isnan(oside_curves).any(axis=1)]
        print(f"Valid oside curves: {len(valid_curves)}")
        if len(valid_curves) > 0:
            print(f"Oside curve statistics:")
            print(f"  Mean max value: {np.mean(np.max(valid_curves, axis=1)):.3f}")
            print(f"  Mean min value: {np.mean(np.min(valid_curves, axis=1)):.3f}")
            print(f"  Mean range: {np.mean(np.max(valid_curves, axis=1) - np.min(valid_curves, axis=1)):.3f}")
    else:
        print("No successful simulations to save!")
