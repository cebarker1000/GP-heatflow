import numpy as np
import matplotlib.pyplot as plt
from uqpy_surrogate import timeseries_model
from analysis.config_utils import get_param_defs_from_config, create_uqpy_distributions
from UQpy.distributions.collection.JointIndependent import JointIndependent

# Define fixed k values for synthetic data generation (same as in test_full_parameter_mcmc.py)
FIXED_K_VALUES = {
    'k_sample': 3.3,
    'k_ins': 10.0,
    'k_coupler': 350.0
}

def get_true_parameter_array():
    """Generate true parameters with fixed k values but random nuisance parameters from priors"""
    param_defs = get_param_defs_from_config(config_path="configs/distributions.yaml")
    uqpy_dists = create_uqpy_distributions(param_defs)
    param_names = [param_def['name'] for param_def in param_defs]
    
    # Draw nuisance parameters from their priors
    nuisance_prior = JointIndependent(marginals=uqpy_dists[:8])  # First 8 parameters
    nuisance_params = nuisance_prior.rvs(nsamples=1)[0]  # Shape (8,)
    
    # Create full parameter array
    true_array = np.zeros(11)
    true_array[:8] = nuisance_params  # Nuisance parameters from priors
    
    # Set fixed k values
    k_indices = [8, 9, 10]  # k_sample, k_ins, k_coupler
    for i, k_name in enumerate(['k_sample', 'k_ins', 'k_coupler']):
        true_array[k_indices[i]] = FIXED_K_VALUES[k_name]
    
    return true_array, param_names

def generate_synthetic_data():
    """Generate synthetic data using fixed k values but random nuisance parameters from priors"""
    true_params, param_names = get_true_parameter_array()
    
    # Print parameter summary
    print("True parameters used for synthetic data:")
    print(f"{'Parameter':<15} {'Value':<15} {'Source':<15}")
    print("-" * 45)
    for i, name in enumerate(param_names):
        if i < 8:
            source = "Prior draw"
        else:
            source = "Fixed"
        print(f"{name:<15} {true_params[i]:<15.3e} {source:<15}")
    
    # Generate synthetic data using the surrogate model
    synthetic_data = timeseries_model(true_params.reshape(1, -1))[0]  # Shape (T,)
    
    # Add noise to simulate experimental uncertainty
    SENSOR_VARIANCE = 1e-4
    noise = np.random.normal(0, np.sqrt(SENSOR_VARIANCE), synthetic_data.shape)
    noisy_data = synthetic_data + noise
    
    print(f"\nSynthetic data shape: {noisy_data.shape}")
    print(f"Data range: [{noisy_data.min():.6f}, {noisy_data.max():.6f}]")
    
    return noisy_data, true_params, param_names

def main():
    print("Testing synthetic data generation...")
    print("Using fixed k values but random nuisance parameters from priors")
    
    # Generate synthetic data
    synthetic_data, true_params, param_names = generate_synthetic_data()
    
    # Create time grid for plotting
    sim_t_final = 7.5e-6  # seconds
    sim_num_steps = 50
    time_grid = np.linspace(0, sim_t_final, sim_num_steps)
    
    # Plot the synthetic data
    plt.figure(figsize=(10, 6))
    plt.plot(time_grid * 1e6, synthetic_data, 'b-', linewidth=2, label='Synthetic data')
    plt.xlabel('Time (μs)')
    plt.ylabel('Normalized temperature')
    plt.title('Synthetic experimental data (fixed k, random nuisance params)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("test_synthetic_data.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Print parameter summary with units
    param_defs = get_param_defs_from_config(config_path="configs/distributions.yaml")
    
    print("\nParameter Summary:")
    print(f"{'Parameter':<15} {'Value':<15} {'Units':<15} {'Source':<15}")
    print("-" * 60)
    for i, name in enumerate(param_names):
        units = param_defs[i].get('units', 'N/A')
        if i < 8:
            source = "Prior draw"
        else:
            source = "Fixed"
        print(f"{name:<15} {true_params[i]:<15.3e} {units:<15} {source:<15}")
    
    print(f"\nSynthetic data saved to test_synthetic_data.png")
    print("You can now run test_full_parameter_mcmc.py to test the MCMC recovery")
    print("Note: k values are fixed and should be recovered well")
    print("Nuisance parameters were drawn from priors and represent experimental uncertainty")

if __name__ == "__main__":
    main() 