#!/usr/bin/env python3
"""
UQpy-compatible wrapper for the heat flow simulation with FPCA output.

This module provides a wrapper that can be used with UQpy's RunModel class.
The wrapper takes parameter arrays and returns FPCA coefficients instead of
full temperature curves, enabling efficient UQ analysis.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from uq_wrapper import (load_fpca_model, project_curve_to_fpca, 
                       run_single_simulation, create_config_from_sample)
import time
import warnings
warnings.filterwarnings('ignore')


class HeatFlowFPCAWrapper:
    """
    UQpy-compatible wrapper for heat flow simulation with FPCA output.
    
    This class provides a callable interface that takes parameter arrays
    and returns FPCA coefficients, making it compatible with UQpy's RunModel.
    """
    
    def __init__(self, 
                 fpca_model_file: str = "outputs/fpca_model.npz",
                 param_defs: Optional[List[Dict[str, Any]]] = None,
                 param_mapping: Optional[Dict[str, List[tuple]]] = None,
                 base_config_path: str = "cfgs/geballe_no_diamond.yaml",
                 suppress_print: bool = True):
        """
        Initialize the FPCA wrapper.
        
        Parameters:
        -----------
        fpca_model_file : str
            Path to the saved FPCA model file
        param_defs : List[Dict[str, Any]], optional
            Parameter definitions (if None, will be loaded from training data)
        param_mapping : Dict[str, List[tuple]], optional
            Parameter mapping to config structure (if None, will use default)
        base_config_path : str
            Path to the base configuration file
        suppress_print : bool
            Whether to suppress print output during simulations
        """
        print(f"Initializing HeatFlowFPCAWrapper...")
        
        # Load FPCA model
        self.fpca_model = load_fpca_model(fpca_model_file)
        print(f"Loaded FPCA model with {self.fpca_model['n_components']} components")
        
        # Set parameter definitions and mapping
        if param_defs is None:
            # Use default parameter definitions from create_initial_train_set.py
            self.param_defs = [
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
        else:
            self.param_defs = param_defs
            
        if param_mapping is None:
            # Use default parameter mapping from create_initial_train_set.py
            self.param_mapping = {
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
        else:
            self.param_mapping = param_mapping
            
        self.base_config_path = base_config_path
        self.suppress_print = suppress_print
        
        # Validate parameter definitions
        self._validate_parameters()
        
        print(f"Wrapper initialized with {len(self.param_defs)} parameters")
        print(f"Output dimension: {self.fpca_model['n_components']} FPCA components")
        
    def _validate_parameters(self):
        """Validate parameter definitions and mapping."""
        param_names = [p["name"] for p in self.param_defs]
        
        # Check that all parameters in mapping exist in definitions
        for param_name in self.param_mapping.keys():
            if param_name not in param_names:
                raise ValueError(f"Parameter '{param_name}' in mapping but not in definitions")
        
        # Check that all parameters in definitions have mappings
        for param_name in param_names:
            if param_name not in self.param_mapping:
                raise ValueError(f"Parameter '{param_name}' in definitions but not in mapping")
    
    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """
        Main interface for UQpy compatibility.
        
        Parameters:
        -----------
        samples : np.ndarray
            2D array where each row is a parameter sample
            Shape: (n_samples, n_parameters)
            
        Returns:
        --------
        np.ndarray
            2D array of FPCA coefficients
            Shape: (n_samples, n_fpca_components)
        """
        if samples.ndim == 1:
            # Single sample - convert to 2D
            samples = samples.reshape(1, -1)
        
        n_samples = samples.shape[0]
        n_params = samples.shape[1]
        
        if n_params != len(self.param_defs):
            raise ValueError(f"Expected {len(self.param_defs)} parameters, got {n_params}")
        
        print(f"Running {n_samples} simulations to get FPCA coefficients...")
        
        # Initialize results array
        fpca_coefficients = np.full((n_samples, self.fpca_model['n_components']), np.nan)
        
        # Process each sample
        for i, sample in enumerate(samples):
            try:
                # Run simulation
                result = run_single_simulation(
                    sample=sample,
                    param_defs=self.param_defs,
                    param_mapping=self.param_mapping,
                    simulation_index=i,
                    config_path=self.base_config_path,
                    suppress_print=self.suppress_print
                )
                
                # Check if simulation was successful
                if 'error' in result:
                    print(f"Simulation {i} failed: {result['error']}")
                    continue
                
                # Extract temperature curve
                if 'watcher_data' in result and 'oside' in result['watcher_data']:
                    curve = result['watcher_data']['oside']['normalized']
                    
                    # Project to FPCA space
                    fpca_scores = project_curve_to_fpca(curve, self.fpca_model)
                    fpca_coefficients[i, :] = fpca_scores
                    
                else:
                    print(f"Simulation {i} completed but no valid curve data")
                    
            except Exception as e:
                print(f"Simulation {i} failed with exception: {e}")
                continue
        
        # Count successful simulations
        successful = np.sum(~np.isnan(fpca_coefficients).any(axis=1))
        print(f"Successfully processed {successful}/{n_samples} simulations")
        
        return fpca_coefficients
    
    def run_single(self, sample: np.ndarray) -> np.ndarray:
        """
        Run a single simulation and return FPCA coefficients.
        
        Parameters:
        -----------
        sample : np.ndarray
            1D array of parameter values
            
        Returns:
        --------
        np.ndarray
            1D array of FPCA coefficients
        """
        result = self.__call__(sample.reshape(1, -1))
        return result[0]
    
    def get_output_dimension(self) -> int:
        """Get the dimension of the output (number of FPCA components)."""
        return self.fpca_model['n_components']
    
    def get_input_dimension(self) -> int:
        """Get the dimension of the input (number of parameters)."""
        return len(self.param_defs)
    
    def get_parameter_names(self) -> List[str]:
        """Get the names of the input parameters."""
        return [p["name"] for p in self.param_defs]
    
    def get_fpca_component_names(self) -> List[str]:
        """Get the names of the FPCA components."""
        return [f"PC{i+1}" for i in range(self.fpca_model['n_components'])]


def create_uqpy_runmodel(fpca_model_file: str = "outputs/fpca_model.npz",
                        **kwargs) -> HeatFlowFPCAWrapper:
    """
    Factory function to create a UQpy-compatible RunModel.
    
    Parameters:
    -----------
    fpca_model_file : str
        Path to the saved FPCA model file
    **kwargs : dict
        Additional arguments passed to HeatFlowFPCAWrapper
        
    Returns:
    --------
    HeatFlowFPCAWrapper
        UQpy-compatible wrapper instance
    """
    return HeatFlowFPCAWrapper(fpca_model_file=fpca_model_file, **kwargs)


# Example usage and testing functions
def test_wrapper_single():
    """Test the wrapper with a single parameter sample."""
    print("=" * 60)
    print("TESTING FPCA WRAPPER - SINGLE SAMPLE")
    print("=" * 60)
    
    # Create wrapper
    wrapper = create_uqpy_runmodel()
    
    # Create a test sample (using mean values from parameter definitions)
    test_sample = np.array([
        1.84e-6,    # d_sample
        2764828,    # rho_cv_sample
        3445520,    # rho_cv_coupler
        2764828,    # rho_cv_ins
        6.2e-8,     # d_coupler
        3.2e-6,     # d_ins_oside
        6.3e-6,     # d_ins_pside
        12e-6,      # fwhm
        3.8,        # k_sample
        10.0,       # k_ins
        350,        # k_coupler
    ])
    
    print(f"Test sample shape: {test_sample.shape}")
    print(f"Expected output dimension: {wrapper.get_output_dimension()}")
    
    # Run simulation
    start_time = time.time()
    fpca_coeffs = wrapper.run_single(test_sample)
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    print(f"FPCA coefficients: {fpca_coeffs}")
    
    return wrapper, test_sample, fpca_coeffs


def test_wrapper_batch():
    """Test the wrapper with multiple parameter samples."""
    print("=" * 60)
    print("TESTING FPCA WRAPPER - BATCH SAMPLES")
    print("=" * 60)
    
    # Create wrapper
    wrapper = create_uqpy_runmodel()
    
    # Create test samples (small batch)
    n_samples = 3
    test_samples = np.array([
        [1.84e-6, 2764828, 3445520, 2764828, 6.2e-8, 3.2e-6, 6.3e-6, 12e-6, 3.8, 10.0, 350],  # Sample 1
        [1.85e-6, 2765000, 3446000, 2765000, 6.3e-8, 3.3e-6, 6.4e-6, 12.1e-6, 3.9, 10.5, 360], # Sample 2
        [1.83e-6, 2764600, 3445000, 2764600, 6.1e-8, 3.1e-6, 6.2e-6, 11.9e-6, 3.7, 9.5, 340],  # Sample 3
    ])
    
    print(f"Test samples shape: {test_samples.shape}")
    print(f"Expected output shape: ({n_samples}, {wrapper.get_output_dimension()})")
    
    # Run simulations
    start_time = time.time()
    fpca_coeffs = wrapper(test_samples)
    end_time = time.time()
    
    print(f"Batch simulation completed in {end_time - start_time:.2f} seconds")
    print(f"FPCA coefficients shape: {fpca_coeffs.shape}")
    print(f"FPCA coefficients:\n{fpca_coeffs}")
    
    return wrapper, test_samples, fpca_coeffs


def main():
    """Run all tests."""
    print("Starting UQpy FPCA wrapper tests...")
    
    # Test single sample
    wrapper, test_sample, single_coeffs = test_wrapper_single()
    
    # Test batch samples
    wrapper, test_samples, batch_coeffs = test_wrapper_batch()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nWrapper is ready for use with UQpy RunModel!")
    print(f"Input dimension: {wrapper.get_input_dimension()}")
    print(f"Output dimension: {wrapper.get_output_dimension()}")
    print(f"Parameter names: {wrapper.get_parameter_names()}")
    print(f"FPCA component names: {wrapper.get_fpca_component_names()}")


if __name__ == "__main__":
    main() 