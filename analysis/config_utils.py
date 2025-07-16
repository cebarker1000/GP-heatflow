"""
Utility functions for reading and parsing configuration files.
"""

import yaml
from typing import Dict, List, Any, Tuple
import numpy as np
from UQpy.distributions.collection import Uniform, Normal, Lognormal


def load_distributions_config(config_path: str = "configs/distributions.yaml") -> Dict[str, Any]:
    """
    Load the distributions configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the loaded configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_param_defs_from_config(config_path: str = "configs/distributions.yaml") -> List[Dict[str, Any]]:
    """
    Extract parameter definitions from config file in the format expected by existing code.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        List of parameter definition dictionaries
    """
    config = load_distributions_config(config_path)
    param_defs = []
    
    for param_name, param_config in config['parameters'].items():
        param_def = {
            "name": param_name,
            "type": param_config["type"]
        }
        
        # Add type-specific parameters
        if param_config["type"] == "lognormal":
            param_def["center"] = param_config["center"]
            param_def["sigma_log"] = param_config["sigma_log"]
        elif param_config["type"] == "normal":
            param_def["center"] = param_config["center"]
            param_def["sigma"] = param_config["sigma"]
        elif param_config["type"] == "uniform":
            param_def["low"] = param_config["low"]
            param_def["high"] = param_config["high"]
        
        param_defs.append(param_def)
    
    return param_defs


def get_param_mapping_from_config(config_path: str = "configs/distributions.yaml") -> Dict[str, List[Tuple]]:
    """
    Extract parameter mapping from config file in the format expected by existing code.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary mapping parameter names to lists of config paths
    """
    config = load_distributions_config(config_path)
    param_mapping = {}
    
    for param_name, mappings in config['parameter_mapping'].items():
        # Convert list of lists to list of tuples
        param_mapping[param_name] = [tuple(mapping) for mapping in mappings]
    
    return param_mapping


def get_sampling_config(config_path: str = "configs/distributions.yaml") -> Dict[str, Any]:
    """
    Extract sampling configuration from config file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing sampling configuration
    """
    config = load_distributions_config(config_path)
    return config.get('sampling', {})


def get_output_config(config_path: str = "configs/distributions.yaml") -> Dict[str, Any]:
    """
    Extract output configuration from config file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing output configuration
    """
    config = load_distributions_config(config_path)
    return config.get('output', {})


def create_uqpy_distributions(param_defs: List[Dict[str, Any]]) -> List:
    """
    Create UQpy distribution objects from parameter definitions.
    
    Args:
        param_defs: List of parameter definition dictionaries
        
    Returns:
        List of UQpy distribution objects
    """
    distributions = []
    for p in param_defs:
        if p["type"] == "lognormal":
            sigma = p["sigma_log"]    
            center = p["center"] 
            # For lognormal, s=sigma_log (shape), scale=center (geometric mean)
            distributions.append(Lognormal(s=sigma, scale=center))
        elif p["type"] == "normal":
            distributions.append(Normal(loc=p["center"], scale=p["sigma"]))
        elif p["type"] == "uniform":
            distributions.append(Uniform(loc=p["low"], scale=p["high"] - p["low"]))
        else:
            raise ValueError(f"Unknown type: {p['type']}")
    
    return distributions


def get_fixed_params_from_config(config_path: str = "configs/distributions.yaml") -> np.ndarray:
    """
    Get the fixed parameter values (excluding k values) from config file.
    These are the centers of the distributions for the nuisance parameters.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Array of fixed parameter values in the order: [d_sample, rho_cv_sample, rho_cv_coupler, rho_cv_ins, d_coupler, d_ins_pside, d_ins_oside, fwhm]
    """
    config = load_distributions_config(config_path)
    
    # Define the order of fixed parameters (excluding k values)
    fixed_param_names = ['d_sample', 'rho_cv_sample', 'rho_cv_coupler', 'rho_cv_ins', 'd_coupler', 'd_ins_pside', 'd_ins_oside', 'fwhm']
    
    fixed_params = []
    for param_name in fixed_param_names:
        if param_name in config['parameters']:
            param_config = config['parameters'][param_name]
            if param_config['type'] == 'lognormal':
                fixed_params.append(param_config['center'])
            else:
                # For uniform distributions, use the midpoint
                fixed_params.append((param_config['low'] + param_config['high']) / 2)
        else:
            raise ValueError(f"Parameter {param_name} not found in config file")
    
    return np.array(fixed_params)


def load_all_from_config(config_path: str = "configs/distributions.yaml") -> Tuple[List[Dict[str, Any]], Dict[str, List[Tuple]], Dict[str, Any], Dict[str, Any]]:
    """
    Load all configuration data from the YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Tuple of (param_defs, param_mapping, sampling_config, output_config)
    """
    param_defs = get_param_defs_from_config(config_path)
    param_mapping = get_param_mapping_from_config(config_path)
    sampling_config = get_sampling_config(config_path)
    output_config = get_output_config(config_path)
    
    return param_defs, param_mapping, sampling_config, output_config 