#!/usr/bin/env python3
"""
Create a full surrogate GP model using all training data.
Fits a GP to each FPCA coefficient and tests predictions within parameter ranges.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from uq_wrapper import load_recast_training_data, load_fpca_model, reconstruct_curve_from_fpca
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FullSurrogateModel:
    """
    Full surrogate model that combines FPCA and GP for each component.
    """
    
    def __init__(self, fpca_model, gps, scaler, parameter_names, param_ranges):
        """
        Initialize the surrogate model.
        
        Parameters:
        -----------
        fpca_model : dict
            The FPCA model
        gps : list
            List of trained GP models (one per FPCA component)
        scaler : StandardScaler
            Fitted scaler for input parameters
        parameter_names : list
            Names of the parameters
        param_ranges : dict
            Dictionary of parameter ranges for validation
        """
        self.fpca_model = fpca_model
        self.gps = gps
        self.scaler = scaler
        self.parameter_names = parameter_names
        self.param_ranges = param_ranges
        self.n_components = fpca_model['n_components']
        self.n_parameters = len(parameter_names)
        
    def predict_fpca_coefficients(self, X):
        """
        Predict FPCA coefficients for given parameter values.
        
        Parameters:
        -----------
        X : np.ndarray
            Parameter values (n_samples, n_parameters)
            
        Returns:
        --------
        np.ndarray
            Predicted FPCA coefficients (n_samples, n_components)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Scale inputs
        X_scaled = self.scaler.transform(X)
        
        # Predict each component
        predictions = []
        uncertainties = []
        
        for i, gp in enumerate(self.gps):
            pred, std = gp.predict(X_scaled, return_std=True)
            predictions.append(pred)
            uncertainties.append(std)
        
        return np.column_stack(predictions), np.column_stack(uncertainties)
    
    def predict_temperature_curves(self, X):
        """
        Predict full temperature curves for given parameter values.
        
        Parameters:
        -----------
        X : np.ndarray
            Parameter values (n_samples, n_parameters)
            
        Returns:
        --------
        np.ndarray
            Predicted temperature curves (n_samples, n_timepoints)
        """
        # Get FPCA coefficients
        fpca_coeffs, fpca_uncertainties = self.predict_fpca_coefficients(X)
        
        # Reconstruct curves
        curves = []
        for coeffs in fpca_coeffs:
            curve = reconstruct_curve_from_fpca(coeffs, self.fpca_model)
            curves.append(curve)
        
        return np.array(curves), fpca_coeffs, fpca_uncertainties
    
    def validate_parameters(self, X):
        """
        Validate that parameters are within defined ranges.
        
        Parameters:
        -----------
        X : np.ndarray
            Parameter values to validate
            
        Returns:
        --------
        bool
            True if all parameters are within ranges
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        for i, (param_name, (min_val, max_val)) in enumerate(self.param_ranges.items()):
            if np.any(X[:, i] < min_val) or np.any(X[:, i] > max_val):
                return False
        return True
    
    def save_model(self, filepath):
        """
        Save the surrogate model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        model_data = {
            'fpca_model': self.fpca_model,
            'gps': self.gps,
            'scaler': self.scaler,
            'parameter_names': self.parameter_names,
            'param_ranges': self.param_ranges,
            'n_components': self.n_components,
            'n_parameters': self.n_parameters
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Surrogate model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a surrogate model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        FullSurrogateModel
            Loaded surrogate model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        return cls(
            fpca_model=model_data['fpca_model'],
            gps=model_data['gps'],
            scaler=model_data['scaler'],
            parameter_names=model_data['parameter_names'],
            param_ranges=model_data['param_ranges']
        )

def get_parameter_ranges():
    """
    Get the parameter ranges from the original definitions.
    """
    param_ranges = {
        "d_sample": (1.84e-6 * 0.8, 1.84e-6 * 1.2),  # ±20% around center
        "rho_cv_sample": (2764828 * 0.8, 2764828 * 1.2),
        "rho_cv_coupler": (3445520 * 0.8, 3445520 * 1.2),
        "rho_cv_ins": (2764828 * 0.8, 2764828 * 1.2),
        "d_coupler": (6.2e-8 * 0.8, 6.2e-8 * 1.2),
        "d_ins_oside": (3.2e-6 * 0.8, 3.2e-6 * 1.2),
        "d_ins_pside": (6.3e-6 * 0.8, 6.3e-6 * 1.2),
        "fwhm": (12e-6 * 0.8, 12e-6 * 1.2),
        "k_sample": (2.8, 4.8),  # Uniform range
        "k_ins": (7.0, 13.0),    # Uniform range
        "k_coupler": (300, 400), # Uniform range
    }
    return param_ranges

def create_gp_model(kernel_type='rbf'):
    """
    Create a Gaussian Process model with specified kernel.
    """
    if kernel_type == 'rbf':
        kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(11))
    elif kernel_type == 'matern':
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(11), nu=1.5)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,  # Small noise for numerical stability
        n_restarts_optimizer=10,
        random_state=42
    )
    
    return gp

def train_surrogate_model():
    """
    Train the full surrogate model using all training data.
    """
    print("=" * 60)
    print("TRAINING FULL SURROGATE MODEL")
    print("=" * 60)
    
    # Load training data
    print("\n1. Loading training data...")
    recast_data = load_recast_training_data("outputs/training_data_fpca.npz")
    X = recast_data['parameters']
    y = recast_data['fpca_scores']
    parameter_names = recast_data['parameter_names']
    
    print(f"Training data: {len(X)} samples, {X.shape[1]} parameters, {y.shape[1]} FPCA components")
    
    # Load FPCA model
    print("\n2. Loading FPCA model...")
    fpca_model = load_fpca_model("outputs/fpca_model.npz")
    
    # Scale the data
    print("\n3. Scaling input parameters...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Get parameter ranges
    param_ranges = get_parameter_ranges()
    
    # Train GP for each FPCA component
    print("\n4. Training GP models for each FPCA component...")
    gps = []
    training_metrics = []
    
    for i in range(fpca_model['n_components']):
        print(f"\nTraining GP for PC{i+1}...")
        
        # Create and train GP
        gp = create_gp_model('rbf')  # Use RBF kernel
        gp.fit(X_scaled, y[:, i])
        
        # Evaluate training performance
        y_pred, y_std = gp.predict(X_scaled, return_std=True)
        train_r2 = r2_score(y[:, i], y_pred)
        train_rmse = np.sqrt(mean_squared_error(y[:, i], y_pred))
        
        print(f"  Training R²: {train_r2:.6f}")
        print(f"  Training RMSE: {train_rmse:.6f}")
        print(f"  Optimized kernel: {gp.kernel_}")
        
        gps.append(gp)
        training_metrics.append({
            'r2': train_r2,
            'rmse': train_rmse,
            'kernel': gp.kernel_
        })
    
    # Create surrogate model
    print("\n5. Creating surrogate model...")
    surrogate = FullSurrogateModel(
        fpca_model=fpca_model,
        gps=gps,
        scaler=scaler,
        parameter_names=parameter_names,
        param_ranges=param_ranges
    )
    
    # Save the model
    print("\n6. Saving surrogate model...")
    surrogate.save_model("outputs/full_surrogate_model.pkl")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SURROGATE MODEL TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Training samples: {len(X)}")
    print(f"Input parameters: {X.shape[1]}")
    print(f"FPCA components: {y.shape[1]}")
    print(f"Parameter ranges defined: {len(param_ranges)}")
    
    print(f"\nTraining Performance:")
    for i, metrics in enumerate(training_metrics):
        print(f"  PC{i+1}: R² = {metrics['r2']:.6f}, RMSE = {metrics['rmse']:.6f}")
    
    return surrogate, training_metrics

def generate_test_samples(param_ranges, n_samples=50):
    """
    Generate test samples within the parameter ranges.
    """
    print(f"\nGenerating {n_samples} test samples within parameter ranges...")
    
    # Use parameter names directly
    param_names = list(param_ranges.keys())
    low = [param_ranges[name][0] for name in param_names]
    high = [param_ranges[name][1] for name in param_names]
    
    # Generate uniform random samples
    test_samples = np.random.uniform(
        low=low,
        high=high,
        size=(n_samples, len(param_names))
    )
    
    print(f"Test samples shape: {test_samples.shape}")
    print(f"Parameter ranges covered:")
    for i, name in enumerate(param_names):
        min_val, max_val = param_ranges[name]
        actual_min = test_samples[:, i].min()
        actual_max = test_samples[:, i].max()
        print(f"  {name}: [{actual_min:.2e}, {actual_max:.2e}] (target: [{min_val:.2e}, {max_val:.2e}])")
    
    return test_samples

def test_surrogate_model(surrogate, test_samples):
    """
    Test the surrogate model with new parameter samples.
    """
    print(f"\n{'='*60}")
    print("TESTING SURROGATE MODEL")
    print(f"{'='*60}")
    
    # Validate parameters
    print("\n1. Validating test parameters...")
    valid_params = surrogate.validate_parameters(test_samples)
    print(f"All parameters within ranges: {valid_params}")
    
    # Predict FPCA coefficients
    print("\n2. Predicting FPCA coefficients...")
    fpca_coeffs, fpca_uncertainties = surrogate.predict_fpca_coefficients(test_samples)
    
    print(f"Predicted FPCA coefficients shape: {fpca_coeffs.shape}")
    print(f"FPCA uncertainties shape: {fpca_uncertainties.shape}")
    
    # Predict full temperature curves
    print("\n3. Predicting temperature curves...")
    curves, coeffs, uncertainties = surrogate.predict_temperature_curves(test_samples)
    
    print(f"Predicted curves shape: {curves.shape}")
    
    # Analyze predictions
    print("\n4. Analyzing predictions...")
    
    # FPCA coefficient statistics
    print(f"\nFPCA Coefficient Statistics:")
    for i in range(fpca_coeffs.shape[1]):
        coeff_mean = np.mean(fpca_coeffs[:, i])
        coeff_std = np.std(fpca_coeffs[:, i])
        uncertainty_mean = np.mean(fpca_uncertainties[:, i])
        print(f"  PC{i+1}: mean={coeff_mean:.4f}, std={coeff_std:.4f}, avg_uncertainty={uncertainty_mean:.4f}")
    
    # Temperature curve statistics
    print(f"\nTemperature Curve Statistics:")
    curve_max = np.max(curves, axis=1)
    curve_min = np.min(curves, axis=1)
    curve_range = curve_max - curve_min
    print(f"  Max temperature: mean={np.mean(curve_max):.4f}, std={np.std(curve_max):.4f}")
    print(f"  Min temperature: mean={np.mean(curve_min):.4f}, std={np.std(curve_min):.4f}")
    print(f"  Temperature range: mean={np.mean(curve_range):.4f}, std={np.std(curve_range):.4f}")
    
    return {
        'fpca_coeffs': fpca_coeffs,
        'fpca_uncertainties': fpca_uncertainties,
        'curves': curves,
        'test_samples': test_samples
    }

def visualize_surrogate_results(surrogate, test_results):
    """
    Create visualizations of the surrogate model results.
    """
    print(f"\n{'='*60}")
    print("CREATING SURROGATE MODEL VISUALIZATIONS")
    print(f"{'='*60}")
    
    fpca_coeffs = test_results['fpca_coeffs']
    fpca_uncertainties = test_results['fpca_uncertainties']
    curves = test_results['curves']
    test_samples = test_results['test_samples']
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: FPCA coefficient distributions
    for i in range(fpca_coeffs.shape[1]):
        axes[0, 0].hist(fpca_coeffs[:, i], bins=20, alpha=0.7, label=f'PC{i+1}')
    axes[0, 0].set_xlabel('FPCA Coefficient Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Predicted FPCA Coefficients')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: FPCA coefficient uncertainties
    for i in range(fpca_uncertainties.shape[1]):
        axes[0, 1].hist(fpca_uncertainties[:, i], bins=20, alpha=0.7, label=f'PC{i+1}')
    axes[0, 1].set_xlabel('Prediction Uncertainty')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Prediction Uncertainties')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: FPCA coefficient scatter (PC1 vs PC2)
    if fpca_coeffs.shape[1] >= 2:
        scatter = axes[0, 2].scatter(fpca_coeffs[:, 0], fpca_coeffs[:, 1], 
                                   c=fpca_uncertainties[:, 0], cmap='viridis', alpha=0.7)
        axes[0, 2].set_xlabel('PC1 Coefficient')
        axes[0, 2].set_ylabel('PC2 Coefficient')
        axes[0, 2].set_title('PC1 vs PC2 (colored by uncertainty)')
        plt.colorbar(scatter, ax=axes[0, 2], label='PC1 Uncertainty')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Sample temperature curves
    n_curves_to_plot = min(20, len(curves))
    time_points = np.arange(curves.shape[1])
    for i in range(n_curves_to_plot):
        axes[1, 0].plot(time_points, curves[i], alpha=0.5, linewidth=1)
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Normalized Temperature')
    axes[1, 0].set_title(f'Sample Predicted Temperature Curves ({n_curves_to_plot} curves)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Mean temperature curve with uncertainty
    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)
    axes[1, 1].plot(time_points, mean_curve, 'b-', linewidth=2, label='Mean')
    axes[1, 1].fill_between(time_points, mean_curve - std_curve, mean_curve + std_curve, 
                           alpha=0.3, label='±1σ')
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Normalized Temperature')
    axes[1, 1].set_title('Mean Predicted Curve with Standard Deviation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Parameter sensitivity (using GP length scales)
    if hasattr(surrogate.gps[0].kernel_, 'k2') and hasattr(surrogate.gps[0].kernel_.k2, 'length_scale'):
        length_scales = surrogate.gps[0].kernel_.k2.length_scale
        sensitivities = 1.0 / length_scales
        sorted_indices = np.argsort(sensitivities)[::-1]
        sorted_sensitivities = sensitivities[sorted_indices]
        sorted_names = [surrogate.parameter_names[i] for i in sorted_indices]
        
        bars = axes[1, 2].bar(range(len(sorted_sensitivities)), sorted_sensitivities)
        axes[1, 2].set_xlabel('Parameters')
        axes[1, 2].set_ylabel('Sensitivity (1/length_scale)')
        axes[1, 2].set_title('Parameter Sensitivity (from GP length scales)')
        axes[1, 2].set_xticks(range(len(sorted_names)))
        axes[1, 2].set_xticklabels(sorted_names, rotation=45, ha='right')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/surrogate_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Surrogate model visualizations saved to outputs/surrogate_model_results.png")

def main():
    """
    Main function to create and test the full surrogate model.
    """
    print("Creating full surrogate GP model...")
    
    # Train the surrogate model
    surrogate, training_metrics = train_surrogate_model()
    
    # Generate test samples
    param_ranges = get_parameter_ranges()
    test_samples = generate_test_samples(param_ranges, n_samples=100)
    
    # Test the surrogate model
    test_results = test_surrogate_model(surrogate, test_samples)
    
    # Create visualizations
    visualize_surrogate_results(surrogate, test_results)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FULL SURROGATE MODEL COMPLETED!")
    print(f"{'='*60}")
    print(f"Model saved to: outputs/full_surrogate_model.pkl")
    print(f"Test samples: {len(test_samples)}")
    print(f"All training metrics R² > 0.99: {all(m['r2'] > 0.99 for m in training_metrics)}")
    print(f"Model ready for use in UQ analysis!")
    
    return surrogate, test_results

if __name__ == "__main__":
    surrogate, test_results = main() 