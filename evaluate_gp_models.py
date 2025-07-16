#!/usr/bin/env python3
"""
Test script to evaluate GP performance using existing training data.
Uses train/test split on the 200 existing datapoints to assess GP quality.
"""

import sys
import sys

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from analysis.uq_wrapper import load_recast_training_data
import warnings

import sys

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """
    Load the recast training data and prepare for GP modeling.
    """
    print("Loading recast training data...")
    
    # Load the FPCA-transformed data
    recast_data = load_recast_training_data("outputs/training_data_fpca.npz")
    
    # Extract parameters and FPCA scores
    X = recast_data['parameters']  # Input parameters
    y = recast_data['fpca_scores']  # FPCA coefficients
    
    print(f"Data loaded:")
    print(f"  Parameters shape: {X.shape}")
    print(f"  FPCA scores shape: {y.shape}")
    print(f"  Number of samples: {len(X)}")
    print(f"  Number of parameters: {X.shape[1]}")
    print(f"  Number of FPCA components: {y.shape[1]}")
    
    return X, y, recast_data['parameter_names']

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Create train/test split of the data.
    """
    print(f"\nCreating train/test split (test_size={test_size})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """
    Scale the input parameters using StandardScaler.
    """
    print("\nScaling input parameters...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Scaler fitted on training data")
    print(f"  Mean: {scaler.mean_}")
    print(f"  Scale: {scaler.scale_}")
    
    return X_train_scaled, X_test_scaled, scaler

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

def train_and_evaluate_gp(X_train, X_test, y_train, y_test, component_idx=0, kernel_type='rbf'):
    """
    Train a GP model on a specific FPCA component and evaluate performance.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING GP FOR FPCA COMPONENT {component_idx + 1}")
    print(f"{'='*60}")
    
    # Extract the target component
    y_train_component = y_train[:, component_idx]
    y_test_component = y_test[:, component_idx]
    
    print(f"Target: FPCA Component {component_idx + 1}")
    print(f"Training data range: [{y_train_component.min():.4f}, {y_train_component.max():.4f}]")
    print(f"Test data range: [{y_test_component.min():.4f}, {y_test_component.max():.4f}]")
    
    # Create and train GP model
    print(f"\nTraining GP with {kernel_type} kernel...")
    gp = create_gp_model(kernel_type)
    
    # Train the model
    gp.fit(X_train, y_train_component)
    
    # Print kernel information
    print(f"Optimized kernel: {gp.kernel_}")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_train, std_train = gp.predict(X_train, return_std=True)
    y_pred_test, std_test = gp.predict(X_test, return_std=True)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train_component, y_pred_train)
    test_mse = mean_squared_error(y_test_component, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_train_component, y_pred_train)
    test_r2 = r2_score(y_test_component, y_pred_test)
    
    print(f"\nPerformance Metrics:")
    print(f"  Training RMSE: {train_rmse:.6f}")
    print(f"  Test RMSE: {test_rmse:.6f}")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    
    return {
        'gp': gp,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'std_train': std_train,
        'std_test': std_test,
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    }

def plot_gp_results(y_train, y_test, results, component_idx=0):
    """
    Create plots to visualize GP performance.
    """
    print(f"\nCreating visualization plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    y_train_component = y_train[:, component_idx]
    y_test_component = y_test[:, component_idx]
    y_pred_train = results['y_pred_train']
    y_pred_test = results['y_pred_test']
    std_train = results['std_train']
    std_test = results['std_test']
    
    # Plot 1: Training data predictions vs actual
    axes[0, 0].scatter(y_train_component, y_pred_train, alpha=0.6, label='Training')
    axes[0, 0].plot([y_train_component.min(), y_train_component.max()], 
                    [y_train_component.min(), y_train_component.max()], 'r--', label='Perfect')
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Training: Predicted vs Actual')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Test data predictions vs actual
    axes[0, 1].scatter(y_test_component, y_pred_test, alpha=0.6, label='Test')
    axes[0, 1].plot([y_test_component.min(), y_test_component.max()], 
                    [y_test_component.min(), y_test_component.max()], 'r--', label='Perfect')
    axes[0, 1].set_xlabel('Actual Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].set_title('Test: Predicted vs Actual')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Prediction uncertainty (training)
    axes[1, 0].scatter(y_train_component, std_train, alpha=0.6)
    axes[1, 0].set_xlabel('Actual Values')
    axes[1, 0].set_ylabel('Prediction Standard Deviation')
    axes[1, 0].set_title('Training: Prediction Uncertainty')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Prediction uncertainty (test)
    axes[1, 1].scatter(y_test_component, std_test, alpha=0.6)
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Prediction Standard Deviation')
    axes[1, 1].set_title('Test: Prediction Uncertainty')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'outputs/gp_component_{component_idx+1}_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_parameter_sensitivity(gp, parameter_names, scaler):
    """
    Analyze parameter sensitivity using the GP model.
    """
    print(f"\n{'='*60}")
    print("PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    
    # Get the optimized kernel
    kernel = gp.kernel_
    
    # Extract length scales (inverse of sensitivity)
    if hasattr(kernel, 'k2') and hasattr(kernel.k2, 'length_scale'):
        length_scales = kernel.k2.length_scale
    else:
        print("Could not extract length scales from kernel")
        return
    
    # Calculate sensitivity (1/length_scale)
    sensitivities = 1.0 / length_scales
    
    # Create sensitivity plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort parameters by sensitivity
    sorted_indices = np.argsort(sensitivities)[::-1]
    sorted_sensitivities = sensitivities[sorted_indices]
    sorted_names = [parameter_names[i] for i in sorted_indices]
    
    bars = ax.bar(range(len(sorted_sensitivities)), sorted_sensitivities)
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Sensitivity (1/length_scale)')
    ax.set_title('Parameter Sensitivity from GP Length Scales')
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, sensitivity) in enumerate(zip(bars, sorted_sensitivities)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{sensitivity:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('outputs/gp_parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print sensitivity ranking
    print("\nParameter Sensitivity Ranking (highest to lowest):")
    for i, (name, sensitivity) in enumerate(zip(sorted_names, sorted_sensitivities)):
        print(f"  {i+1:2d}. {name:15s}: {sensitivity:.4f}")

def compare_kernels(X_train, X_test, y_train, y_test, component_idx=0):
    """
    Compare different kernel types for GP performance.
    """
    print(f"\n{'='*60}")
    print("KERNEL COMPARISON")
    print(f"{'='*60}")
    
    kernels = ['rbf', 'matern']
    results = {}
    
    for kernel_type in kernels:
        print(f"\nTesting {kernel_type.upper()} kernel...")
        result = train_and_evaluate_gp(X_train, X_test, y_train, y_test, 
                                     component_idx, kernel_type)
        results[kernel_type] = result['metrics']
    
    # Compare results
    print(f"\n{'='*40}")
    print("KERNEL COMPARISON SUMMARY")
    print(f"{'='*40}")
    print(f"{'Kernel':<10} {'Train R²':<10} {'Test R²':<10} {'Train RMSE':<12} {'Test RMSE':<12}")
    print(f"{'-'*60}")
    
    for kernel_type, metrics in results.items():
        print(f"{kernel_type.upper():<10} {metrics['train_r2']:<10.4f} "
              f"{metrics['test_r2']:<10.4f} {metrics['train_rmse']:<12.6f} "
              f"{metrics['test_rmse']:<12.6f}")
    
    return results

def main():
    """
    Main function to run all GP tests.
    """
    print("Starting GP evaluation with existing training data...")
    
    # Load and prepare data
    X, y, parameter_names = load_and_prepare_data()
    
    # Create train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    
    # Scale the data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    # Test GP on first FPCA component
    component_idx = 0  # First FPCA component
    results = train_and_evaluate_gp(X_train_scaled, X_test_scaled, y_train, y_test, component_idx)
    
    # Create visualization plots
    plot_gp_results(y_train, y_test, results, component_idx)
    
    # Analyze parameter sensitivity
    analyze_parameter_sensitivity(results['gp'], parameter_names, scaler)
    
    # Compare different kernels
    kernel_comparison = compare_kernels(X_train_scaled, X_test_scaled, y_train, y_test, component_idx)
    
    # Summary
    print(f"\n{'='*60}")
    print("GP EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {len(X)} total samples, {len(X_train)} training, {len(X_test)} test")
    print(f"Input dimension: {X.shape[1]} parameters")
    print(f"Output dimension: {y.shape[1]} FPCA components")
    print(f"Tested component: PC{component_idx + 1}")
    
    best_kernel = max(kernel_comparison.items(), key=lambda x: x[1]['test_r2'])
    print(f"Best kernel: {best_kernel[0].upper()} (Test R² = {best_kernel[1]['test_r2']:.4f})")
    
    print(f"\nFiles created:")
    print(f"- outputs/gp_component_{component_idx+1}_results.png: GP performance plots")
    print(f"- outputs/gp_parameter_sensitivity.png: Parameter sensitivity analysis")
    
    print(f"\nAssessment:")
    if best_kernel[1]['test_r2'] > 0.8:
        print("✅ GP quality: GOOD - High R² suggests good predictive performance")
    elif best_kernel[1]['test_r2'] > 0.6:
        print("⚠️  GP quality: FAIR - Moderate R², may benefit from more data")
    else:
        print("❌ GP quality: POOR - Low R² suggests need for more training data")
    
    return results, kernel_comparison

if __name__ == "__main__":
    results, kernel_comparison = main() 