#!/usr/bin/env python3
"""
Example workflow script showing the typical usage of v2-heatflow.
This script demonstrates the complete pipeline from data generation to UQ analysis.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
    else:
        print("❌ Failed!")
        print("Error:", result.stderr)
        return False
    
    return True

def main():
    """Run the complete workflow."""
    print("V2-HeatFlow Complete Workflow Example")
    print("This script demonstrates the typical usage pipeline.")
    
    # Step 1: Generate training data
    if not run_command(
        "python scripts/generate_training_data.py",
        "Generating training dataset"
    ):
        return
    
    # Step 2: Analyze FPCA decomposition
    if not run_command(
        "python scripts/analyze_fpca.py",
        "Performing FPCA analysis"
    ):
        return
    
    # Step 3: Train surrogate models
    if not run_command(
        "python scripts/train_surrogate_models.py",
        "Training surrogate models"
    ):
        return
    
    # Step 4: Evaluate GP models
    if not run_command(
        "python scripts/evaluate_gp_models.py",
        "Evaluating GP model performance"
    ):
        return
    
    # Step 5: Validate surrogates
    if not run_command(
        "python scripts/validate_surrogates.py",
        "Validating surrogate models"
    ):
        return
    
    print("\n🎉 Complete workflow finished successfully!")
    print("Check the outputs/ directory for results.")

if __name__ == "__main__":
    main()
