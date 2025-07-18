#!/usr/bin/env python3
"""
Wrapper script to run FPCA analysis on Edmund training data.

This script demonstrates how to use the modified analyze_fpca.py
with Edmund-specific data files.
"""

import subprocess
import sys
import os

def main():
    """Run FPCA analysis on Edmund training data."""
    
    print("Running FPCA analysis on Edmund training data...")
    print("=" * 60)
    
    # Check if the Edmund data file exists
    edmund_data_file = "outputs/edmund1/uq_batch_results.npz"
    if not os.path.exists(edmund_data_file):
        print(f"Error: Edmund data file not found at {edmund_data_file}")
        print("Please run generate_edmund_training_data.py first to create the training data.")
        sys.exit(1)
    
    # Define the command with Edmund-specific arguments
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "analyze_fpca.py",
        "--input", edmund_data_file,
        "--output-dir", "outputs/edmund1",
        "--components", "6"  # Analyze 6 components for Edmund data
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n" + "="*60)
        print("FPCA analysis completed successfully!")
        print("="*60)
        
        # Print summary of what was created
        output_dir = "outputs/edmund1"
        if os.path.exists(output_dir):
            print(f"\nFPCA analysis files created in {output_dir}:")
            fpca_files = [
                'fpca_eigenfunctions.png',
                'fpca_reconstruction_examples.png', 
                'fpca_reconstruction_errors.png',
                'fpca_scores.png',
                'fpca_report.txt'
            ]
            for file in fpca_files:
                file_path = os.path.join(output_dir, file)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  - {file} ({size:,} bytes)")
                else:
                    print(f"  - {file} (not found)")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError running FPCA analysis: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nFPCA analysis interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main() 