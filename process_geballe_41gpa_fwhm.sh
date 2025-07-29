#!/bin/bash

# Script to generate surrogate models and run MCMC for all fwhm folders in geballe 41 gpa
# This script will:
# 1. Find all fwhm folders in outputs/geballe/41Gpa/
# 2. Train a surrogate model for each folder
# 3. Run MCMC analysis for each folder

set -e  # Exit on any error

# Configuration
BASE_DIR="outputs/geballe/41Gpa"
EXPERIMENTAL_DATA_DIR="data/experimental/geballe"
CONFIG_DIR="configs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a directory exists
check_directory() {
    if [ ! -d "$1" ]; then
        print_error "Directory $1 does not exist!"
        exit 1
    fi
}

# Function to check if required files exist
check_required_files() {
    local folder=$1
    local required_files=("training_data_fpca.npz" "fpca_model.npz" "distributions.yaml" "config_5_materials.yaml")
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$folder/$file" ]; then
            print_error "Required file $folder/$file not found!"
            return 1
        fi
    done
    return 0
}

# Function to train surrogate model
train_surrogate() {
    local folder=$1
    local run_name=$(basename "$folder")
    
    print_status "Training surrogate model for $run_name..."
    
    local input_path="$folder/training_data_fpca.npz"
    local fpca_model_path="$folder/fpca_model.npz"
    local output_path="$folder/full_surrogate_model.pkl"
    local training_config="$folder/config_5_materials.yaml"
    
    python train_surrogate_models.py \
        --input_path "$input_path" \
        --fpca_model_path "$fpca_model_path" \
        --output_path "$output_path" \
        --training_config "$training_config" \
        --test_fraction 0.2 \
        --random_state 42
    
    if [ $? -eq 0 ]; then
        print_success "Surrogate model trained successfully for $run_name"
    else
        print_error "Failed to train surrogate model for $run_name"
        return 1
    fi
}

# Function to run MCMC analysis
run_mcmc() {
    local folder=$1
    local run_name=$(basename "$folder")
    
    print_status "Running MCMC analysis for $run_name..."
    
    # Determine experimental data file based on run name
    local exp_data_file=""
    case $run_name in
        "run1_fwhm")
            exp_data_file="$EXPERIMENTAL_DATA_DIR/geballe_41Gpa_1.csv"
            ;;
        "run2_fwhm")
            exp_data_file="$EXPERIMENTAL_DATA_DIR/geballe_41Gpa_2.csv"
            ;;
        "run3_fwhm")
            exp_data_file="$EXPERIMENTAL_DATA_DIR/geballe_41Gpa_3.csv"
            ;;
        *)
            print_error "Unknown run name: $run_name"
            return 1
            ;;
    esac
    
    # Check if experimental data file exists
    if [ ! -f "$exp_data_file" ]; then
        print_error "Experimental data file $exp_data_file not found!"
        return 1
    fi
    
    local config_path="$folder/distributions.yaml"
    local surrogate_path="$folder/full_surrogate_model.pkl"
    local sim_cfg="$folder/config_5_materials.yaml"
    local output_path="$folder/mcmc_results.npz"
    local plot_prefix="$folder/mcmc"
    
    python uqpy_MCMC.py \
        --config_path "$config_path" \
        --surrogate_path "$surrogate_path" \
        --exp_data_path "$exp_data_file" \
        --sim_cfg "$sim_cfg" \
        --output_path "$output_path" \
        --plot_path_prefix "$plot_prefix" \
        --n_walkers 60 \
        --n_samples 1000000 \
        --burn_length 20000
    
    if [ $? -eq 0 ]; then
        print_success "MCMC analysis completed successfully for $run_name"
    else
        print_error "Failed to run MCMC analysis for $run_name"
        return 1
    fi
}

# Function to process a single folder
process_folder() {
    local folder=$1
    local run_name=$(basename "$folder")
    
    print_status "Processing folder: $folder"
    
    # Check if required files exist
    if ! check_required_files "$folder"; then
        print_error "Missing required files in $folder"
        return 1
    fi
    
    # Check if surrogate model already exists
    if [ -f "$folder/full_surrogate_model.pkl" ]; then
        print_warning "Surrogate model already exists for $run_name, skipping training"
    else
        # Train surrogate model
        if ! train_surrogate "$folder"; then
            print_error "Failed to train surrogate model for $run_name"
            return 1
        fi
    fi
    
    # Check if MCMC results already exist
    if [ -f "$folder/mcmc_results.npz" ]; then
        print_warning "MCMC results already exist for $run_name, skipping MCMC"
    else
        # Run MCMC analysis
        if ! run_mcmc "$folder"; then
            print_error "Failed to run MCMC analysis for $run_name"
            return 1
        fi
    fi
    
    print_success "Completed processing for $run_name"
}

# Main script
main() {
    print_status "Starting processing of geballe 41 gpa fwhm folders..."
    
    # Check if base directory exists
    check_directory "$BASE_DIR"
    
    # Find all fwhm folders
    fwhm_folders=($(find "$BASE_DIR" -maxdepth 1 -type d -name "*fwhm" | sort))
    
    if [ ${#fwhm_folders[@]} -eq 0 ]; then
        print_error "No fwhm folders found in $BASE_DIR"
        exit 1
    fi
    
    print_status "Found ${#fwhm_folders[@]} fwhm folders:"
    for folder in "${fwhm_folders[@]}"; do
        echo "  - $(basename "$folder")"
    done
    
    # Process each folder
    local failed_folders=()
    local successful_folders=()
    
    for folder in "${fwhm_folders[@]}"; do
        print_status "=========================================="
        print_status "Processing: $(basename "$folder")"
        print_status "=========================================="
        
        if process_folder "$folder"; then
            successful_folders+=("$(basename "$folder")")
        else
            failed_folders+=("$(basename "$folder")")
        fi
        
        echo ""
    done
    
    # Summary
    print_status "=========================================="
    print_status "PROCESSING SUMMARY"
    print_status "=========================================="
    
    if [ ${#successful_folders[@]} -gt 0 ]; then
        print_success "Successfully processed ${#successful_folders[@]} folders:"
        for folder in "${successful_folders[@]}"; do
            echo "  ✓ $folder"
        done
    fi
    
    if [ ${#failed_folders[@]} -gt 0 ]; then
        print_error "Failed to process ${#failed_folders[@]} folders:"
        for folder in "${failed_folders[@]}"; do
            echo "  ✗ $folder"
        done
    fi
    
    print_status "=========================================="
    
    if [ ${#failed_folders[@]} -eq 0 ]; then
        print_success "All folders processed successfully!"
        exit 0
    else
        print_error "Some folders failed to process. Check the logs above."
        exit 1
    fi
}

# Run main function
main "$@"