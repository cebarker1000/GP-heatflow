#!/bin/bash

# Configuration for Edmund data
# Copy this file and modify the variables as needed

# Base directory for the data
BASE_DIR="outputs/edmund"

# Experimental data directory
EXPERIMENTAL_DATA_DIR="data/experimental"

# Experimental data mapping for Edmund runs
declare -A EXPERIMENTAL_DATA_MAPPING=(
    ["run1_fwhm"]="edmund_71Gpa_run1.csv"
    ["run2_fwhm"]="edmund_71Gpa_run2.csv"
    ["run3_fwhm"]="edmund_71Gpa_run3.csv"
    ["run4_fwhm"]="edmund_71Gpa_run4.csv"
)

# MCMC Parameters
N_WALKERS=60
N_SAMPLES=1000000
BURN_LENGTH=20000

# Training Parameters
TEST_FRACTION=0.2
RANDOM_STATE=42

# Source the main script with these configurations
source process_geballe_fwhm_generic.sh