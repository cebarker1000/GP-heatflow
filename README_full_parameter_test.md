# Full Parameter MCMC Test

This directory contains files to test MCMC sampling on the full 11-parameter distribution using synthetic data to check for potential bias in k estimates.

## Files

### Core Test Files
- `test_full_parameter_mcmc.py` - Main MCMC test that samples all 11 parameters
- `test_synthetic_data.py` - Simple test to generate and visualize synthetic data
- `plot_full_mcmc_results.py` - Comprehensive plotting and diagnostics for results

### Key Differences from Original MCMC

1. **No Monte Carlo Integration**: Unlike the original which uses Monte Carlo integration over nuisance parameters, this samples all 11 parameters directly
2. **Synthetic Data**: Uses known true k values but random nuisance parameters from priors
3. **Recovery Testing**: Compares posterior means against true k values to test for bias
4. **More Walkers**: Uses 24 walkers instead of 12 for better exploration in 11 dimensions
5. **Longer Burn-in**: Uses 15,000 burn-in steps for the higher dimensional space

## Synthetic Data Generation

The synthetic data is generated using a **hybrid approach** that better represents real experimental uncertainty:

- **Fixed k values**: The thermal conductivities are set to known true values
- **Random nuisance parameters**: The experimental geometry parameters (thicknesses, heat capacities, etc.) are drawn randomly from their prior distributions

This approach tests whether the MCMC can recover the true k values when the nuisance parameters are uncertain (as they are in real experiments).

## Usage

### Step 1: Test Synthetic Data Generation
```bash
python test_synthetic_data.py
```
This will:
- Generate synthetic data using fixed k values but random nuisance parameters from priors
- Create a plot of the synthetic data curve
- Print parameter summary showing which parameters are fixed vs. drawn from priors

### Step 2: Run Full Parameter MCMC
```bash
python test_full_parameter_mcmc.py
```
This will:
- Generate synthetic data using the hybrid approach
- Run MCMC sampling on all 11 parameters
- Compare posterior means to true k values
- Save results to `test_full_mcmc_results.npz`

### Step 3: Analyze Results
```bash
python plot_full_mcmc_results.py
```
This will create comprehensive diagnostics including:
- R-hat statistics for all 11 parameters
- Effective Sample Size (ESS) calculations
- Corner plots for all parameters with true values overlaid
- Trace plots for convergence diagnostics
- Parameter recovery statistics

## Fixed k Values Used

The synthetic data is generated using these fixed k values:

```python
FIXED_K_VALUES = {
    'k_sample': 3.3,             # Sample thermal conductivity (W/m/K)
    'k_ins': 10.0,               # Insulator thermal conductivity (W/m/K)
    'k_coupler': 350.0           # Coupler thermal conductivity (W/m/K)
}
```

## Nuisance Parameters

The nuisance parameters (experimental geometry) are drawn randomly from their prior distributions:

- `d_sample` - Sample thickness
- `rho_cv_sample` - Sample volumetric heat capacity
- `rho_cv_coupler` - Coupler volumetric heat capacity
- `rho_cv_ins` - Insulator volumetric heat capacity
- `d_coupler` - Coupler thickness
- `d_ins_pside` - P-side insulator thickness
- `d_ins_oside` - O-side insulator thickness
- `fwhm` - Full width at half maximum of heating pulse

## Recovery Percentage

The "Recovery %" represents the percentage error between the posterior mean and the true parameter value:

```
Recovery % = |posterior_mean - true_value| / |true_value| × 100
```

**Interpretation for k parameters:**
- **< 5%**: Excellent recovery - the MCMC is finding the right k values
- **5-10%**: Good recovery - some bias but reasonable
- **10-20%**: Moderate bias - the full parameter sampling might be introducing some bias
- **> 20%**: Poor recovery - significant bias, suggesting the approach may not work well

**Note**: For nuisance parameters, the "true" values are just one random draw from the prior, so recovery percentages may be higher and are less meaningful.

## Expected Output Files

After running the tests, you should have:

1. **Data Files:**
   - `test_full_mcmc_results.npz` - MCMC samples and metadata
   - `test_synthetic_data.png` - Synthetic data visualization

2. **Diagnostic Plots:**
   - `test_full_corner_plot.png` - Corner plot for all 11 parameters
   - `test_k_corner_plot.png` - Corner plot for k parameters only
   - `test_nuisance_corner_plot.png` - Corner plot for nuisance parameters only
   - `test_parameter_statistics.png` - Box plots of all parameters
   - `test_full_trace_plots.png` - Trace plots for all parameters
   - `test_likelihood_analysis.png` - Likelihood analysis plots

## Convergence Criteria

The plotting script checks for:
- **ESS > 200**: Effective Sample Size should be above 200 for reliable statistics
- **R-hat < 1.01**: Gelman-Rubin diagnostic should be close to 1.0 for convergence

## Purpose

This test helps determine whether sampling from the full 11-parameter distribution biases your k estimates compared to the original approach of marginalizing over nuisance parameters. 

The key question is: **Can the MCMC recover the true k values when nuisance parameters are uncertain?**

If the k recovery percentages are high (poor recovery), it suggests that:

1. The full parameter space might be too complex for MCMC to explore effectively
2. There might be parameter degeneracies or correlations making it hard to identify the true k values
3. The original approach of marginalizing over nuisance parameters might indeed be better

If the k recovery percentages are low (good recovery), it suggests the full parameter approach could work well and might not introduce significant bias in k estimates.

## Key Insight

This approach better represents real experimental conditions where:
- We have some knowledge of the k values we're trying to measure
- The experimental geometry parameters have uncertainty described by their prior distributions
- We want to know if the MCMC can "see through" the nuisance parameter uncertainty to find the true k values 