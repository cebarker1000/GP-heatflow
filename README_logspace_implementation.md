# Log-Space MCMC Implementation

This document describes the implementation of log-space MCMC sampling for thermal conductivity estimation, which should improve convergence and increase Effective Sample Size (ESS).

## Overview

The log-space MCMC implementation transforms all parameters to log-space for sampling, which provides several benefits:

1. **Better parameterization**: Thermal conductivity and other physical parameters often span multiple orders of magnitude
2. **Improved mixing**: Log-space sampling provides more efficient exploration of parameter space
3. **Natural constraints**: Ensures all parameters remain positive
4. **Higher ESS**: Should result in better effective sample sizes and reduced autocorrelation

## Implementation Details

### Core Components

1. **`analysis/config_utils.py`** - Enhanced with log-space transformation utilities:
   - `real_to_log_space()` - Transform parameters from real-space to log-space
   - `log_to_real_space()` - Transform parameters from log-space back to real-space
   - `compute_jacobian_correction()` - Compute Jacobian correction for the transformation
   - `create_logspace_distributions()` - Create UQpy distributions for log-space sampling
   - `get_logspace_bounds()` - Get appropriate bounds for log-space sampling

2. **`uqpy_MCMC_logspace.py`** - New MCMC script that:
   - Samples all parameters in log-space
   - Applies proper Jacobian corrections
   - Saves results in both log-space and real-space formats
   - Includes comprehensive diagnostics and plotting

3. **`plot_mcmc_results_logspace.py`** - Enhanced plotting script that:
   - Loads and compares results from different sampling approaches
   - Creates comparison plots for convergence diagnostics
   - Shows ESS and R-hat comparisons

4. **`test_logspace_transformations.py`** - Test script to verify:
   - Transformation reversibility
   - Jacobian correction accuracy
   - Parameter range handling

### Mathematical Details

**Transformation**: For each parameter θ, we use the log transformation:
- Log-space: `log_θ = log(θ)`
- Real-space: `θ = exp(log_θ)`

**Jacobian Correction**: The Jacobian determinant is:
- `|∂θ/∂log_θ| = θ = exp(log_θ)`
- For multiple parameters: `J = ∏ᵢ θᵢ`

**Log-likelihood**: The total log-likelihood includes the Jacobian correction:
- `log L_total = log L_real + log J`

## Usage Instructions

### 1. Test the Implementation

First, test that the log-space transformations work correctly:

```bash
python test_logspace_transformations.py
```

This will verify that:
- Transformations are reversible
- Jacobian corrections are accurate
- Parameter ranges are handled correctly

### 2. Run Log-Space MCMC

Run the log-space MCMC sampling:

```bash
python uqpy_MCMC_logspace.py
```

This will:
- Sample all 11 parameters in log-space
- Save results to `mcmc_results_logspace.npz` (log-space) and `mcmc_results_realspace.npz` (real-space)
- Create trace plots and corner plots for both spaces
- Print comprehensive statistics

### 3. Compare Results

Compare the log-space results with your original MCMC results:

```bash
python plot_mcmc_results_logspace.py
```

This will:
- Load results from all available approaches (log-space, real-space, original)
- Create comparison plots for parameter distributions
- Compare ESS and convergence diagnostics
- Generate comprehensive comparison reports

## Expected Improvements

### Convergence Metrics

You should see improvements in:

1. **Effective Sample Size (ESS)**:
   - Higher ESS values across all parameters
   - More uniform ESS across parameters
   - Better exploration of parameter space

2. **R-hat (Gelman-Rubin diagnostic)**:
   - R-hat values closer to 1.0
   - Better mixing between chains
   - More stable convergence

3. **Trace Plots**:
   - Smoother, less noisy individual traces
   - Better mixing and exploration
   - Reduced autocorrelation

### Parameter Estimates

The log-space sampling should provide:
- More robust parameter estimates
- Better uncertainty quantification
- More efficient exploration of the posterior

## Output Files

### MCMC Results

- **`mcmc_results_logspace.npz`**: Log-space samples and diagnostics
- **`mcmc_results_realspace.npz`**: Real-space samples (transformed from log-space)
- **`trace_plots_logspace.png`**: Trace plots in log-space
- **`trace_plots_realspace.png`**: Trace plots in real-space
- **`k_corner_plot_logspace.png`**: Corner plot of k parameters in log-space
- **`k_corner_plot_realspace.png`**: Corner plot of k parameters in real-space

### Comparison Results

- **`parameter_comparison.png`**: Histogram comparison of parameter distributions
- **`trace_plots_comparison.png`**: Side-by-side trace plot comparison
- **`ess_comparison.png`**: ESS comparison across sampling approaches
- **`k_corner_plot_*.png`**: Corner plots for each approach

## Key Features

### 1. Full Log-Space Sampling

All 11 parameters are sampled in log-space, not just the thermal conductivity parameters. This provides:
- Better exploration of parameter space across orders of magnitude
- More natural parameterization for positive physical parameters
- Improved mixing for all parameters

### 2. Proper Jacobian Corrections

The implementation includes correct Jacobian corrections to ensure:
- Unbiased parameter estimates
- Proper posterior probability calculations
- Correct uncertainty quantification

### 3. Dual Output Format

Results are saved in both log-space and real-space formats, allowing you to:
- Analyze the sampling efficiency in log-space
- Interpret results in the original physical units
- Compare with previous real-space results

### 4. Comprehensive Diagnostics

The implementation provides extensive diagnostics:
- ESS and R-hat calculations
- Trace plots in both spaces
- Corner plots for parameter correlations
- Detailed statistical summaries

## Troubleshooting

### Common Issues

1. **Import Errors**: Some linter warnings about missing UQpy or corner libraries are expected and don't affect functionality.

2. **Large Jacobian Values**: Very large Jacobian corrections may indicate numerical issues. Check that parameters are in reasonable ranges.

3. **Convergence Issues**: If log-space sampling doesn't improve convergence, consider:
   - Adjusting the number of walkers
   - Increasing the number of samples
   - Checking parameter bounds

### Performance Considerations

- Log-space sampling may be slightly slower due to the transformation overhead
- The Jacobian correction adds computational cost
- Memory usage is similar to the original implementation

## Comparison with Original Approach

| Aspect | Original MCMC | Log-Space MCMC |
|--------|---------------|----------------|
| Parameter Space | Real-space | Log-space |
| Sampling Efficiency | Lower | Higher |
| ESS | Lower | Higher |
| Mixing | Slower | Faster |
| Parameter Constraints | Manual bounds | Natural (positive) |
| Numerical Stability | Good | Better |
| Interpretation | Direct | Requires transformation |

## Next Steps

After running the log-space MCMC:

1. **Compare ESS values** between approaches
2. **Check trace plots** for improved mixing
3. **Analyze parameter correlations** in both spaces
4. **Validate results** against physical expectations
5. **Consider further improvements** based on the results

The log-space implementation should provide significant improvements in convergence and sampling efficiency, especially for the thermal conductivity parameters that span multiple orders of magnitude. 