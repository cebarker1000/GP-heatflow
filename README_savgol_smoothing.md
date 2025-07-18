# Savitzky-Golay Smoothing for Experimental Heating Data

This document describes the implementation of Savitzky-Golay smoothing for experimental heating data in the heat flow simulation system.

## Overview

The Savitzky-Golay filter is a digital filter that can be used to smooth data while preserving the shape and width of peaks. This is particularly useful for experimental heating data that may contain noise or small fluctuations that can cause the simulation's pside curve to overshoot the actual experimental temperatures.

## Implementation

### 1. Configuration

To enable Savitzky-Golay smoothing, add the following configuration to your YAML config file:

```yaml
heating:
  ic_temp: 300.0
  fwhm: 13.2e-06
  file: "data/experimental/geballe_heat_data.csv"
  
  # Savitzky-Golay smoothing configuration
  smoothing:
    enabled: true
    window_length: 11  # Must be odd number, smaller than data length
    polyorder: 3       # Polynomial order (typically 2-4)
  
  bc:
    type: "gaussian"
    location: "zmin"
    material: "p_ins"
    center: 0.0
```

### 2. Parameters

- **`enabled`**: Set to `true` to enable smoothing, `false` to disable
- **`window_length`**: The length of the filter window (must be odd)
  - Recommended: 11 for moderate smoothing
  - Smaller values (5-9): minimal smoothing
  - Larger values (15-21): heavy smoothing
- **`polyorder`**: The order of the polynomial used to fit the samples
  - Recommended: 3 for most cases
  - Lower values (2): less smoothing, preserves more features
  - Higher values (4): more smoothing, may lose some features

### 3. How It Works

1. **Data Loading**: When smoothing is enabled, the experimental heating data is loaded and preprocessed
2. **Filtering**: The Savitzky-Golay filter is applied to the temperature data
3. **Storage**: Both raw and smoothed data are stored for comparison
4. **Interpolation**: The smoothed data is used for the heating boundary condition interpolation
5. **Visualization**: Output plots show both raw and smoothed experimental data

## Usage Examples

### Basic Usage

```python
from core.simulation_engine import OptimizedSimulationEngine
import yaml

# Load configuration with smoothing enabled
with open('configs/config_3_materials.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Create and run simulation
engine = OptimizedSimulationEngine(cfg, 'meshes/test', 'outputs/test')
results = engine.run()
```

### Testing Different Parameters

Use the test script to experiment with different smoothing parameters:

```bash
python test_savgol_smoothing.py
```

This will:
- Load both experimental data files
- Apply smoothing with different parameters
- Create comparison plots
- Save results for analysis

## Output

When smoothing is enabled, the simulation output includes:

1. **Enhanced Temperature Curves Plot**: Shows both raw and smoothed experimental data
2. **Console Output**: Indicates when smoothing is applied and with what parameters
3. **Residual Statistics**: Includes smoothing information in the analysis results

### Plot Features

- **P-side Comparison**: Shows simulation vs smoothed experimental data, with raw experimental data overlaid
- **O-side Comparison**: Shows simulation vs experimental oside data
- **Legend**: Clearly distinguishes between raw and smoothed experimental data

## Benefits

1. **Reduced Overshooting**: Smoothed data reduces the tendency of the simulation to overshoot experimental peaks
2. **Noise Reduction**: Removes high-frequency noise while preserving important features
3. **Better Agreement**: Improves agreement between simulation and experimental results
4. **Configurable**: Easy to adjust parameters for different experimental conditions

## Recommendations

### Parameter Selection

- **Start with defaults**: `window_length=11`, `polyorder=3`
- **For noisy data**: Increase `window_length` to 15-21
- **For clean data**: Decrease `window_length` to 7-9
- **For sharp peaks**: Use `polyorder=2` to preserve peak shapes
- **For smooth curves**: Use `polyorder=4` for maximum smoothing

### Validation

1. **Compare plots**: Always check that smoothed data preserves the essential features
2. **Test parameters**: Use the test script to find optimal parameters for your data
3. **Monitor residuals**: Check that smoothing improves agreement without over-smoothing

## Technical Details

### Algorithm

The Savitzky-Golay filter works by:
1. Fitting a polynomial to a window of data points
2. Using the polynomial to estimate the smoothed value at the center point
3. Moving the window and repeating the process

### Edge Handling

- The filter automatically handles edge effects
- Window length is adjusted if it exceeds the data length
- Even window lengths are automatically incremented to odd values

### Performance

- Minimal computational overhead
- Applied once during data loading
- No impact on simulation performance

## Troubleshooting

### Common Issues

1. **Window too large**: If `window_length` is larger than the data, it will be automatically reduced
2. **Even window length**: Automatically incremented to odd value
3. **Import error**: Ensure `scipy` is installed for the `savgol_filter` function

### Error Messages

- `"scipy.signal not available"`: Install scipy: `pip install scipy`
- `"Error applying Savitzky-Golay smoothing"`: Check parameter values and data format

## Files Modified

- `core/simulation_engine.py`: Added smoothing to `_load_heating_data()`
- `run_simulation.py`: Enhanced plotting to show raw and smoothed data
- `configs/config_3_materials.yaml`: Added smoothing configuration
- `configs/edmund.yaml`: Added smoothing configuration
- `test_savgol_smoothing.py`: Test script for parameter exploration 