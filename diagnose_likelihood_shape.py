"""diagnose_likelihood_shape.py

Generate 1-D slices of the (negative) log-likelihood as each κ parameter is varied
individually while keeping the remaining κ's fixed and all nuisance parameters
ϕ fixed at their prior centres.

This implements diagnostic A1 from the suggestions: visual inspection of the
likelihood “shape” to determine whether it is informative or essentially flat.

The script produces a PNG file for each parameter as well as an on-screen plot.
"""

from __future__ import annotations

import pathlib
from typing import List

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import utilities from the existing codebase
# ---------------------------------------------------------------------------
from uqpy_MCMC import (
    load_experimental_data,
    interpolate_to_surrogate_grid,
)
from uqpy_surrogate import timeseries_model  # surrogate forward model
from analysis.config_utils import get_fixed_params_from_config, load_distributions_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Sensor noise variance (must match the value used in the main MCMC script)
SENSOR_VARIANCE: float = 1e-4  # (0.01)² for 1% standard deviation

# Number of grid points per κ dimension in the diagnostic plot
N_GRID: int = 100

# Output directory for the plots
OUTPUT_DIR = pathlib.Path("outputs/diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load fixed quantities
# ---------------------------------------------------------------------------
print("Loading experimental data and fixed parameters …")

y_obs_raw, exp_time = load_experimental_data()
y_obs_interp = interpolate_to_surrogate_grid(y_obs_raw, exp_time)

a_phi_fixed = get_fixed_params_from_config()  # shape (8,)

# κ prior bounds taken directly from the YAML config (robust to API changes)
config = load_distributions_config()
k_param_names = ["k_sample", "k_ins", "k_coupler"]
_k_low = np.array([config["parameters"][p]["low"] for p in k_param_names])
_k_high = np.array([config["parameters"][p]["high"] for p in k_param_names])
_k_mid = (_k_low + _k_high) / 2

PARAM_NAMES: List[str] = [
    r"$k_{\mathrm{sample}}$",
    r"$k_{\mathrm{ins}}$",
    r"$k_{\mathrm{coupler}}$",
]

# ---------------------------------------------------------------------------
# Helper: log-likelihood with fixed ϕ, variable κ
# ---------------------------------------------------------------------------

def gaussian_loglike(y_pred: np.ndarray, y_obs: np.ndarray, *, sigma2: float) -> float:
    """Scalar Gaussian log-likelihood."""
    resid = y_pred - y_obs
    return -0.5 * np.sum(resid ** 2 / sigma2 + np.log(2 * np.pi * sigma2))


def loglike_fixed_phi(k_vector: np.ndarray) -> float:
    """Return log ℓ for a single set of κ values, keeping ϕ fixed."""
    # Build full 11-vector: [ϕ_fixed | κ]
    params_full = np.hstack([a_phi_fixed, k_vector])
    # Surrogate forward model expects 2-D array; we pass shape (1, 11)
    y_pred = timeseries_model(params_full[None, :])[0]
    return gaussian_loglike(y_pred, y_obs_interp, sigma2=SENSOR_VARIANCE)


# ---------------------------------------------------------------------------
# Main diagnostic: 1-D slices
# ---------------------------------------------------------------------------

def diagnostic_1d_slices() -> None:
    """Compute and plot 1-D log-likelihood slices for each κ parameter."""
    for idx, name in enumerate(PARAM_NAMES):
        # Create grid for the *current* κ, others fixed at midpoint
        grid = np.linspace(_k_low[idx], _k_high[idx], N_GRID)
        k_vec_fixed = _k_mid.copy()

        # Vectorised evaluation: build (N_GRID, 11) array
        params_all = np.tile(np.hstack([a_phi_fixed, k_vec_fixed]), (N_GRID, 1))
        params_all[:, 8 + idx] = grid  # ϕ (0-7) | κ (8-10)

        # Forward model predictions (N_GRID, T)
        y_preds = timeseries_model(params_all)

        # Compute log ℓ for each grid point
        log_likelihoods = np.empty(N_GRID)
        for i in range(N_GRID):
            log_likelihoods[i] = gaussian_loglike(
                y_preds[i], y_obs_interp, sigma2=SENSOR_VARIANCE
            )

        # ------------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(grid, log_likelihoods, "-o", ms=3)
        ax.set_xlabel(name)
        ax.set_ylabel("log-likelihood (fixed ϕ)")
        ax.set_title(f"1-D log-likelihood slice – {name}")
        ax.grid(True, ls=":", lw=0.5)
        fig.tight_layout()

        # Save + show
        # Build a file-safe name by stripping LaTeX/TeX characters
        safe_name = (
            name.replace("$", "")
                .replace("\\", "")
                .replace("{", "")
                .replace("}", "")
        )
        outfile = OUTPUT_DIR / f"loglike_slice_{idx}_{safe_name}.png"
        fig.savefig(outfile, dpi=300)
        print(f"Saved plot to {outfile}")
        plt.show()


if __name__ == "__main__":
    diagnostic_1d_slices() 