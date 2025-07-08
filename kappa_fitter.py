# kappa_fitter.py
"""End-to-end fitter to estimate the three thermal conductivities (κ_sample, κ_au_coupler, κ_insulator)
from experimental oside-temperature data, using the previously-trained GP/FPCA surrogate.

The implementation follows the game-plan discussed in the design conversation:
1. Pre-process experiment CSV: load, normalise, interpolate, project to FPCA coefficients.
2. Build a black-box Gaussian log-likelihood that combines surrogate predictions and
   (optional) sensor noise.
3. Optimise θ = (κ_sample, κ_au_coupler, κ_insulator) with L-BFGS-B under bounds.
4. Estimate the local covariance at the optimum from the numerical Hessian.
5. (Optional) Two-stage Monte-Carlo to propagate uncertainty in eight nuisance parameters.
6. Wrap everything in a tidy KappaFitResult dataclass.

The code purposefully keeps dependencies light (numpy/pandas/scipy) and only relies on
utility modules that are already part of this repository (analysis.uq_wrapper and the
FullSurrogateModel definition from train_surrogate_models.py).  If the ``UQpy`` package
is available in the environment, the MaximumLikelihoodEstimation class will be used for
step-3; otherwise we gracefully fall back to scipy.optimize.
"""

from __future__ import annotations

import warnings
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import inv
from scipy import stats

# -----------------------------------------------------------------------------
# Project-specific imports (already present in the repo)
# -----------------------------------------------------------------------------
from analysis.uq_wrapper import load_fpca_model, project_curve_to_fpca
from train_surrogate_models import FullSurrogateModel, get_parameter_ranges

# -----------------------------------------------------------------------------
# Optional UQpy import – used if available
# -----------------------------------------------------------------------------
try:
    from UQpy.inference import MLE  # type: ignore

    _UQPY_AVAILABLE = True
except ImportError:  # pragma: no cover – gracefully degrade if UQpy is missing
    _UQPY_AVAILABLE = False
    warnings.warn(
        "UQpy package is not available – falling back to scipy.optimize for the\n"
        "maximum-likelihood estimation step.  Install UQpy to make use of its\n"
        "advanced optimisation and diagnostics features.")

# -----------------------------------------------------------------------------
# Dataclass to hold results
# -----------------------------------------------------------------------------
@dataclass
class KappaFitResult:
    """Container for conductivity-fit results."""

    theta_hat: np.ndarray                        # Shape (3,) – MLE point estimate
    loglik_hat: float                            # Log-likelihood at optimum
    hessian: np.ndarray                          # Numerical Hessian (3×3)
    cov: np.ndarray                              # Inverse Hessian (local covariance)

    # Optional Monte-Carlo results ------------------------------------------------
    mc_thetas: Optional[np.ndarray] = None        # Shape (n_outer, 3)
    mc_stats: Optional[Dict[str, np.ndarray]] = None  # mean, cov, etc.

    # Meta information -----------------------------------------------------------
    success: bool = True
    message: str = ""

    def __post_init__(self):
        self.theta_hat = np.asarray(self.theta_hat, dtype=float)
        self.cov = np.asarray(self.cov, dtype=float)
        self.hessian = np.asarray(self.hessian, dtype=float)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def std(self) -> np.ndarray:
        """1-σ standard deviations of the three κ parameters from the local Hessian."""
        return np.sqrt(np.diag(self.cov))

    def __repr__(self) -> str:  # pragma: no cover – pretty print only
        names = ["k_sample", "k_au_coupler", "k_insulator"]
        lines = ["KappaFitResult (MLE)"]
        for n, v, s in zip(names, self.theta_hat, self.std):
            lines.append(f"  {n:15s}: {v:10.4g}  ± {s:8.2g} (1σ)")
        lines.append(f"  log-likelihood   : {self.loglik_hat:.4f}")
        if self.mc_thetas is not None:
            mu = self.mc_stats["mean"]  # type: ignore[index]
            sd = np.sqrt(np.diag(self.mc_stats["cov"]))  # type: ignore[index]
            lines.append("  ---- Monte-Carlo summary (outer loop) ----")
            for n, m, s in zip(names, mu, sd):
                lines.append(f"  {n:15s}: {m:10.4g}  ± {s:8.2g}")
        if not self.success:
            lines.append(f"  optimisation FAILED: {self.message}")
        return "\n".join(lines)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _load_surrogate_model(path: str) -> FullSurrogateModel:
    """Load a stored :class:`FullSurrogateModel` instance from *path*."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    # Check if it's already a FullSurrogateModel instance
    if isinstance(data, FullSurrogateModel):
        return data
    
    # If it's a dictionary (saved by save_model), reconstruct the instance
    if isinstance(data, dict):
        return FullSurrogateModel(
            fpca_model=data['fpca_model'],
            gps=data['gps'],
            scaler=data['scaler'],
            parameter_names=data['parameter_names'],
            param_ranges=data['param_ranges']
        )
    
    raise TypeError(f"Unexpected data type in {path}: {type(data)}")


def _preprocess_experimental_curve(
    csv_path: str,
    fpca_model: Dict[str, np.ndarray | np.generic],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load *csv_path* and produce an interpolated, normalised oside curve.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (time_grid, curve_normalised) where *curve_normalised* has the same
        length as ``fpca_model['mean_curve']``.
    """
    df = pd.read_csv(csv_path)
    required_cols = {"time", "oside"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Experimental CSV must contain columns {required_cols}, got {df.columns}.")

    t_exp = df["time"].values.astype(float)
    y_raw = df["oside"].values.astype(float)

    # Simple normalisation: remove initial value and divide by max excursion
    initial = y_raw[0]
    excursion = y_raw.max() - y_raw.min()
    if excursion <= 0:
        raise RuntimeError("Experimental data has zero excursion – cannot normalise.")
    y_norm = (y_raw - initial) / excursion

    # Interpolate onto the time grid expected by the FPCA model
    # Cast to ndarray in case NPZ loader returned a numpy scalar object
    n_time = int(np.asarray(fpca_model["mean_curve"]).shape[0])
    t_grid = np.linspace(float(t_exp[0]), float(t_exp[-1]), n_time)
    curve_interp = np.interp(t_grid, t_exp, y_norm)
    return t_grid, curve_interp


def _build_param_vector(
    theta_three: np.ndarray,
    surrogate: FullSurrogateModel,
    nuisance_values: Dict[str, float] | None = None,
) -> np.ndarray:
    """Create a full 8-parameter vector for the surrogate from the three κ values.

    *nuisance_values* can be used to specify fixed values for the remaining five
    nuisance parameters.  If omitted, they are set to the midpoint of their
    training ranges.
    """
    theta_three = np.asarray(theta_three, dtype=float).ravel()
    if theta_three.size != 3:
        raise ValueError("theta_three must contain exactly three elements.")

    param_vec = np.empty(surrogate.n_parameters, dtype=float)
    ranges = surrogate.param_ranges  # dict(name -> (lo, hi))

    # Mapping for the three conductivities -------------------------------------------------
    mapping = {
        "k_sample": theta_three[0],
        "k_coupler": theta_three[1],
        "k_ins": theta_three[2],
    }

    # Build parameter vector in the correct order
    for i, name in enumerate(surrogate.parameter_names):
        if name in mapping:
            param_vec[i] = mapping[name]
        else:
            if nuisance_values and name in nuisance_values:
                param_vec[i] = nuisance_values[name]
            else:
                lo, hi = ranges[name]
                param_vec[i] = 0.5 * (lo + hi)  # midpoint default
    return param_vec


def _gaussian_loglik(
    theta_three: np.ndarray,
    surrogate: FullSurrogateModel,
    fpca_obs: np.ndarray,
    sensor_sigma: float,
    nuisance_values: Dict[str, float] | None = None,
) -> float:
    """Gaussian log-likelihood of *fpca_obs* given *theta_three*.

    The GP surrogate provides predictive means and *per-component* standard
    deviations (assumed independent).  We augment those variances with a
    constant *sensor_sigma**2* measurement noise.
    """

    # Build full 8-D vector expected by the surrogate ------------------------
    x_full = _build_param_vector(theta_three, surrogate, nuisance_values)

    mu, std = surrogate.predict_fpca_coefficients(x_full)
    mu = mu.ravel()
    std = std.ravel()
    var = std ** 2 + sensor_sigma ** 2

    # Diagonal Gaussian likelihood
    res2 = (fpca_obs - mu) ** 2
    logprob = -0.5 * np.sum(res2 / var + np.log(2 * np.pi * var))
    return logprob


# -----------------------------------------------------------------------------
# Public API – main driver
# -----------------------------------------------------------------------------

def fit_kappas(
    csv_path: str,
    surrogate_path: str = "outputs/full_surrogate_model.pkl",
    fpca_model_path: str = "outputs/fpca_model.npz",
    sensor_sigma: float = 0.02,
    bounds_override: Dict[str, Tuple[float, float]] | None = None,
    monte_carlo_outer: int = 0,
    random_state: int | None = 42,
) -> KappaFitResult:
    """Estimate the three κ parameters from experimental *csv_path*.

    Parameters
    ----------
    csv_path
        Path to the experimental CSV file (must contain *time* and *oside* columns).
    surrogate_path, fpca_model_path
        File paths to the trained surrogate and FPCA model artefacts.
    sensor_sigma
        One-sigma measurement noise (applied to each FPCA coefficient).
    bounds_override
        Optional dict to override the default parameter bounds.
    monte_carlo_outer
        If >0, perform a two-stage Monte-Carlo with *monte_carlo_outer* outer
        samples of the eight nuisance parameters.  For each outer sample we
        re-optimise θ; the distribution of point estimates is returned.
    random_state
        Seed for the random number generator (for reproducibility).
    """

    # ------------------------------------------------------------------
    # Load artefacts and preprocess data
    # ------------------------------------------------------------------
    surrogate = _load_surrogate_model(surrogate_path)
    fpca_model = load_fpca_model(fpca_model_path)

    _, curve_norm = _preprocess_experimental_curve(csv_path, fpca_model)
    fpca_obs = project_curve_to_fpca(curve_norm, fpca_model)

    # ------------------------------------------------------------------
    # Build optimiser settings
    # ------------------------------------------------------------------
    ranges = surrogate.param_ranges  # default ranges from training
    default_bounds = {
        "k_sample": ranges["k_sample"],
        "k_coupler": ranges["k_coupler"],
        "k_ins": ranges["k_ins"],
    }
    if bounds_override:
        default_bounds.update(bounds_override)

    bnds = [default_bounds["k_sample"], default_bounds["k_coupler"], default_bounds["k_ins"]]

    # Initial guess – mid-point of each bound
    x0 = np.array([0.5 * (lo + hi) for lo, hi in bnds], dtype=float)

    # Objective (negative log-likelihood for minimisation) -------------------
    def _nll(theta: np.ndarray, nuisance_vals: Dict[str, float] | None = None) -> float:
        return -_gaussian_loglik(theta, surrogate, fpca_obs, sensor_sigma, nuisance_vals)

    # ------------------------------------------------------------------
    # Perform the optimisation (either via UQpy or SciPy)
    # ------------------------------------------------------------------
    if _UQPY_AVAILABLE:
        # Build wrapper for UQpy – it expects bounds separately
        mle_runner = MLE(
            loglikelihood=lambda t: _gaussian_loglik(t, surrogate, fpca_obs, sensor_sigma),
            initial_guess=x0,
            bounds=bnds,
            method="L-BFGS-B",
        )
        theta_hat = mle_runner.mle
        loglik_hat = _gaussian_loglik(theta_hat, surrogate, fpca_obs, sensor_sigma)
        success = True
        message = "UQpy optimisation completed."
    else:
        res = minimize(_nll, x0, method="L-BFGS-B", bounds=bnds)
        theta_hat = res.x
        loglik_hat = -res.fun
        success = res.success
        message = res.message

    # ------------------------------------------------------------------
    # Numerical Hessian around optimum (simple finite-difference)
    # ------------------------------------------------------------------
    eps = np.maximum(1e-6, 1e-4 * np.abs(theta_hat))
    hessian = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            th_ij_pp = theta_hat.copy()
            th_ij_pm = theta_hat.copy()
            th_ij_mp = theta_hat.copy()
            th_ij_mm = theta_hat.copy()

            th_ij_pp[[i, j]] += [eps[i], eps[j]]
            th_ij_pm[i] += eps[i]
            th_ij_pm[j] -= eps[j]
            th_ij_mp[i] -= eps[i]
            th_ij_mp[j] += eps[j]
            th_ij_mm[[i, j]] -= [eps[i], eps[j]]

            f_pp = _nll(th_ij_pp)
            f_pm = _nll(th_ij_pm)
            f_mp = _nll(th_ij_mp)
            f_mm = _nll(th_ij_mm)

            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps[i] * eps[j])

    # Symmetrise to improve numerical stability
    hessian = 0.5 * (hessian + hessian.T)

    # Invert to get local covariance; use pseudo-inverse if necessary
    try:
        cov = inv(hessian)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hessian)
        warnings.warn("Hessian was singular – using pseudo-inverse for covariance estimate.")

    # ------------------------------------------------------------------
    # Optional two-stage Monte-Carlo over nuisance parameters
    # ------------------------------------------------------------------
    mc_thetas = None
    mc_stats = None
    if monte_carlo_outer > 0:
        rng = np.random.default_rng(random_state)
        nuisance_names = [
            n for n in surrogate.parameter_names
            if n not in ("k_sample", "k_au_coupler", "k_insulator")
        ]
        ranges_all = surrogate.param_ranges

        mc_thetas = np.zeros((monte_carlo_outer, 3), dtype=float)
        for i in range(monte_carlo_outer):
            nuisance_vals = {
                n: rng.uniform(*ranges_all[n]) for n in nuisance_names
            }
            # Re-optimise θ for this outer sample (inner loop handled by optimiser)
            res_i = minimize(lambda th: _nll(th, nuisance_vals), theta_hat, method="L-BFGS-B", bounds=bnds)
            mc_thetas[i] = res_i.x

        mc_stats = {
            "mean": mc_thetas.mean(axis=0),
            "cov": np.cov(mc_thetas, rowvar=False),
        }

    # ------------------------------------------------------------------
    # Package results ---------------------------------------------------
    # ------------------------------------------------------------------
    return KappaFitResult(
        theta_hat=theta_hat,
        loglik_hat=loglik_hat,
        hessian=hessian,
        cov=cov,
        mc_thetas=mc_thetas,
        mc_stats=mc_stats,
        success=success,
        message=message,
    )


# -----------------------------------------------------------------------------
# CLI helper (so the module can be run as a script) ----------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover – manual CLI usage only
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="Estimate κ parameters from experimental CSV data.")
    parser.add_argument("csv", help="Path to experimental CSV file with time + oside columns.")
    parser.add_argument("--surrogate", default="outputs/full_surrogate_model.pkl", help="Pickle file with trained FullSurrogateModel.")
    parser.add_argument("--fpca", default="outputs/fpca_model.npz", help="NPZ file with trained FPCA model.")
    parser.add_argument("--sensor-sigma", type=float, default=0.02, help="Sensor noise sigma (in FPCA coefficient space).")
    parser.add_argument("--mc", type=int, default=0, help="Number of outer Monte-Carlo samples for nuisance params.")

    args = parser.parse_args()

    result = fit_kappas(
        csv_path=args.csv,
        surrogate_path=args.surrogate,
        fpca_model_path=args.fpca,
        sensor_sigma=args.sensor_sigma,
        monte_carlo_outer=args.mc,
    )

    print(result) 