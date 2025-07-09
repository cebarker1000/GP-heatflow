#!/usr/bin/env python3
"""
Numeric diagnostics for Monte Carlo kappa samples.
"""
import numpy as np

# Load MC results
mc_data = np.load("outputs/propagated_k_values.npz", allow_pickle=True)
k_samples = mc_data['k_samples']  # shape (N, 3)
param_names = ["κ_sample", "κ_ins", "κ_coupler"]

# Best-fit point
best_draw_idx = mc_data['best_draw_idx'] if 'best_draw_idx' in mc_data else None
best_point = k_samples[best_draw_idx] if best_draw_idx is not None else None

# Fixed parameters used in MC (if available)
fixed_params_samples = mc_data['fixed_params_samples'] if 'fixed_params_samples' in mc_data else None

print("=" * 60)
print("MONTE CARLO NUMERIC DIAGNOSTICS")
print("=" * 60)
print(f"Number of samples: {len(k_samples)}")
print(f"Parameter names: {param_names}")

# 1. Empirical covariance & correlation
print("\n" + "=" * 40)
print("1. COVARIANCE & CORRELATION MATRICES")
print("=" * 40)

cov = np.cov(k_samples, rowvar=False)
corr = np.corrcoef(k_samples, rowvar=False)

print("Covariance matrix:")
for i, row in enumerate(cov):
    print(f"  [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]  # {param_names[i]}")

print("\nCorrelation matrix:")
for i, row in enumerate(corr):
    print(f"  [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]  # {param_names[i]}")

# 2. Principal directions & identifiability
print("\n" + "=" * 40)
print("2. PRINCIPAL DIRECTIONS & IDENTIFIABILITY")
print("=" * 40)

eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

print(f"Eigenvalues: {eigvals}")
print(f"Condition number: {eigvals[0]/eigvals[-1]:.2e}")

print("\nPrincipal axes (columns):")
for i, (eigval, eigvec) in enumerate(zip(eigvals, eigvecs.T)):
    print(f"  PC{i+1} (λ={eigval:.4f}): [{eigvec[0]:8.4f}, {eigvec[1]:8.4f}, {eigvec[2]:8.4f}]")

# Check for identifiability issues
condition_number = eigvals[0] / eigvals[-1]
if condition_number > 100:
    print(f"\n⚠️  WARNING: Large condition number ({condition_number:.2e}) suggests poor identifiability")
    print("   This indicates a 'flat' direction in parameter space")
else:
    print(f"\n✅ Condition number ({condition_number:.2e}) suggests good identifiability")

# 3. Mahalanobis distance of the best-fit
if best_point is not None:
    print("\n" + "=" * 40)
    print("3. MAHALANOBIS DISTANCE OF BEST-FIT")
    print("=" * 40)
    
    mu = k_samples.mean(axis=0)
    delta = best_point - mu
    D2 = delta @ np.linalg.inv(cov) @ delta
    
    print(f"Best-fit point: {best_point}")
    print(f"Sample mean: {mu}")
    print(f"Deviation: {delta}")
    print(f"Mahalanobis D² of θ̂: {D2:.4f}")
    
    # Interpret the Mahalanobis distance
    if D2 < 3:
        print("✅ Best-fit is typical within the cloud (D² < 3)")
    elif D2 < 6:
        print("⚠️  Best-fit is somewhat atypical (3 ≤ D² < 6)")
    else:
        print("❌ Best-fit is very atypical (D² ≥ 6)")

# 4. Credible intervals (numeric)
print("\n" + "=" * 40)
print("4. CREDIBLE INTERVALS")
print("=" * 40)

ci68 = np.percentile(k_samples, [16, 84], axis=0)
ci95 = np.percentile(k_samples, [2.5, 97.5], axis=0)

print("68% Credible Intervals:")
for i, (low, high) in enumerate(ci68.T):
    print(f"  {param_names[i]}: [{low:.4f}, {high:.4f}]")

print("\n95% Credible Intervals:")
for i, (low, high) in enumerate(ci95.T):
    print(f"  {param_names[i]}: [{low:.4f}, {high:.4f}]")

# Additional summary statistics
print("\n" + "=" * 40)
print("5. SUMMARY STATISTICS")
print("=" * 40)

means = k_samples.mean(axis=0)
stds = k_samples.std(axis=0)

print("Parameter means and standard deviations:")
for i, (name, mean, std) in enumerate(zip(param_names, means, stds)):
    print(f"  {name}: {mean:.4f} ± {std:.4f}")

# Parameter ranges
mins = k_samples.min(axis=0)
maxs = k_samples.max(axis=0)

print("\nParameter ranges:")
for i, (name, min_val, max_val) in enumerate(zip(param_names, mins, maxs)):
    print(f"  {name}: [{min_val:.4f}, {max_val:.4f}]")

# 6. Fixed parameters used in MC
if fixed_params_samples is not None:
    print("\n" + "=" * 40)
    print("6. FIXED PARAMETERS USED IN MC")
    print("=" * 40)
    
    # Convert list of dicts to array for easier computation
    fixed_param_names = list(fixed_params_samples[0].keys())
    fixed_param_array = np.array([[sample[name] for name in fixed_param_names] 
                                 for sample in fixed_params_samples])
    
    print("Fixed parameter averages (from MC draws):")
    for i, name in enumerate(fixed_param_names):
        mean_val = np.mean(fixed_param_array[:, i])
        std_val = np.std(fixed_param_array[:, i])
        print(f"  {name}: {mean_val:.6e} ± {std_val:.6e}")
    
    print(f"\nNote: These are the averages of the {len(fixed_params_samples)} fixed parameter")
    print("draws used in the Monte Carlo process (not the kappa parameters above).")

print("\n" + "=" * 60)
print("DIAGNOSTICS COMPLETE")
print("=" * 60)
