from UQpy.run_model import RunModel
from UQpy.run_model.model_execution import PythonModel
from UQpy.inference.inference_models.LogLikelihoodModel import LogLikelihoodModel
from uqpy_surrogate import uncertainty_model, log_likelihood
import pandas as pd
import numpy as np

print('creating models')

# get names of parameters to map to outputs
param_list = ['d_sample', 'rho_cv_sample', 'rho_cv_coupler', 'rho_cv_ins', 'd_coupler', 'd_ins_oside', 'd_ins_pside', 'fwhm', 'k_sample', 'k_ins', 'k_coupler']

fpca_model = PythonModel(model_script="uqpy_surrogate.py", model_object_name="fpca_model")
fpca = RunModel(model=fpca_model)
uncertainty_model = PythonModel(model_script="uqpy_surrogate.py", model_object_name="uncertainty_model")
uncertainty = RunModel(model=uncertainty_model)

# define free and fixed indexies
IDX_FIXED = [0, 1, 2, 3, 4, 5, 6, 7]
IDX_FREE = [8, 9, 10]



# get fpca of experimental data
df = pd.read_csv('data/experimental/geballe_heat_data.csv')
df['oside normed'] = (df['oside'] - df['oside'].iloc[0]) / (df['temp'].max() - df['temp'].iloc[0])

from analysis.uq_wrapper import project_curve_to_fpca
from train_surrogate_models import FullSurrogateModel
surrogate = FullSurrogateModel.load_model(f"outputs/full_surrogate_model.pkl")
fpca_model = surrogate.fpca_model
fpca_scores = project_curve_to_fpca(df['oside normed'].values[1:], fpca_model)

sigma_meas = 0.0
def ll_free(params=None, data=None):
    return log_likelihood(
        params,
        data,
        use_sigma_gp=False
    )

ll_model = LogLikelihoodModel(
    n_parameters=3,
    log_likelihood=ll_free,
)

from uqpy_surrogate import least_squares

def ls_log_likelihood(params=None, data=None):
    # Use least_squares to compute residuals, then return Gaussian log-likelihood (unit variance)
    from analysis.config_utils import get_fixed_params_from_config
    
    # Get fixed parameters from config file
    fixed_params = get_fixed_params_from_config()
    
    residual = least_squares(
        params,
        data_full=data,
        fpca_model=fpca_model,
        surrogate_model=surrogate,
        PARAMS_FIXED=fixed_params
    )
    # Flatten residual if needed
    residual = np.asarray(residual).ravel()
    # Standard Gaussian log-likelihood (unit variance, ignore constant)
    logl = -0.5 * np.sum(residual ** 2)
    return logl

ls_model = LogLikelihoodModel(
    n_parameters=3,
    log_likelihood=ls_log_likelihood,
)


# reasonable bounds for conductivities
bounds = [
    (2.8,   4.8),   # k_sample
    (7.0,   13.0),   # k_ins
    (300.0, 400)   # k_coupler
]

from UQpy.inference import MLE
from UQpy.utilities.MinimizeOptimizer import MinimizeOptimizer

optimizer = MinimizeOptimizer(bounds=bounds)

mle_solver = MLE(
    inference_model = ls_model,
    n_optimizations = 50,            # multi-start for robustness
    data = fpca_scores,
    optimizer = optimizer
)
mle_solver.run()
print(mle_solver.mle)








