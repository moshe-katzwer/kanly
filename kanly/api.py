"""
Static references to key regression methods
"""
from __future__ import absolute_import, print_function

from kanly.automatic_differentiation.function_callable import func_str_to_callable
from kanly.bayes.bayesian_linear_regression_conjugate_prior.bayesian_linear_regression_analytical \
    import BayesianLinearRegressionConjugatePriorModel
from kanly.bayes.bayesian_model import BayesianModel
from kanly.bayes.bayesian_regression_model import BayesianLinearModel, BayesianGeneralizedLinearModel, \
    BayesianNonlinearLeastSquaresModel
from kanly.bayes.data_model import DataModel
from kanly.bayes.mcmc.adaptive_metropolis.adaptive_metropolis_mcmc import amha
from kanly.bayes.mcmc.coordinate_mala.coordinate_mala_mcmc import mala
from kanly.bootstrap.bootstrap import get_joint_bootstrapped_distribution
from kanly.dill_object import read, save
from kanly.formula.add_constant import add_constant
from kanly.formula.data_getter import SparseDataGetter
from kanly.general_models.general_model_fit import fit_general_model_callable
from kanly.nonparametric.gaussian_kernel_smooth import gaussian_kernel_smooth, gaussian_kernel_smooth_direct, \
    gaussian_kernel_smooth_fft
from kanly.nonparametric.interpolate import interp, cubic_spline, linear_spline, quadratic_spline
from kanly.nonparametric.kde import kde
from kanly.nonparametric.lowess import LOWESS, lowess
from kanly.nonparametric.mstl import mstl, plot_mstl
from kanly.nonparametric.stl import stl, plot_stl
from kanly.optimize.bfgs_bounded_quasi_newton import bfgs_pqn
from kanly.optimize.coordinate_descent_bounded import cdb
from kanly.plot.ascii_plotlib import plot, scatter, hist
from kanly.regression.generalized_linear_models.model import SparseGeneralizedLinearModel
from kanly.regression.generalized_method_of_moments.model import SparseGeneralizedMethodOfMomentsModel
from kanly.regression.linear_models.model import SparseLinearModel
from kanly.regression.partial_least_squares.pls import PLS1, pls1, PLS2
from kanly.regression.linear_models.penalized.model import SparsePenalizedLinearModel
from kanly.regression.linear_models.quantile_regression.model import SparseQuantileRegressionModel
from kanly.regression.linear_models.robust.model import SparseRobustLinearModel
from kanly.regression.linear_models.shapley import Shapley
from kanly.regression.generalized_linear_models.generalized_additive_models.model import SparseGeneralizedAdditiveModel
from kanly.regression.nonlinear_least_squares.formula.sparse_nonlinear_formula_parser import \
    build_prediction_function_from_formula
from kanly.regression.nonlinear_least_squares.model import SparseNonlinearLeastSquaresModel
from kanly.regression.nonlinear_least_squares.optimize.nlls_minimize_internal import nlls_minimize_internal
from kanly.stats.distributions.fit_distributions_mle import get_normal_cdf_x_y_from_data, get_normal_pdf_x_y_from_data, \
    get_normal_cdf_x_y, get_normal_pdf_x_y, get_mle_x_y, get_mle_distribution
from kanly.stats.distributions.nopython_logpdf import (
    logpdf_beta, logpdf_cauchy, logpdf_chi2, logpdf_expon, logpdf_f, logpdf_gamma, logpdf_genextreme, logpdf_halfcauchy,
    logpdf_halfnorm, logpdf_invgamma, logpdf_laplace, logpdf_logistic, logpdf_lognorm, logpdf_multivariate_normal,
    logpdf_multivariate_t, logpdf_norm, logpdf_pareto, logpdf_t, logpdf_truncnorm, logpdf_weibull_min,
    pdf_beta, pdf_cauchy, pdf_chi2, pdf_expon, pdf_f, pdf_gamma, pdf_genextreme, pdf_halfcauchy, pdf_halfnorm,
    pdf_invgamma, pdf_laplace, pdf_logistic, pdf_lognorm, pdf_multivariate_normal, pdf_multivariate_t, pdf_norm,
    pdf_pareto, pdf_t, pdf_truncnorm, pdf_weibull_min,
)
from kanly.stats.distributions.nopython_scipy_special import (
    gammaln, betaln, erf, ndtr, gamma, beta, erfc
)
from kanly.time_series.auto_correlation_function import (
    auto_correlation_function, partial_autocorrelation_function, autocovariance_function)
from kanly.time_series.autoregression.estimate_ar import yule_walker, css, estimate_ar, burg, mle as ar_mle
from kanly.time_series.autoregression.autoreg import autoreg, AUTOREG, ARDL, ardl
from kanly.time_series.sarimax.model import SarimaxModel
from kanly.time_series.sarimax.simulate import simulate_sarima
from kanly.utils.compare_results import compare_results
from kanly.utils.highest_density_interval import get_highest_density_interval
from kanly.utils.latex_table import latex_table
from kanly.utils.timer import timer, clear_timers
from kanly.sparse_data_frame import SparseDataFrame


# ################################# #
# GENERAL PURPOSE LINEAR REGRESSION #
# ################################# #

lm = SparseLinearModel.lm
LM = SparseLinearModel.LM

lm_model = SparseLinearModel.build_model_from_formula
LM_model = SparseLinearModel

lm_fast = SparseLinearModel.lm_fast
LM_fast = SparseLinearModel.LM_fast

linear_model_from_formula = SparseLinearModel.build_model_from_formula

# ############################### #
# SEEMINGLY UNRELATED REGRESSIONS #
# ############################### #

sure = SparseLinearModel.sure

# ####################### #
# NONLINEAR LEAST SQUARES #
# ####################### #

nlls = SparseNonlinearLeastSquaresModel.nlls
nlls_en = SparseNonlinearLeastSquaresModel.nlls_en

NLLS = SparseNonlinearLeastSquaresModel.NLLS
NLLS_EN = SparseNonlinearLeastSquaresModel.NLLS_EN

nlls_model = SparseNonlinearLeastSquaresModel.build_model_from_formula
build_prediction_function_from_formula = build_prediction_function_from_formula

nlls_minimize_internal = nlls_minimize_internal

# ############################# #
# GENERALIZED METHOD OF MOMENTS #
# ############################# #

gmm = SparseGeneralizedMethodOfMomentsModel.gmm
gmm_iv_linear = SparseGeneralizedMethodOfMomentsModel.gmm_iv_linear
gmm_iv_nonlinear = SparseGeneralizedMethodOfMomentsModel.gmm_iv_nonlinear
gmm_mle = SparseGeneralizedMethodOfMomentsModel.gmm_mle

GMM = SparseGeneralizedMethodOfMomentsModel.GMM

# ############################ #
# PENALIZED LINEAR REGRESSIONS #
# ############################ #

elastic_net = SparsePenalizedLinearModel.elastic_net
en = elastic_net
ELASTIC_NET = SparsePenalizedLinearModel.ELASTIC_NET
EN = ELASTIC_NET

# ######################### #
# GENERALIZED LINEAR MODELS #
# ######################### #

glm = SparseGeneralizedLinearModel.glm
GLM = SparseGeneralizedLinearModel.GLM

# ########################### #
# GENERALIZED ADDITIVE MODELS #
# ########################### #

gam = SparseGeneralizedAdditiveModel.gam
GAM = SparseGeneralizedAdditiveModel.GAM

#######################
# QUANTILE REGRESSION #
#######################

qr = SparseQuantileRegressionModel.qr
QR = SparseQuantileRegressionModel.QR

# ################################ #
# ROBUST REGRESSION (M-ESTIMATION) #
# ################################ #

rlm = SparseRobustLinearModel.rlm
RLM = SparseRobustLinearModel.RLM

# ############## #
# SHAPLEY VALUES #
# ############## #

shapley_value = Shapley.shapley_value

# ############## #
# OUTPUT SUMMARY #
# ############## #

compare_results = compare_results
latex_table = latex_table

# ####################
# Fit general models #
######################

fit_general_model_callable = fit_general_model_callable

# ########################################## #
# Build a sparse matrix from a patsy formula #
# ########################################## #

sparse_dmatrix = SparseDataGetter.sparse_dmatrix
sparse_dmatrices = SparseDataGetter.sparse_dmatrices
add_constant = add_constant

# ###################################################### #
# Build a joint distribution across different estimators #
# that were bootstapped identically.                     #
# ###################################################### #

get_joint_bootstrapped_distribution = get_joint_bootstrapped_distribution

# ################################## #
# Bayesian MCMC                      #
# ################################## #

BayesianModel = BayesianModel
bmodel = BayesianModel  # Model object

DataModel = DataModel
build_data_model = DataModel.build_data_model

amha = amha
mala = mala

# ################### #
# Bayesian Regression #
# ################### #

blm = BayesianLinearRegressionConjugatePriorModel.blm
BLM = BayesianLinearRegressionConjugatePriorModel.BLM

bayes_nlls_model = BayesianNonlinearLeastSquaresModel.build_model_from_formula
bayes_lm_model = BayesianLinearModel.build_model_from_formula
bayes_glm_model = BayesianGeneralizedLinearModel.build_model_from_formula

# ############ #
# OPTIMIZATION #
# ############ #

cdb = cdb
bfgs_pqn = bfgs_pqn
func_str_to_callable = func_str_to_callable

# ########### #
# TIME SERIES #
# ########### #

acf = auto_correlation_function
pacf = partial_autocorrelation_function
autocovariance_function = autocovariance_function

sarimax = SarimaxModel.sarimax
bsarimax = SarimaxModel.bsarimax
SARIMAX = SarimaxModel.SARIMAX
simulate_sarima = simulate_sarima

css = css
yule_walker = yule_walker
ar_mle = ar_mle
burg = burg
estimate_ar = estimate_ar

glsar = SparseLinearModel.glsar
GLSAR = SparseLinearModel.GLSAR

ardl = ardl
ARDL = ARDL

autoreg = autoreg
AUTOREG = AUTOREG

# ############ #
# LOAD OBJECTS #
# ############ #

read = read
save = save

# ######################## #
# Highest Density Interval #
# ######################## #

get_highest_density_interval = get_highest_density_interval

# ######################### #
# ASCII TEXT PLOT OF SERIES #
# ######################### #

plot = plot
scatter = scatter
hist = hist

# ################################### #
# DISTRIBUTIONS AND SPECIAL FUNCTIONS #
# NUMBA-COMPATIBLE                    #
# ################################### #

logpdf_beta = logpdf_beta
logpdf_cauchy = logpdf_cauchy
logpdf_chi2 = logpdf_chi2
logpdf_expon = logpdf_expon
logpdf_f = logpdf_f
logpdf_gamma = logpdf_gamma
logpdf_genextreme = logpdf_genextreme
logpdf_halfcauchy = logpdf_halfcauchy
logpdf_halfnorm = logpdf_halfnorm
logpdf_invgamma = logpdf_invgamma
logpdf_laplace = logpdf_laplace
logpdf_logistic = logpdf_logistic
logpdf_lognorm = logpdf_lognorm
logpdf_multivariate_normal = logpdf_multivariate_normal
logpdf_multivariate_t = logpdf_multivariate_t
logpdf_norm = logpdf_norm
logpdf_pareto = logpdf_pareto
logpdf_t = logpdf_t
logpdf_truncnorm = logpdf_truncnorm
logpdf_weibull_min = logpdf_weibull_min

pdf_beta = pdf_beta
pdf_cauchy = pdf_cauchy
pdf_chi2 = pdf_chi2
pdf_expon = pdf_expon
pdf_f = pdf_f
pdf_gamma = pdf_gamma
pdf_genextreme = pdf_genextreme
pdf_halfcauchy = pdf_halfcauchy
pdf_halfnorm = pdf_halfnorm
pdf_invgamma = pdf_invgamma
pdf_laplace = pdf_laplace
pdf_logistic = pdf_logistic
pdf_lognorm = pdf_lognorm
pdf_multivariate_normal = pdf_multivariate_normal
pdf_multivariate_t = pdf_multivariate_t
pdf_norm = pdf_norm
pdf_pareto = pdf_pareto
pdf_t = pdf_t
pdf_truncnorm = pdf_truncnorm
pdf_weibull_min = pdf_weibull_min

gammaln = gammaln
betaln = betaln
erf = erf
erfc = erfc
ndtr = ndtr
gamma = gamma
beta = beta

# ############################################ #
# LOCAL REGRESSION / SMOOTHING / NONPARAMETRIC #
# ############################################ #

LOWESS = LOWESS
lowess = lowess
kde = kde
interp = interp
cubic_spline = cubic_spline
quadratic_spline = quadratic_spline
linear_spline = linear_spline
stl = stl
plot_stl = plot_stl
mstl = mstl
plot_mstl = plot_mstl
gaussian_kernel_smooth = gaussian_kernel_smooth
gaussian_kernel_smooth_direct = gaussian_kernel_smooth_direct
gaussian_kernel_smooth_fft = gaussian_kernel_smooth_fft

# ####### #
# ALIASES #
# ####### #

reg = lm
REG = LM

OLS = LM
ols = lm

WLS = LM
wls = lm

QUANTREG = QR
quantreg = qr

ARIMA = SARIMAX
arima = sarimax

bfgs = bfgs_pqn
interp1d = interp

# ######################################### #
# Plotting help of distributions given data #
# ######################################### #

get_normal_pdf_x_y_from_data
get_normal_cdf_x_y_from_data
get_normal_pdf_x_y
get_normal_cdf_x_y
get_mle_x_y
get_mle_distribution

# ###############################
# # Generalized Additive Models #
# ###############################
#
# gam
# GAM


# ##################### #
# Partial Least Squares #
# ##################### #

pls1
PLS1
PLS2

#########
# TIMER #
#########

timer = timer
clear_timers = clear_timers

# ################# #
# SPARSE DATA FRAME #
# ################# #

SparseDataFrame = SparseDataFrame
