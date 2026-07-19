r"""
================================================================================
Ridge regression one way, many estimators (this example file)
================================================================================

We simulate a weighted linear data-generating process

    y = 3 + 10·x + ε,   ε ~ N(0, 3²),

with an extra regressor ``z`` that is **in the design matrix but not in the DGP**
(so its coefficient should be shrunk toward 0 when penalized — here it is
**left unpenalized** on purpose).  Observation weights ``wts`` are strictly
positive random scalars, so every path below is **weighted** least squares /
weighted likelihood unless noted.

The **only** coefficient we deliberately L2-regularize is the slope on ``x``,
with a **nominal** scalar ``l2_penalty_x = 3.3``.  Intercept and ``z`` stay
unpenalized.  Different APIs package that single ridge idea in different
parameterizations; the choices in this script (e.g. multiplying by
``sum(wts)`` in ``blm`` and using ``normalize=False`` where applicable) are
picked so the implied **strength of shrinkage on** ``β_x`` is aligned across
methods as closely as the implementations allow.  ``compare_results`` at the
bottom overlays the fitted coefficients to verify they agree up to numerical /
Monte Carlo error.

**``normalize=False`` in this script:** Both ``lm`` (``ridge_kwds``) and
``elastic_net`` default to ``normalize=True``, which scales each penalty by the
**standard deviation** of the corresponding predictor column (weighted population
std when weights are present) — the same scale-invariance goal as sklearn's
``StandardScaler`` workflow.  Older kanly (and legacy sklearn
``ElasticNet(normalize=True)``) used the **L2 norm of demeaned columns** instead;
that path is retained internally only via ``OLD_SKLEARN_NORMALIZATION`` in
``sparse_elastic_net_internal``.  Here we set ``normalize=False`` on purpose so
the same nominal ``l2_penalty_x`` is passed through without an extra per-column
std scaling layer, which makes cross-API comparison easier.

--------------------------------------------------------------------------------
Common weighted linear model (conceptual)
--------------------------------------------------------------------------------

Design matrix ``X`` has columns ``[1, x, z]``.  Let ``W = diag(wts)``.  Plain WLS
minimizes the **weighted** residual sum of squares

    RSS(β) = (y − Xβ)ᵀ W (y − Xβ) = ‖W^{1/2}(y − Xβ)‖²₂          (no extra ½)

Ridge WLS **adds a quadratic penalty on selected coordinates**.  In this file,
only ``β_x`` is penalized.  Abstractly you want something of the form

    RSS(β)  +  λ_x · β_x²

or, equivalently, ``+ ½ κ β_x²`` depending on whether the software folds a
``½`` into the penalty or keeps it only in the first-order conditions.  The
implementations below differ in that bookkeeping **and** in whether ``λ`` is
entered as a user-facing ``alpha`` that is later multiplied by ``∑ w_i`` for
weighted problems.

--------------------------------------------------------------------------------
1) ``lm`` + ``ridge_kwds``  —  closed-form ridge WLS
--------------------------------------------------------------------------------

**Call:** ``lm('y ~ x + z $ wts', data, ridge_kwds={...}, ...)``

**Mechanism:** Sparse linear model machinery solves the normal equations with a
diagonal **Tikhonov** term on the Gram matrix:

    ( Xᵀ W X + D ) β̂ = Xᵀ W y,

where ``D`` is diagonal, ``D_{jj} = λ_j`` from ``_get_ridge_parameters``.  Here
``ridge_kwds['alpha']`` is a **dict** ``{'x': l2_penalty_x}`` (zeros implied for
other columns), ``normalize=False``, and default ``penalize_intercept=False`` so
the intercept column’s penalty is forced to **0**.

With ``normalize=True`` (the default), kanly would multiply each ``α_j`` by
``std(x_j)²`` via the shared ``_get_normalizing_factors`` helper (column
standard deviation, not the legacy demeaned L2 norm).  This script sets
``normalize=False`` so ``l2_penalty_x`` is used as-is before weight scaling.

For **weighted** models with ``normalize=False``, kanly **multiplies** the
user-supplied ``alpha`` values by ``sum(wts)`` so the ridge term scales with
total weight like the rest of the WLS objective_function.  So the effective diagonal
penalty on ``β_x`` is

    λ_x  ≈  l2_penalty_x · (∑ w_i).

**Implied objective_function (up to convention):** minimize

    (y − Xβ)ᵀ W (y − Xβ)  +  Σ_j λ_j β_j²,

with ``λ_{Intercept}=λ_z=0`` and ``λ_x`` as above.

This is the reference **frequentist ridge** solution in closed form.

--------------------------------------------------------------------------------
2) ``elastic_net`` with ``l1_ratio=0``  —  pure L2 / “elastic net without L1”
--------------------------------------------------------------------------------

**Call:** ``elastic_net(..., alpha={'x': l2_penalty_x}, l1_ratio=0, normalize=False, ...)``

**Mechanism:** Coordinate descent on the **elastic-net** objective_function with the L1
share set to **zero**, which collapses the penalty to **pure ridge** on the
coordinates with positive ``alpha``.  Internally (see
``_get_penalties`` in ``sparse_elastic_net_internal``), with ``normalize=False``,

    L2 penalty contribution = Σ_j  ( ½ · α_j · (1 − l1_ratio_j) · β_j² ).

With ``l1_ratio = 0`` this is ``½ α_x β_x²`` for the penalized ``x`` column only
(others have ``α=0``).  So the **elastic-net** code path uses an explicit ``½``
on the ridge term, whereas ``lm``’s diagonal uses **unsplit** ``λ`` in
``(XᵀWX + D)``.  The numeric ``alpha`` dict is chosen so that, together with
weight handling inside the elastic-net objective_function, the **fitted** ``β`` matches
the ``lm`` ridge solution for this toy setup.

With ``normalize=True`` (the default), each ``α_j`` would be multiplied by
``std(x_j)`` for the L1 part and ``std(x_j)²`` for the L2 part before the
``½ (1 − l1_ratio)`` split — matching sklearn's recommended ``StandardScaler``
scale-invariance.  This example uses ``normalize=False`` for the same reason as
``lm`` above.

**Implied objective_function:** weighted least squares **plus** ``½ α_x β_x²`` (and
analogous zeros elsewhere).

--------------------------------------------------------------------------------
3) ``blm`` — conjugate **Bayesian** linear regression (Normal–Inverse-Gamma)
--------------------------------------------------------------------------------

**Call:** ``blm('y ~ x + z $ wts', data, Lambda0=diag([0, l2_penalty_x·∑w, 0]), mu0=0, a0=0, b0=0, ...)``

**Mechanism:** ``BayesianLinearRegressionConjugatePriorModel`` with a
multivariate Normal prior on **coefficients** (conditional on ``σ²`` in the
NIG hierarchy) encoded by prior **precision** matrix ``Lambda0`` and prior mean
``mu0``.  Here ``Lambda0`` is diagonal: **no** prior precision on the intercept or
``z`` (entries ``0`` — improper / infinitely diffuse in those directions), and
prior precision ``l2_penalty_x · ∑ w_i`` on the ``x`` slope — matching the same
``∑ w`` scaling used in ``lm``’s ridge dict for weighted data.

``a0 = b0 = 0`` is an **improper** inverse-gamma tail on ``σ²`` (very vague scale);
combined with the Gaussian prior on ``β | σ²`` this is the standard conjugate
route to shrinkage that parallels ridge **when the prior precision on** ``β_x``
**is chosen to mirror the frequentist penalty**.

**Implied target:** not a single “loss function” to minimize, but a **posterior**
under the NIG; the reported point estimates (e.g. posterior means under the
fitted model) are the Bayesian counterparts to the penalized WLS point estimates
above when the prior is calibrated the same way.

--------------------------------------------------------------------------------
4) ``nlls`` — nonlinear least squares API on a **linear** mean function
--------------------------------------------------------------------------------

**Call:** ``nlls('[y] ~ {Intercept} + {x}*[x] + {z}*[z] $ [wts]', data,
            l2_penalties={'x': l2_penalty_x}, scale_l2_penalties=True, ...)``

**Mechanism:** The formula is still **linear in the parameters** ``Intercept``,
``x``, ``z``, but it is parsed through the NLLS stack so the objective_function is the
usual **sum of squared (possibly weighted) residuals** plus an explicit **L2**
penalty block.  With ``scale_l2_penalties=True``, the per-parameter L2 weights
are **multiplied by** ``∑ w_i`` in the weighted case (see
``nlls_minimize_internal``), paralleling ``lm``’s ridge scaling.

The optimizer minimizes (schematically)

    ½ ‖√W · r(β)‖²₂  +  ½ Σ_j λ_j β_j²

(with ``λ_x`` coming from ``l2_penalties`` after scaling, other coordinates
zero), using analytic Jacobians of the mean with respect to parameters.

**Why include it:** Shows that the same ridge answer can be obtained from the
**generic NLLS + Tikhonov** route, not only from ``lm`` / ``elastic_net``.

--------------------------------------------------------------------------------
5) ``nlls_en`` — same mean, **elastic-net / coordinate-descent** optimizer
--------------------------------------------------------------------------------

**Call:** ``nlls_en(..., alpha={'x': l2_penalty_x}, l1_ratio=0, scale_penalties=True, ...)``

**Mechanism:** Same structural model as ``nlls``, but the **elastic-net style**
coordinate-descent backend (``l1_ratio=0`` ⇒ no L1 part) with ``scale_penalties=True``.
Tighter tolerances ``xtol``, ``gtol``, ``ftol`` are set only to drive the CD
solver to a tight optimum.

**Implied objective_function:** same family as (4) / (2): weighted SSR plus pure L2 on
``β_x``, with penalty bookkeeping consistent with the EN_sk-style ``½ α (1−r)`` split
and weight scaling flags.

--------------------------------------------------------------------------------
6) ``BayesianLinearModel`` + ``amha`` — **MCMC** with a hand-specified ridge-like log-prior
--------------------------------------------------------------------------------

**Call:** ``BayesianLinearModel.build_model_from_formula(..., priors={'': <string>}, ...).amha(...)``

**Mechanism:** This is **not** the closed-form conjugate ``blm`` path.  The
``priors`` string adds a custom log-density fragment including a term

    − (∑ w_i · l2_penalty_x) / (2 · σ²) · β_x²

which is the log of a **Normal(0, σ² / (∑ w_i · l2_penalty_x))** kernel on the
``x`` coefficient in a Gaussian linear model — i.e. a **ridge-like conditional
Gaussian prior** on ``β_x`` given ``σ²``, aligned with the same ``∑ w`` and
``l2_penalty_x`` scaling as elsewhere.  The additional sparse_terms in ``-3/2·log σ² −
log σ²`` shape a **vague** prior on the error variance (related to Jeffreys-type
reference ideas; not identical to ``blm``’s ``a0=b0=0`` NIG marginal).

**Sampling:** ``amha`` runs an adaptive Metropolis–Hastings / tempering-style
MCMC chain.  The object in ``compare_results`` is therefore summarized MCMC
output (e.g. posterior means / medians depending on the results class), **not**
the same closed-form posterior mean as ``blm``.

**Implied target:** maximize **posterior density** (MAP) in the limit of zero
step size, or more generally explore the full posterior; finite ``n_samples``
means Monte Carlo error remains.

--------------------------------------------------------------------------------
Bottom line
--------------------------------------------------------------------------------

* **Frequentist ridge point estimates:** ``lm`` (reference normal equations),
  ``elastic_net`` (``l1_ratio=0``), ``nlls``, ``nlls_en`` — different solvers /
  parameterizations of **penalized weighted least squares** with L2 only on
  ``β_x``.

* **Bayesian analogues:** ``blm`` — **conjugate** NIG with a precision prior
  matching the ridge structure; ``BayesianLinearModel`` + ``amha`` —
  **non-conjugate MCMC** with an explicitly coded ridge-like Normal kernel on
  ``β_x | σ²`` plus a diffuse prior on ``σ²``.

Adjust ``l2_penalty_x`` or weight scaling flags if you port this pattern to
another dataset and need bit-for-bit alignment across APIs.  On a new dataset,
if you want **scale-invariant** ridge (default behavior), omit
``normalize=False`` and interpret ``alpha`` relative to column standard
deviations; use this script's ``normalize=False`` pattern only when you need a
fixed nominal penalty on the native coefficient scale.
================================================================================
"""
from kanly.api import lm, elastic_net, nlls, nlls_en, blm, compare_results
from kanly.bayes.bayesian_linear_regression_conjugate_prior.bayesian_linear_regression_analytical \
    import BayesianLinearRegressionConjugatePriorModel
from kanly.bayes.bayesian_regression_model import BayesianLinearModel
from kanly.bayes.bayesian_linear_regression_conjugate_prior.normal_inverse_gamma import NormalInverseGamma
import numpy as np

np.random.seed(0)
n = 50
x = np.random.randn(n)
z = np.random.rand(n)
y = 3 + 10 * x + np.random.randn(n) * 3
wts = .01 + np.random.rand(n)
data = {'x': x, 'y': y, 'z': z, 'wts': wts}

l2_penalty_x = 3.3

fit_lm = lm('y ~ x + z $ wts', data,
            ridge_kwds={'alpha': {'x': l2_penalty_x},
                        # normalize=False: skip std-dev scaling (default True)
                        'normalize': False,
                        },
            specification_name='LM')

fit_en = elastic_net('y ~ x + z $ wts', data,
                     alpha={'x': l2_penalty_x}, l1_ratio=0,
                     # normalize=False: same nominal alpha as lm ridge above
                     normalize=False, specification_name='Elastic Net')

# Bayesian with Normal Inverse Gamma Conjugate Prior
fit_blm = blm('y ~ x + z $ wts', data,
              Lambda0=np.diag([0, l2_penalty_x * np.sum(wts), 0]), mu0=np.zeros(3),
              a0=0, b0=0,
              specification_name='BLM')

fit_nlls = nlls('[y] ~ {Intercept} + {x}*[x] + {z}*[z] $ [wts]', data,
                l2_penalties={'x': l2_penalty_x}, scale_l2_penalties=True, specification_name='NLLS')

fit_nlls_en = nlls_en('[y] ~ {Intercept} + {x}*[x] + {z}*[z] $ [wts]', data,
                      alpha={'x': l2_penalty_x}, l1_ratio=0, scale_penalties=True,
                      xtol=1e-10, gtol=1e-6, ftol=1e-20,
                      specification_name='NLLS Coord Descent')

fit_mcmc = BayesianLinearModel.build_model_from_formula(
    'y ~ x + z $ wts', data,

    # log pdf, jeffrey's prior on scale, normal marginal for beta
    priors={'': f'-{sum(wts) * l2_penalty_x}/(2 * {{__sigma2}}) * {{x}}**2 '
                f'- {3 / 2} * np.log({{__sigma2}}) - np.log({{__sigma2}})'
            },
    specification_name='MCMC'
).amha(start_params=[2, 3, 3, 40], n_samples=100_000, n_burnin=10_000, n_chains=8, thinning=4,
       max_subchain_draws_sample=50_000,
       # debug=True, user_prompt_for_more_iters=True
       )

compare_results([fit_lm, fit_en, fit_blm, fit_nlls, fit_nlls_en, fit_mcmc],
                show_bse=False, print_result=True)
