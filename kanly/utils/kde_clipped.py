# TODO DEPRECATED, TO DELETE
# """
# Clips a Kernel Density to a given interval
# so that you don't get bad tail behavior where you have mass
# ouside the support
# """
# from __future__ import absolute_import, print_function
#
# import numpy as np
# from scipy.interpolate import interp1d
# from statsmodels.nonparametric.kde import KDEUnivariate
#
#
# def get_kde_clipped(data_1d, max_kde_sample_points=5_000, clip=True, num_kde_interp_points=1000, seed=0):
#     """
#     :clip: can be True (clipped to min max), False (not clipped), or a custom range
#     """
#
#     l, h = min(data_1d), max(data_1d)
#
#     rand = np.random.RandomState(seed)
#     if len(data_1d) > max_kde_sample_points:
#         data_1d = rand.choice(data_1d, max_kde_sample_points, replace=False)
#
#     kde_raw = KDEUnivariate(data_1d)
#     kde_raw.fit()
#
#     if clip is None:
#         pass
#     elif isinstance(clip, bool):
#         if clip:
#             clip = (l, h)
#         else:
#             clip = None
#     else:
#         assert np.shape(clip) == (2,)
#         assert clip[0] < clip[1]
#
#     # data_1d = np.asarray(data_1d).flatten()
#
#     diff = h - l
#     a = 1e-6
#     while kde_raw.evaluate(l) > 1e-8:
#         l -= a * diff
#         a *= 1.25
#     a = 1e-6
#     while kde_raw.evaluate(h) > 1e-8:
#         h += a * diff
#         a *= 1.25
#
#     if clip is not None:
#         l = clip[0]
#     if clip is not None:
#         h = clip[1]
#
#     rng = np.linspace(l, h, num_kde_interp_points)
#
#     density_new = kde_raw.evaluate(rng)
#     if clip is not None:
#         if clip[0] != -np.inf:
#             density_new += kde_raw.evaluate(2 * clip[0] - rng)
#         if clip[1] != np.inf:
#             density_new += kde_raw.evaluate(2 * clip[1] - rng)
#
#     pdf = interp1d(rng, density_new, fill_value=0.0, bounds_error=False)
#
#     return pdf, l, h
#
#
# def get_kde_clipped_series(data_1d, plot_points=100, max_kde_sample_points=5_000, clip=True, num_kde_interp_points=1000,
#                            seed=0):
#     pdf, l, h = get_kde_clipped(data_1d, max_kde_sample_points=max_kde_sample_points, clip=clip,
#                                 num_kde_interp_points=num_kde_interp_points, seed=seed)
#     xrng = np.linspace(l, h, plot_points)
#     return xrng, pdf(xrng), pdf, l, h
