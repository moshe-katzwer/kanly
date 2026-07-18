# """
# Generates (x,y) series for plotting normal pdfs and cdfs in
# Cartesian plane.
# """
# from __future__ import absolute_import, print_function
#
# from scipy.stats import norm
# import numpy as np
#
#
# def get_normal_pdf_x_y(mean=0., scale=1., num_points=200, num_sigma=3.5):
#     x = np.linspace(mean - num_sigma * scale, mean + num_sigma * scale, num_points)
#     y = norm.pdf(x, mean, scale)
#     return x, y
#
#
# def get_normal_pdf_x_y_from_data(data, num_points=200, num_sigma=3.5):
#     return get_normal_pdf_x_y(np.mean(data), np.std(data), num_points=num_points, num_sigma=num_sigma)
#
#
# def get_normal_cdf_x_y(mean=0., scale=1., num_points=200, num_sigma=3.5):
#     x = np.linspace(mean - num_sigma * scale, mean + num_sigma * scale, num_points)
#     y = norm.cdf(x, mean, scale)
#     return x, y
#
#
# def get_normal_cdf_x_y_from_data(data, num_points=200, num_sigma=3.5):
#     return get_normal_cdf_x_y(np.mean(data), np.std(data), num_points=num_points, num_sigma=num_sigma)
