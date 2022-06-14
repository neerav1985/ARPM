#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_bias_reduction_toy [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_bias_reduction_toy&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_bias_reduction_toy).

# +
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from arpym.statistics.simulate_normal import simulate_normal

import warnings
warnings.filterwarnings("ignore")
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_bias_reduction_toy-parameters)

mu = np.zeros(2)
sigma2 = np.array([[1, 0.5], [0.5, 1]])

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_bias_reduction_toy-implementation-step00): Generate sample

np.random.seed(1234)
xz = simulate_normal(mu, sigma2, 10**6)
xz = 10**-3 + np.exp(xz)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_bias_reduction_toy-implementation-step01): Define predictors

chi_theta = lambda theta, z: theta[0] + theta[1]*z
f_theta = lambda theta, x, z: theta*z*np.exp(-theta*x*z)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_bias_reduction_toy-implementation-step02): Compute point bias

# +
theta_point = np.array([np.exp(0.5)*(1-(np.exp(0.5)-1)/(np.exp(1)-1)), (np.exp(0.5)-1)/(np.exp(1)-1)])
print(theta_point)
err_point = np.mean((xz[:, 0]-(theta_point[0]+theta_point[1]*xz[:,1]))**2)

true_point_pred = lambda z : np.exp(0.375)*np.sqrt(z)
true_err_point = np.mean((xz[:, 0]- true_point_pred(xz[:, 1]))**2)

bias_point = err_point-true_err_point
print(bias_point)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_bias_reduction_toy-implementation-step03): Compute probabilistic bias

# +
theta_prob = np.linspace(10**-2, 5, 10**2)
err_prob = np.zeros(len(theta_prob))

for i in range(len(theta_prob)):
    err_prob[i] = np.mean(-np.log(f_theta(theta_prob[i], xz[:,0], xz[:,1])))

# Optimal theta
theta_prob_opt = theta_prob[np.argmin(err_prob)]
print(theta_prob_opt)

# Optimal error
err_prob_opt = np.min(err_prob)
bias_prob = err_prob_opt
print(bias_prob)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_bias_reduction_toy-implementation-step04): Add interactions to point prediction

# +
chi_theta_inter = lambda theta, z: theta[0] + theta[1]*z + theta[2]*z**2

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
z_inter = poly.fit_transform(xz[:, [1]])

from sklearn.linear_model import LinearRegression
reg = LinearRegression(fit_intercept=False).fit(z_inter, xz[:, 0])
theta_point_inter = reg.coef_
print(theta_point_inter)

err_point_inter = np.mean((xz[:, 0]-(theta_point_inter[0]+theta_point_inter[1]*xz[:,1]+theta_point_inter[2]*xz[:,1]**2))**2)

bias_point_inter = err_point_inter - err_point
print(bias_point_inter)
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_bias_reduction_toy-implementation-step05): Add interactions to probabilistic prediction

# +
f_theta_inter = lambda theta, x, z: (theta[0]*z + theta[1]*z**2)*np.exp(-x*(theta[0]*z + theta[1]*z**2))

f_min = lambda theta : np.mean(-np.log(f_theta_inter(theta, xz[:,0], xz[:,1])))

# Optimal theta
theta_prob_opt_inter = fmin(f_min, [0.2, -10**-3])
print(theta_prob_opt_inter)

# Optimal error
err_prob_opt_inter = f_min(theta_prob_opt_inter)

bias_prob_inter = err_prob_opt_inter
print(bias_prob_inter)
