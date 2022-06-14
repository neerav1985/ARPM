#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_intertemporal_valuation_complete [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_intertemporal_valuation_complete&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_intertemporal_valuation_complete).

# +
import numpy as np
from scipy.stats import chi, multivariate_normal, lognorm

from arpym.statistics.simulate_normal import simulate_normal
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_intertemporal_valuation_complete-parameters)

# +
n_ = 50  # number of instruments
rho = 0.7  # correlation between GBM
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_intertemporal_valuation_complete-implementation-step01): Current values and values' scenarios

# +
# current values
v_tnow = 20 * np.random.rand(n_) + 10

mu = 0.05 * np.random.rand(n_) # expectation vector
c2 = (1 - rho) * np.eye(n_) + rho * np.ones((n_, n_))  # correlation matrix
sig2 = 0.1 * np.random.rand(1) + 0.1

j_ = n_  # number of scenarios

# scenarios for value process at t=1
z_1 = simulate_normal(np.zeros(n_), c2, j_)
v_delta_1 = v_tnow * np.exp(mu + np.sqrt(sig2) * z_1)

# scenarios for value process at t=2
v_delta_2 = np.ones((n_, j_**2))
for j_1 in np.arange(j_):
    for j_2 in np.arange(j_):
        z_2 = simulate_normal(np.zeros(n_), c2, j_)
        v_delta_2[:, j_2*j_:j_*(j_2+1)]=v_delta_1[:, j_1]*np.exp(mu + np.sqrt(sig2)*z_2)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_intertemporal_valuation_complete-implementation-step02): Probabilities

# +
# probabilities at t = 1
p_1 =  np.random.uniform(0, 1, j_)
p_1 = p_1 / np.sum(p_1)

# probabilities at t = 2
p_2 = np.zeros(j_**2)
for j2 in np.arange(j_):
    p_12 = np.random.uniform(0, 1, (j_))
    p_2[j2*j_:j_*(j2+1)] = p_12*p_1[j2]
p_2 = p_2/np.sum(p_2)

# conditional probabilities
p_cond = p_2 / np.reshape(np.tile(p_1, (j_, 1)).T, j_**2)
p_cond = p_cond.reshape(j_, j_)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_intertemporal_valuation_complete-implementation-step03): Cumulative stochastic discount factor scenarios

# +
# cumulative SDF at time t=1
sdf_delta_1 = np.linalg.solve(v_delta_1*p_1, v_tnow)

# cumulative SDF at time t=2
sdf_delta_2 = np.zeros(j_**2)
for j in range(j_):
    sdf_delta_12 = np.linalg.solve(v_delta_2[:, j*j_:(j+1)*j_]*p_cond[j,:], v_delta_1[:,j])
    sdf_delta_2[j*j_:(j+1)*j_] = sdf_delta_1[j]*sdf_delta_12
