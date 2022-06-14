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

# # s_risk_neutral_density [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_risk_neutral_density&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-comprnnumsdf).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.stats import norm, lognorm

from arpym.pricing.bsm_function import bsm_function
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_risk_neutral_density-parameters)

# +
mu = 1e-3  # location parameter of lognormal distribution
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_risk_neutral_density-implementation-step00): Upload data

# +
path = '~/databases/temporary-databases/'
db_simcall = pd.read_csv(path+'db_simcall.csv', index_col=0)
k_j = db_simcall.k_j.values
s_omega_j = db_simcall.s_omega_j.values
v_call = db_simcall.v_call.values
db_tools = pd.read_csv(path+'db_simcall_tools.csv', index_col=0)
s_tnow = db_tools.s_tnow.values[0]
delta_k = db_tools.delta_s.values[0]
delta_t = db_tools.delta_t.values[0]
r = db_tools.r.values[0]
sigma2 = db_tools.sigma2.values[0]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_risk_neutral_density-implementation-step01): Compute the normalized underlying probabilities

# +
scale_p = s_tnow * np.exp((mu - sigma2 ** 2 / 2) * delta_t)
p = lognorm.cdf(k_j + 3 * delta_k / 2, sigma2*np.sqrt(delta_t), scale=scale_p) - \
    lognorm.cdf(k_j + delta_k / 2, sigma2*np.sqrt(delta_t), scale=scale_p)
p = p / np.sum(p)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_risk_neutral_density-implementation-step02): Compute risk-neutral probabilities

# +
j_ = len(k_j)  # number of scenarios (=number of basis call options)

delta2_vcall = np.zeros(j_)
for j in range(j_-2):
    delta2_vcall[j] = (v_call[j+2]-2*v_call[j+1]+v_call[j])/delta_k**2

delta2_vcall[-2] = (-2*v_call[-1] + v_call[-2])/delta_k**2
delta2_vcall[-1] = v_call[-1]/delta_k**2

p_rn = delta_k*delta2_vcall*np.exp(-delta_t*r)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_risk_neutral_density-implementation-step03): Compute pdf

# +
s_low =s_omega_j[0]
s_up = s_omega_j[-1]  # upper bound of underlying at the horizon
s_omega_j_ = np.linspace(s_low, s_up, 100000)
f_s = lognorm.pdf(s_omega_j_, sigma2 * np.sqrt(delta_t), scale=scale_p)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_risk_neutral_density-implementation-step04): Compute risk-neutral pdf

# +
scale_q = s_tnow * np.exp((r - sigma2 ** 2 / 2) * delta_t)
f_q_s = lognorm.pdf(s_omega_j_, sigma2 * np.sqrt(delta_t), scale=scale_q)
# -

# ## Plots

# +
fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
plt.style.use('arpm')

# plot histograms
plt.bar(s_omega_j, p / delta_k, width=delta_k, facecolor='none', edgecolor='b',
        label='simulated real world probability')
plt.bar(s_omega_j, p_rn / delta_k, width=delta_k, facecolor='none', edgecolor='g',
        linestyle='--', label='simulated risk-neutral probability')

# plot pdfs
plt.plot(s_omega_j_, f_s, 'b', lw=1.5, label='analytical real world pdf')
plt.plot(s_omega_j_, f_q_s, 'g--', lw=1.5, label='analytical risk-neutral pdf')
plt.xlabel('$S_{t_{\mathit{hor}}}$', fontsize = 24, labelpad=10)
plt.ylabel('pdf', fontsize = 24, labelpad=10)
plt.legend(fontsize = 23)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)

add_logo(fig, location=4, set_fig_size=False)
