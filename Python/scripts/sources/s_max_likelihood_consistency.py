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

# # s_max_likelihood_consistency [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_max_likelihood_consistency&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerMaxLikConsist).

# ## Prepare the environment

# +
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

from arpym.estimation.fit_locdisp_mlfp import fit_locdisp_mlfp
from arpym.statistics.simulate_t import simulate_t
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_max_likelihood_consistency-parameters)

t_ = 500  # number of observations
mu = 0  # location parameter
sigma2 = 2  # dispersion
nu = 3  # degrees of freedom


# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_max_likelihood_consistency-implementation-step01): Generate Student t observations

epsi = simulate_t(mu, sigma2, nu, t_)


# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_max_likelihood_consistency-implementation-step02): Compute maximum likelihood parameters

mu_ml, sigma2_ml = fit_locdisp_mlfp(epsi, nu=nu, threshold=1e-4)  # maximum likelihood estimates


# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_max_likelihood_consistency-implementation-step03): Compute maximum likelihood pdf and cdf and true pdf and cdf

sigma = np.sqrt(sigma2)
x = np.linspace(mu + sigma * t.ppf(0.01, nu), mu + sigma * t.ppf(0.99, nu), 10**5)  # equally spaced grid
sigma_ml = np.sqrt(sigma2_ml)
f_ml_eps = t.pdf((x - mu_ml) / sigma_ml, nu)  # maximum likelihood pdf
f_eps = t.pdf((x - mu) / sigma, nu) # true pdf
cdf_ml_eps = t.cdf((x - mu_ml) / sigma_ml, nu)  # maximum likelihood cdf
cdf_eps = t.cdf((x - mu) / sigma, nu)  # true cdf


# ## Plot

# +
orange = [.9, .4, .2]
b = [0, 0.5, 1]
plt.style.use('arpm')

fig, axs = plt.subplots(2)

# plot the maximum likelihood pdf
axs[0].plot(x, f_ml_eps, lw=1.5, color=orange)
plt.xlim([np.min(x), np.max(x)])
plt.ylim([0, np.max(f_ml_eps) + 0.15])

# plot the true pdf
axs[0].plot(x, f_eps, lw=1.5, color=b)
axs[0].set_xlim(x[0], x[-1])
axs[0].text(mu + 3, 0.3, 'Number of observations: '+str(t_), color='black', fontsize=12)

# Display the maximum likelihood cdf and overlay the true cdf
# plot the maximum likelihood cdf
axs[1].plot(x, cdf_ml_eps, color=orange, lw=1.5)
plt.xlim([np.min(x), np.max(x)])
plt.ylim([0, np.max(cdf_eps) + 0.15])

# plot the true cdf
axs[1].plot(x, cdf_eps, lw=1.5, color=b)
axs[1].set_xlim(x[0], x[-1])
axs[1].legend(['True','Max Likelihood'], loc='lower right')
plt.tight_layout();
# -
