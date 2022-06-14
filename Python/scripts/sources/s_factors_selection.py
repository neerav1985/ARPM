#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_factors_selection [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_factors_selection&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-lfmselect).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.statistics.objective_r2 import objective_r2
from arpym.tools.naive_selection import naive_selection
from arpym.tools.forward_selection import forward_selection
from arpym.tools.backward_selection import backward_selection
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_factors_selection-parameters)

n_ = 1  # target variable dimension
m_ = 50  # number of factors

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_factors_selection-implementation-step01): Generate random positive definite matrix

sigma = np.random.randn(n_ + m_, n_ + m_ + 1)
sigma2_xz = sigma @ sigma.T / (n_ + m_ + 1)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_factors_selection-implementation-step02): Setup objective function

def g(s_k):
    all = objective_r2(s_k, sigma2_xz, n_)
    return all[0], all[1]

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_factors_selection-implementation-step03): Compute the r2 based on number of factors via naive, stepwise forward and backward routines

s_k_naive, g_s_star_naive, _ = naive_selection(g, m_)
s_k_fwd, g_s_star_fwd, _ = forward_selection(g, m_)
s_k_bwd, g_s_star_bwd, _ = backward_selection(g, m_)

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure()
plt.plot(np.abs(g_s_star_naive))
plt.plot(np.abs(g_s_star_fwd), color='red')
plt.plot(np.abs(g_s_star_bwd), color='blue')
plt.legend(['naive', 'forward stepwise', 'backward stepwise'])
plt.xlabel('number of risk factors Z')
plt.ylabel('R-square')
plt.title('n-choose-k routines comparison')
add_logo(fig)
