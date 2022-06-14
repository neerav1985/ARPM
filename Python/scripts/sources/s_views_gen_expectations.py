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

# # s_views_gen_expectations [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_views_gen_expectations&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-example-fpgen-views).

# +
import numpy as np

from arpym.views.min_rel_entropy_sp import min_rel_entropy_sp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_views_gen_expectations-parameters)

# +
# scenarios of market variables
x = np.array([[0.2, 1.7, 2, 3.4], [5, 3.4, -1.3, 1]]).T
mu_view = np.array([1, 2.5])  # views on expectations
p_base = np.ones(x.shape[0]) / x.shape[0]  # base flexible probabilities
c = 0.2  # confidence level


def v(y):
    return np.array([y[:, 0], y[:, 0] ** 2])  # view function


# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_views_gen_expectations-implementation-step01): Compute parameters specifying the constraints

z_ineq = v(x)[:1]
mu_view_ineq = mu_view[:1]

z_eq = v(x)[-1:]
mu_view_eq = mu_view[-1:]
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_views_gen_expectations-implementation-step02): Compute covariance matrix and effective rank

# +

def eff_rank(s2):
    lam2_n, _ = np.linalg.eig(s2)
    w_n = lam2_n / np.sum(lam2_n)
    return np.exp(- w_n @ np.log(w_n))


z = np.vstack((z_ineq, z_eq))
covariance = np.cov(z)
effrank = eff_rank(np.corrcoef(z))
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_views_gen_expectations-implementation-step03): Compute updated probabilities

p_upd = min_rel_entropy_sp(p_base, z_ineq, mu_view_ineq, z_eq, mu_view_eq,
                           normalize=False)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_views_gen_expectations-implementation-step04): Compute additive/multiplicative confidence-weighted probabilities

p_c_add = c * p_upd + (1 - c) * p_base
p_c_mul = p_upd ** c * p_base ** (1 - c) /\
    np.sum(p_upd ** c * p_base ** (1 - c))
