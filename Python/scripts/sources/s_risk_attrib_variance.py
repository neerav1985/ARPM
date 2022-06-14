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

# # s_risk_attrib_variance [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_risk_attrib_variance&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExBetsPCAandTors).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from arpym.portfolio.effective_num_bets import effective_num_bets
from arpym.portfolio.minimum_torsion import minimum_torsion
from arpym.tools.transpose_square_root import transpose_square_root
from arpym.tools.logo import add_logo
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attrib_variance-implementation-step00): Load data

# +
# (generated by script s_risk_attribution_norm)
path = '~/databases/temporary-databases/'
db = pd.read_csv(path + 'db_risk_attribution_normal.csv')

k_ = int(np.array(db['k_'].iloc[0]))
beta = np.array(db['beta_new'].iloc[:k_+1]).reshape(-1)
mu_z = np.array(db['mu'].iloc[:k_+1]).reshape(-1)
sig2_z = np.array(db['sigma2_z'].iloc[:(k_+1)*(k_+1)]).\
            reshape((k_+1, k_+1))
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attrib_variance-implementation-step01): Principal components decomposition of the covariance matrix

lam, e = np.linalg.eig(sig2_z)  # eigenvectors of factors' covariance matrix
flip = (e[2] < 0)
for j in range(len(flip)):
    e[:, flip] = -e[:, flip]
index = np.argsort(lam)[::-1]
e = e[:, index]
lambda_sort = np.sort(lam)[::-1]

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attrib_variance-implementation-step02): Principal components exposures

beta_pc = beta @ e

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attrib_variance-implementation-step03): Computation of the marginal contributions in the principal components framework

satis_pc = np.zeros((1, k_ + 1))
for k in range(k_+1):
    # variance of marginal principal components bets
    var_zk = e[:, k].T @ sig2_z@ e[:, k]
    # principal components marginal contributions
    satis_pc[0, k] = (beta_pc[k] ** 2) * var_zk

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attrib_variance-implementation-step04): Princ. comp. diversification distr. and the effective numb. of bets

[enb_pc, p_pc] = effective_num_bets(beta, sig2_z, e.T)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attrib_variance-implementation-step05): Minimum-torsion transformation

t_mt = minimum_torsion(sig2_z)

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attrib_variance-implementation-step06): Minimum-torsion exposures

beta_mt = beta.dot(np.linalg.solve(t_mt, np.eye(k_ + 1)))

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attrib_variance-implementation-step07): Marginal contributions in the minimum-torsions framework

satis_mt = np.zeros((1, k_ + 1))
satis_mt1111 = np.zeros((1, k_ + 1))
sig2_zmt = t_mt.T @ sig2_z @ t_mt
for k in range(k_+1):
    # variance of marginal minimum-torsion bets
    var_zk = t_mt[k, :] @ sig2_z@ t_mt[k, :].T
    # minimum-torsion marginal contributions
    satis_mt[0, k] = (beta_mt[k] ** 2) * var_zk

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attrib_variance-implementation-step08): Min-tors. diversification distr. and the effective num. of bets

[enb_mt, p_mt] = effective_num_bets(beta, sig2_z, t_mt.T)

# ## [Step 9](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attrib_variance-implementation-step09): Pure and spurious contributions in the marginal contribution corresponding to Z_1

# pure components of Euler's marginal contributions
pure = (beta[1] ** 2)*sig2_z[1, 1]
# correlation matrix of factors
corr_z0z1 = sig2_z[0, 1]/np.sqrt(sig2_z[0, 0]*sig2_z[1, 1])
corr_z1z2 = sig2_z[1, 2]/np.sqrt(sig2_z[1, 1]*sig2_z[2, 2])
# spurious components of Euler's marginal contributions
spurious =  beta[1]*np.sqrt(sig2_z[1, 1])*\
            (corr_z0z1*beta[0]*np.sqrt(sig2_z[0, 0]) +
             corr_z1z2*beta[2]*np.sqrt(sig2_z[2, 2]))

# ## [Step 10](https://www.arpm.co/lab/redirect.php?permalink=s_risk_attrib_variance-implementation-step10): Relative marginal contributions

m = beta.T * (sig2_z @ beta.T) / (beta @ sig2_z @ beta.T)

# ## Plots

# +
plt.style.use('arpm')

corr = np.diagflat(np.ones(3)/np.sqrt(np.diag(sig2_z)))@sig2_z @\
        np.diagflat(np.ones(3) / np.sqrt(np.diag(sig2_z)))
z1 = transpose_square_root(corr)
z = z1@np.diagflat(np.sqrt(np.diag(sig2_z)))  # original factors
z_pc = e @ z  # principal components factors
z_mt = t_mt @ z  # minimum-torsion factors

starts = np.tile(mu_z.reshape((3, 1)), (1, 3))
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
ax.view_init(36, -56)
factors_original = ax.quiver(starts[0], starts[1], starts[2], z[0], z[1],
                             z[2], color='g', lw=2, length=0.003)
factors_pca = ax.quiver(starts[0], starts[1], starts[2], z_pc[0], z_pc[1],
                        z_pc[2], color='m', lw=2, length=0.003)
factors_mt = ax.quiver(starts[0], starts[1], starts[2], z_mt[0], z_mt[1],
                       z_mt[2], color='b', lw=2, length=0.003)
# [factors_original, factors_pca, factors_MT]
plt.legend(['original factors', 'pca factors / bets',
            ' minimum - torsion factors / bets'])
add_logo(fig)
plt.tight_layout()
# -


