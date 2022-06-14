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

# # s_interactions [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_interactions&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_interactions).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from arpym.statistics.simulate_normal import simulate_normal
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_interactions-parameters)

j_in_sample = 2000  # number of in-sample simulations
mu_z = np.zeros(2)  # expectation
sigma2_z = np.array([[1, 0], [0, 1]])  # covariance
q_max = 10  # maximum degree of polynomials considered
j_out_sample = 1000  # simulations of out-of-sample error

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_interactions-implementation-step01): Define features and target variables

# +
z_in = simulate_normal(mu_z, sigma2_z, j_in_sample)  # scenarios of features

def muf(z1, z2):
    return z1 - np.tanh(10*z1*z2)

def sigf(z1, z2):
    return np.sqrt(np.minimum(z1**2, 1/(10*np.pi)))

x_in = muf(z_in[:, 0], z_in[:, 1]) +\
       sigf(z_in[:, 0], z_in[:, 1]) * simulate_normal(0, 1, j_in_sample)  # scenarios of target variables
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_interactions-implementation-step02): Fit conditional expectation and compute in-sample error

# +
#initialize variables
err_in = np.zeros(q_max)
err_out = np.zeros((j_out_sample, q_max))
err_out_med = np.zeros(q_max)
err_out_iqr = np.zeros(q_max)

for q in np.arange(q_max):

    #Construct inputs products in-sample
    poly = PolynomialFeatures(degree=q+1, include_bias=False)
    z_inter_in = poly.fit_transform(z_in)

    #Fit conditional expectation
    reg = linear_model.LinearRegression()
    exp_in_sample = reg.fit(z_inter_in, x_in).predict(z_inter_in)

    # Compute in-sample error
    err_in[q] = np.mean((x_in-exp_in_sample)**2)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_interactions-implementation-step03): Compute out-of-sample error

    for i in np.arange(j_out_sample):
        
        # out-of-sample features and target variables
        z_out = simulate_normal(mu_z, sigma2_z, j_in_sample)
        x_out = muf(z_out[:, 0], z_out[:, 1]) +\
                sigf(z_out[:, 0], z_out[:, 1]) * simulate_normal(0, 1, j_in_sample)
        poly = PolynomialFeatures(degree=q+1, include_bias=False)
        z_inter_out = poly.fit_transform(z_out)
        
        # out-of-sample error
        exp_out_sample = reg.predict(z_inter_out)
        err_out[i, q] = np.mean((x_out-exp_out_sample)**2)

    err_out_med[q] = np.median(err_out[:, q]) # out-of-sample error location
    err_out_iqr[q] = np.percentile(err_out[:, q], 75) -\
        np.percentile(err_out[:, q], 25) # out-of-sample error dispersion

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_interactions-implementation-step04): Save database

output = {'z': pd.Series(z_in.reshape(-1)),
          'x': pd.Series(x_in),
          'j_in_sample': pd.Series(j_in_sample)}
df = pd.DataFrame(output)
df.to_csv('~/databases/temporary-databases/db_ml_variables.csv')

# ## Plots

# +
plt.style.use('arpm')

idxx0 = np.where(np.abs(z_in[:, 0]) <= 2)[0]
idxx1 = np.where(np.abs(z_in[:, 1]) <= 2)[0]
idxx = np.intersect1d(idxx0, idxx1)
lightblue = [0.2, 0.6, 1]
lightgreen = [0.6, 0.8, 0]

fig = plt.figure()

# Error
ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=4)
insamplot = ax1.plot(np.arange(q_max)+1, err_in, color='k')
ax1.set_ylabel('In-sample error', color='k')
ax1.tick_params(axis='y', colors='k')
ax1.set_xticks(np.arange(q_max)+1)
ax12 = ax1.twinx()
outsamplot = ax12.plot(np.arange(q_max)+1, err_out_med, color='r',
                       lw=1.15)
ax12.tick_params(axis='y', colors='r')
ax12.set_ylabel('Out-of-sample error', color='r')
ax1.set_xlabel('Degree of the polynomial')
plt.xlim([0, q_max + 1])
ax1.set_title('In-sample vs out-of-sample errors as ' +
              'function of polynomial degree', fontweight='bold')
ax1.grid(False)
ax12.grid(False)

# Conditional expectation surface
ax2 = plt.subplot2grid((3, 4), (1, 0), colspan=2, rowspan=2, projection='3d')
step = 0.01
zz1, zz2 = np.meshgrid(np.arange(-2, 2, step), np.arange(-2, 2, step))
ax2.plot_surface(zz1, zz2, muf(zz1, zz2), color=lightblue, alpha=0.7,
                 label='$\mu(z_1, z_2)$')

ax2.scatter3D(z_in[idxx, 0], z_in[idxx, 1],
              x_in[idxx], s=10, color=lightblue, alpha=1,
              label='$(Z_1, Z_2, X)$')
ax2.set_xlabel('$Z_1$')
ax2.set_ylabel('$Z_2$')
ax2.set_zlabel('$X$')
ax2.set_title('Conditional expectation surface', fontweight='bold')
ax2.set_xlim([-2, 2])
ax2.set_ylim([-2, 2])

# Fitted surface
ax3 = plt.subplot2grid((3, 4), (1, 2), colspan=2, rowspan=2, projection='3d')
step = 0.01
zz1, zz2 = np.meshgrid(np.arange(-2, 2, step), np.arange(-2, 2, step))
zz = poly.fit_transform(np.c_[zz1.ravel(), zz2.ravel()])
xx = reg.predict(zz)
ax3.plot_surface(zz1, zz2, xx.reshape((zz1.shape)), color=lightgreen,
                 alpha=0.7, label='Fitted surface')

ax3.scatter3D(z_in[idxx, 0], z_in[idxx, 1],
              reg.predict(z_inter_in)[idxx], s=10, color=lightgreen,
              alpha=1, label='$(Z_1,Z_2, \overline{X})$')
ax3.set_xlabel('$Z_1$')
ax3.set_ylabel('$Z_2$')
ax3.set_zlabel('$\overline{X}$')
ax3.set_title('Fitted surface', fontweight='bold')
ax3.set_xlim([-2, 2])
ax3.set_ylim([-2, 2])

add_logo(fig, axis=ax1)
plt.tight_layout()
