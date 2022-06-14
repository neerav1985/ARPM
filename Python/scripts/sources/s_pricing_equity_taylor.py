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

# # s_pricing_equity_taylor [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_pricing_equity_taylor&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-taylor-equity-pl).

# +
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from arpym.statistics.saddle_point_quadn import saddle_point_quadn
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_equity_taylor-parameters)

# +
delta_t = 3  #  time to horizon (in days)
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_equity_taylor-implementation-step00): Import data

# +
data = pd.read_csv('~/databases/temporary-databases/db_stocks_proj_bm.csv')
j_ =  data['j_'][0]
# parameters for GBM driving the log-value process
mu =  data['mu_hat'][0]
sig2 =  data['sig2_hat'][0]

data_pricing = pd.read_csv('~/databases/temporary-databases/db_equity_pl.csv')
v_t_now = data_pricing['v_t_now'][0]  # current value
pl_tnow_thor = data_pricing['pl_tnow_thor']
pl_tnow_thor = pl_tnow_thor.values.reshape((j_, -1))
t_m = data['t_m']
t_m.dropna(inplace=True)
index = np.where(delta_t <= t_m.values)[0][0]
pl_tnow_thor = pl_tnow_thor[:, :index+1]  # equity P&L
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_equity_taylor-implementation-step01): Analytical distribution of first order Taylor approximation

# +
l_ = 2000  # number of points
pl_grid = np.linspace(np.min(pl_tnow_thor)-20,
                      np.max(pl_tnow_thor)+20, l_)

# parameters
mu_thor = mu*delta_t
sig_thor = np.sqrt(sig2*delta_t)

# analytical pdf of first order approximation
pl_tl_pdf_1 = stats.norm.pdf(pl_grid, v_t_now*mu_thor, v_t_now*sig_thor)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_equity_taylor-implementation-step02): Analytical distribution of second order Taylor approximation

# +
# analytical pdf of second order approximation
pl_tl_pdf_2 = saddle_point_quadn(pl_grid.T, 0, v_t_now*np.array([1]),
                                 v_t_now*np.array([[1/2]]), mu_thor.reshape(-1),
                                 (sig_thor**2).reshape((-1, 1)))[1]
# -

# ## Plots

# +
# preliminary settings
plt.style.use('arpm')
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.2, 0.2, 0.2]  # dark grey
lblue = [0.27, 0.4, 0.9]  # light blue
orange = [0.94, 0.35, 0]  # orange

s_ = 0  # number of plotted observation before projecting time

# log-normal parameters of exact P&L
mu_pl_thor = np.log(v_t_now)+mu*delta_t
sig_pl_thor = np.sqrt(sig2*delta_t)
# analytical pdf at horizon
v_thor = pl_grid + v_t_now
y_pdf_hor = (np.exp(-(np.log(v_thor)-mu_pl_thor)**2/(2*sig_pl_thor**2))
            /(v_thor * sig_pl_thor * np.sqrt(2 * np.pi)))

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

# axes settings
t_line = np.arange(0, t_m[index] + 0.01, 0.01)
t = np.concatenate((np.arange(-s_, 0), t_m))
max_scale = t_m[index] / 4
scale = max_scale*0.96/np.max(y_pdf_hor)

plt.axis([t[0], t[index] + max_scale, np.min(pl_grid),
          np.max(pl_grid)])
plt.xlabel('time (days)', fontsize=18)
plt.ylabel('P&L', fontsize=18)
plt.yticks()
plt.grid(False)
plt.title('Equity P&L Taylor approximation', fontsize=20, fontweight='bold')

# simulated paths
plt.plot(t_m[:index+1], pl_tnow_thor.T, color=lgrey, lw=2)

# expectation line
mu_line = v_t_now*(np.exp((mu+0.5*sig2)*t_line)-1)
plt.plot(t_line, mu_line, color='g',
         label='expectation', lw=2)
# standard deviation lines
num_sd = 2
sig_line = v_t_now*np.exp((mu+0.5*sig2)*t_line) * \
           np.sqrt(np.exp(sig2*t_line)-1)
plt.plot(t_line, mu_line +  num_sd*sig_line, color='r',
         label='+ / - %d st.deviation' %num_sd, lw=2)
plt.plot(t_line, mu_line - num_sd*sig_line, color='r', lw=2)

# analytical pdf at the horizon
for k, y in enumerate(y_pdf_hor):
    plt.plot([t_m[index], t_m[index]+y_pdf_hor[k]*scale],
             [pl_grid[k], pl_grid[k]],
             color=lgrey, lw=2)

plt.plot(t_m[index] + y_pdf_hor*scale, pl_grid,
         color=dgrey, label='horizon pdf', lw=2)

# first order Taylor approximation
plt.plot(t_m[index] + pl_tl_pdf_1*scale, pl_grid, color=orange,
         label='first order approx. pdf', lw=2)

# second order Taylor approximation
plt.plot(t_m[index] + pl_tl_pdf_2*scale, pl_grid, color=lblue,
         label='second order approx. pdf', lw=2)

# legend
plt.legend(loc=2, fontsize=17)
add_logo(fig, location=4, alpha=0.8, set_fig_size=False)
