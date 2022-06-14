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

# # s_call_bms_hedging [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_call_bms_hedging&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_call_bms_hedging).

# +
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rc, rcParams

from arpym.pricing.call_option_value import call_option_value
from arpym.tools.logo import add_logo

rc('text', usetex=True)
rcParams['text.latex.preamble']=[r'\usepackage{amsmath} \usepackage{amssymb}']
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_call_bms_hedging-implementation-step00): Upload data

# +
path = '~/databases/temporary-databases/'

# upload repriced projected call option data
db_call_data = pd.read_csv(path+'db_call_data.csv', parse_dates=[3,6], index_col=0)
t_end = np.datetime64(db_call_data.t_end[0], 'D')  # expiry date of the options
k_strk = db_call_data.k_strike[0]  # strike of the options on the S&P500 ($US)
m_ = db_call_data.m_[0].astype(int)  # number of monitoring times
t_now = np.datetime64(db_call_data.t_m[0], 'D')  # current date
t_hor = np.datetime64(db_call_data.t_m.iloc[m_], 'D')  # valuation horizon
j_ = db_call_data.j_[0].astype(int)  # number of scenarios
v_call_tnow_thor = db_call_data.v_call_thor.values.reshape(j_, m_+1)  # call option value scenarios
log_v_sandp = db_call_data.log_s.values.reshape(j_, m_+1)  # underlying log-value scenarios
y = db_call_data.y_rf[0]  # yield curve (assumed flat and constant)
tau_y = np.busday_count(t_now, t_end)/252  # time to option expiry

# upload implied volatility surface at t_now
db_calloption_proj = pd.read_csv(path+'db_calloption_proj.csv', index_col=0)
m_grid = np.array([float(col[col.find('m=')+2:col.find(' tau=')])
                        for col in db_calloption_proj.columns[1:]])
m_grid = np.unique(m_grid)
tau_grid = np.array([float(col[col.find(' tau=')+5:])
                        for col in db_calloption_proj.columns[1:]])
tau_grid = np.unique(tau_grid)
logsigma_tnow = db_calloption_proj.values.reshape(j_, -1, db_calloption_proj.shape[1])
logsigma_tnow = logsigma_tnow[0, 0, 1:]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_call_bms_hedging-implementation-step01): Compute return scenarios

# +
# extract current values of call option and underlying
v_call_tnow = v_call_tnow_thor[0, 0]  # value of the call option at t_now
v_sandp_tnow = np.exp(log_v_sandp[0, 0])  # value of S&P 500 index at t_now

# extract scenarios of call option and the underlying values at horizon
v_call_thor = v_call_tnow_thor[:, m_]  # scenarios at the horizon of the call option
v_sandp_thor = np.exp(log_v_sandp[:, m_])  # scenarios at the horizon S&P 500 index

# compute returns of the call option and the underlying between t_now and t_hor
r_call = (v_call_thor/v_call_tnow - 1)
r_sandp = (v_sandp_thor/v_sandp_tnow - 1)


# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_call_bms_hedging-implementation-step02): Compute Black-Scholes hedge

# +
# Black-Merton-Scholes call values

# BMS hedging function
def chi_bs(z):
    s = np.atleast_1d(v_sandp_tnow*(z + 1))
    v_bar_call_thor = call_option_value(t_now, np.log(s), np.atleast_1d(y), tau_y,
                                        np.exp(logsigma_tnow), m_grid, tau_grid,
                                        k_strk, t_end, sr=0, logsig=0).squeeze()
    x_bar = v_bar_call_thor/v_call_tnow-1
    return x_bar

# apply BMS hedging function to scenarios
r_bar_call_bs = np.zeros_like(r_sandp)
for j, r_sandp_j in enumerate(r_sandp):
    r_bar_call_bs[j] = chi_bs(r_sandp_j)  # hedge returns
x_minus_xbar_bs = r_call-r_bar_call_bs  # hedged returns (residuals)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_call_bms_hedging-implementation-step03): Compute Black-Scholes hedge on a grid of underlying values

# +
# grid of underlying values and returns
i_ = 50  # number points in grid
r_sandp_grid = np.linspace(r_sandp.min(), r_sandp.max(), i_)
v_sandp_grid = v_sandp_tnow*(r_sandp_grid + 1)

# apply BSM hedging function to underlying grid
r_bar_call_bs_grid = np.zeros_like(r_sandp_grid)
for i, r_sandp_i in enumerate(r_sandp_grid):
    r_bar_call_bs_grid[i] = chi_bs(r_sandp_i)

# payoff of call option as the function of the underlying values
v_call_tend = np.maximum(v_sandp_grid-k_strk, 0)
r_call_tend = v_call_tend/v_call_tnow - 1
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_call_bms_hedging-implementation-step04): Save database

# +
output = {'v_sandp_grid': pd.Series(v_sandp_grid),
          'r_sandp_grid': pd.Series(r_sandp_grid),
          'r_bar_call_bs_grid': pd.Series(r_bar_call_bs_grid),
          'r_call_tend': pd.Series(r_call_tend)}

df = pd.DataFrame(output)
df.to_csv(path + 'db_calloption_bms_values.csv')
# -

# ## Plots:

# +
# colors
teal = '#3c9591'  # teal
light_green_1 = '#d7ead0'  # light green
light_green_2 = '#47b4af'  # light teal
light_grey = '#969696'  # gray
orange = '#ff9900'  # orange
colf = '#b5e1df'  # light blue 1

markersize = 6
# number of plotted simulations
j_plot = random.sample(range(j_), min(j_, 2000))
ratio = v_sandp_tnow/v_call_tnow
ylim = [-1.7, 4]
xstart = -0.3
xlim = [xstart, (ylim[1]-ylim[0])/ratio+xstart]

plt.style.use('arpm')
fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0)

# plot of underlying vs option returns
ax1 = plt.subplot2grid((8, 10), (0, 1), colspan=5, rowspan=6)
ax1.tick_params(axis='x', which='major', pad=-15, direction='out')
ax1.tick_params(axis='y', which='major', pad=-20, direction='out')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax1.set_xlabel(r'$R^{\mathit{S\&P}}$', fontdict={'size': 17}, labelpad=-30)
ax1.set_ylabel(r'$R^{\mathit{call}}$', fontdict={'size': 17}, labelpad=-35)
ax1.scatter(r_sandp[j_plot], r_call[j_plot], s=markersize,
            c=[light_grey])
l6, = ax1.plot(r_sandp_grid, r_bar_call_bs_grid, c='k', lw=1)
l7, = ax1.plot(r_sandp_grid, r_call_tend, '--', c='k', lw=1.5)
l9, = ax1.plot(0, 0, 'o', color='k')
ax1.set_title('Black-Scholes hedge',
              fontdict={'fontsize': 20, 'fontweight': 'bold'}, usetex=False)

ax2 = plt.subplot2grid((8, 10), (0, 0), colspan=1, rowspan=6, sharey=ax1)
ax2.invert_xaxis()
ax2.hist(r_call, bins='auto', density=True, facecolor=teal, ec=teal,
         orientation='horizontal')
ax2.tick_params(axis='both', colors='none')
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax3 = plt.subplot2grid((8, 10), (6, 1), colspan=5, rowspan=1, sharex=ax1)
ax3.tick_params(axis='both', colors='none')
ax3.invert_yaxis()
ax3.hist(r_sandp, bins='auto', density=True, facecolor=light_green_2,
         ec=light_green_2)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax1.set_ylim(ylim)
ax1.set_xlim(xlim)

# plot of option vs hedge returns
ax4 = plt.subplot2grid((48, 60), (0, 41), colspan=19, rowspan=20)
ax4.tick_params(axis='x', which='major', pad=-16)
ax4.tick_params(axis='y', which='major', pad=-17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax4.set_xlabel(r'$\chi({R}^{\mathit{S\&P}})$',
               fontdict={'size': 17}, labelpad=-32, x=0.86)
ax4.set_ylabel(r'$R^{\mathit{call}}$', fontdict={'size': 17}, labelpad=-45)
ax4.scatter(r_bar_call_bs[j_plot], r_call[j_plot],
            s=markersize, c=[light_grey])
ax6 = plt.subplot2grid((48, 60), (20, 41), colspan=19, rowspan=4, sharex=ax4)
ax6.tick_params(axis='both', colors='none')
aaa = ax6.hist(r_bar_call_bs, bins='auto', density=True,
               facecolor=light_green_1, ec=light_green_1)
val, edg = aaa[0], aaa[1]
cent = edg[:-1]+0.5*(edg[1]-edg[0])
ax6.invert_yaxis()
ax6.spines['right'].set_visible(False)
ax6.spines['bottom'].set_visible(False)
ax6.spines['left'].set_visible(False)
ax5 = plt.subplot2grid((48, 60), (0, 37), colspan=4, rowspan=20, sharey=ax4)
ax5.tick_params(axis='both', colors='none')
ax5.invert_xaxis()
ax5.hist(r_call, bins='auto', density=True, facecolor=teal, ec=teal,
         orientation='horizontal')
ax5.plot(val, cent, color=light_green_1, lw=2)
ax5.spines['top'].set_visible(False)
ax5.spines['bottom'].set_visible(False)
ax5.spines['left'].set_visible(False)
ax4.set_xlim(ylim)
ax4.set_ylim(ylim)

# plot of hedge vs hedged returns (residual)
ax7 = plt.subplot2grid((48, 60), (24, 41), colspan=19, rowspan=20)
ulim = 0.45
ax7.tick_params(axis='x', which='major', pad=-16)
ax7.tick_params(axis='y', which='major', pad=-27)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax7.set_xlabel(r'$\chi({R}^{\mathit{S\&P}})$',
               fontdict={'size': 17}, labelpad=-32, x=0.86)
ax7.set_ylabel(r'$X-\bar{X}$', fontdict={'size': 17}, labelpad=-45)
ax7.fill_between(ylim, -ulim, 0, alpha=0.1, color=orange)
ax7.scatter(r_bar_call_bs[j_plot], x_minus_xbar_bs[j_plot], s=markersize, c=[light_grey])
ax8 = plt.subplot2grid((48, 60), (24, 37), colspan=4, rowspan=20, sharey=ax7)
ax8.tick_params(axis='both', colors='none')
ax8.tick_params(axis='both', colors='none')
ax8.invert_xaxis()
ax8.hist(x_minus_xbar_bs, bins='auto', density=True, facecolor=colf, ec=colf,
         orientation='horizontal')
ax8.spines['top'].set_visible(False)
ax8.spines['bottom'].set_visible(False)
ax8.spines['left'].set_visible(False)
ax9 = plt.subplot2grid((48, 60), (44, 41), colspan=19, rowspan=4,
                       sharex=ax7)
ax9.tick_params(axis='both', colors='none')
ax9.invert_yaxis()
ax9.hist(r_bar_call_bs, bins='auto', density=True, facecolor=light_green_1,
         ec=light_green_1)
ax9.spines['right'].set_visible(False)
ax9.spines['bottom'].set_visible(False)
ax9.spines['left'].set_visible(False)
ax7.set_xlim(ylim)
ax7.set_ylim([-ulim, ulim])

l1 = Rectangle((0, 0), 1, 1, color=light_green_2, ec='none')
l2 = Rectangle((0, 0), 1, 1, color=teal, ec='none')
l3 = Rectangle((0, 0), 1, 1, color=light_green_1, ec='none')
l4 = Rectangle((0, 0), 1, 1, color=colf, ec='none')
rc('text', usetex=False)
fig.legend((l1, l2, l3, l4, l6, l9, l7),
           ('Input', 'Output', 'Predictor', 'Residual',
            'BMS value', 'Current value', 'Payoff'),
            loc=(0.12, 0.03), fontsize=17,
            facecolor='none', edgecolor='none', ncol=2)
rc('text', usetex=True)
plt.subplots_adjust(left=0)
add_logo(fig, axis=ax1, location=1, size_frac_x=1/12, set_fig_size=False)
plt.show()
