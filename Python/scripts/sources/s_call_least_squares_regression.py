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

# # s_call_least_squares_regression [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_call_least_squares_regression&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_call_least_squares_regression).

# +
import random
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rc, rcParams
from sklearn.tree import DecisionTreeRegressor

from arpym.estimation.smooth_kernel_fp import smooth_kernel_fp
from arpym.statistics.meancov_sp import meancov_sp
from arpym.tools.logo import add_logo

rc('text', usetex=True)
rcParams['text.latex.preamble']=[r'\usepackage{amsmath} \usepackage{amssymb}']
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_call_least_squares_regression-parameters)

l_ = 10  # number of leaves in CART

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_call_least_squares_regression-implementation-step00): Upload data

# +
path = '~/databases/temporary-databases/'

# upload repriced projected call option data
db_call_data = pd.read_csv(path+'db_call_data.csv', parse_dates=[3,6], index_col=0)
m_ = db_call_data.m_[0].astype(int)  # number of monitoring times
j_ = db_call_data.j_[0].astype(int)  # number of scenarios
v_call_tnow_thor = db_call_data.v_call_thor.values.reshape(j_, m_+1)  # call option value scenarios
log_v_sandp = db_call_data.log_s.values.reshape(j_, m_+1)  # underlying log-value scenarios

# upload Black-Scholes hedge data
db_calloption_bms_values = pd.read_csv(path+'db_calloption_bms_values.csv', index_col=0)
v_sandp_grid = db_calloption_bms_values.loc[:, 'v_sandp_grid'].values  # grid of underlying values
r_sandp_grid = db_calloption_bms_values.loc[:, 'r_sandp_grid'].values  # grid of underlying returns
r_bar_call_bs_grid = db_calloption_bms_values.loc[:, 'r_bar_call_bs_grid'].values  # grid of BMS hedge returns
r_call_tend = db_calloption_bms_values.loc[:, 'r_call_tend'].values  # grid of option payoffs
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_call_least_squares_regression-implementation-step01): Compute return scenarios

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

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_call_least_squares_regression-implementation-step02): Compute optimal least squares hedge

# +
# kernel bandwidth
i_ = r_sandp_grid.shape[0]
h = (r_sandp_grid[-1] - r_sandp_grid[0])/i_
# smoothing parameter
gamma = 2

# compute mean regression predictor on the grid
e_r_call_given_rsandp = np.zeros(i_)
for i, r_sandp_grid_i in enumerate(r_sandp_grid):
    # smooth kernel conditional probabilities
    p_given_rsandp_i = smooth_kernel_fp(r_sandp, r_sandp_grid_i, h, gamma)
    # conditional expectation
    e_r_call_given_rsandp[i], _ = meancov_sp(r_call, p_given_rsandp_i)

# mean regression predictor as piecewise linear interpolation
def chi(z):
    return np.interp(z, r_sandp_grid, e_r_call_given_rsandp)

r_bar_call_opt = chi(r_sandp)  # hedge returns
x_minus_xbar_opt = r_call-r_bar_call_opt  # hedged returns (residuals)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_call_least_squares_regression-implementation-step03): Compute linear least squares hedge 

# +
# expectation and covariance of the joint returns
e_r_call_r_sandp_joint, cov_r_call_r_sandp_joint = meancov_sp(np.c_[r_call, r_sandp])
e_r_call, e_r_sandp = e_r_call_r_sandp_joint
cov_r_call_r_sandp = cov_r_call_r_sandp_joint[0, 1]
var_r_sandp = cov_r_call_r_sandp_joint[1, 1]
e_r_call_r_sandp = cov_r_call_r_sandp + e_r_call*e_r_sandp
e_r_sandp2 = var_r_sandp + e_r_sandp**2

# parameters of the linear least squares predictor
beta = e_r_call_r_sandp/e_r_sandp2

# linear least squares predictor
def chi_beta(z):
    return beta*np.array(z)

r_bar_call_beta = chi_beta(r_sandp)  # hedge returns
x_minus_xbar_beta = r_call-r_bar_call_beta  # hedged returns (residuals)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_call_least_squares_regression-implementation-step04): Compute CART least squares hedge 

# +
# fit CART least squares regression
tree_delta_star = DecisionTreeRegressor(criterion='mse', max_leaf_nodes=l_) \
                                        .fit(r_sandp.reshape(-1, 1), r_call)

# CART regression predictor 
def chi_cart(z):
    return tree_delta_star.predict(np.array(z).reshape(-1, 1))

r_bar_call_cart = chi_cart(r_sandp)  # hedge returns
x_minus_xbar_cart = r_call-r_bar_call_cart  # hedged returns (residuals)
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
sand = '#f7d2a9'  # sand

markersize = 6
# number of plotted simulations
j_plot = random.sample(range(j_), min(j_, 2000))

ratio = v_sandp_tnow/v_call_tnow
ylim = [-1.7, 4]
xstart = -0.3
xlim = [xstart, (ylim[1]-ylim[0])/ratio+xstart]

def plot_call_returns(title, predictor, pred_label, legend_label, r_bar_call, u, method=None):
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
    if method=='cart':
        l5, = ax1.step(np.sort(r_sandp[j_plot]), predictor(np.sort(r_sandp[j_plot])),
                       c=orange, lw=1.5)
    else:
        l5, = ax1.plot(np.sort(r_sandp[j_plot]), predictor(np.sort(r_sandp[j_plot])),
                       c=orange, lw=1.5)
    l6, = ax1.plot(r_sandp_grid, r_bar_call_bs_grid, c='k', lw=1)
    l7, = ax1.plot(r_sandp_grid, r_call_tend, '--', c='k', lw=1.5)
    l9, = ax1.plot(0, 0, 'o', color='k')
    ax1.set_title(title,
                  fontdict={'fontsize': 20, 'fontweight': 'bold'},
                  usetex=False)
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
    ax4.set_xlabel(pred_label,
                   fontdict={'size': 17}, labelpad=-32, x=0.86)
    ax4.set_ylabel(r'$R^{\mathit{call}}$', fontdict={'size': 17}, labelpad=-45)
    ax4.scatter(r_bar_call[j_plot], r_call[j_plot],
                s=markersize, c=[light_grey])
    ax6 = plt.subplot2grid((48, 60), (20, 41), colspan=19, rowspan=4, sharex=ax4)
    ax6.tick_params(axis='both', colors='none')
    aaa = ax6.hist(r_bar_call, bins='auto', density=True,
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
    ax7.set_xlabel(pred_label,
                   fontdict={'size': 17}, labelpad=-32, x=0.86)
    ax7.set_ylabel(r'$X-\bar{X}$', fontdict={'size': 17}, labelpad=-45)
    ax7.fill_between(ylim, -ulim, 0, alpha=0.1, color=orange)
    ax7.scatter(r_bar_call[j_plot], u[j_plot], s=markersize, c=[light_grey])
    ax8 = plt.subplot2grid((48, 60), (24, 37), colspan=4, rowspan=20, sharey=ax7)
    ax8.tick_params(axis='both', colors='none')
    ax8.tick_params(axis='both', colors='none')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax8.invert_xaxis()
    ax8.hist(u, bins='auto', density=True, facecolor=colf, ec=colf,
             orientation='horizontal')
    ax8.spines['top'].set_visible(False)
    ax8.spines['bottom'].set_visible(False)
    ax8.spines['left'].set_visible(False)
    ax9 = plt.subplot2grid((48, 60), (44, 41), colspan=19, rowspan=4,
                           sharex=ax7)
    ax9.tick_params(axis='both', colors='none')
    ax9.invert_yaxis()
    ax9.hist(r_bar_call, bins='auto', density=True, facecolor=light_green_1,
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
    fig.legend((l1, l2, l3, l4, l5, l6, l9, l7),
               ('Input', 'Output', 'Predictor', 'Residual', legend_label,
                'BMS value', 'Current value', 'Payoff'),
                loc=(0.12, 0.03), fontsize=17,
                facecolor='none', edgecolor='none', ncol=2)
    rc('text', usetex=True)
    plt.subplots_adjust(left=0)
    add_logo(fig, axis=ax1, location=1, size_frac_x=1/12, set_fig_size=False)
    plt.show()
    return fig


# figure for optimal prediction
fig_condexp = plot_call_returns(title = 'Mean regression',
                                predictor = chi,
                                pred_label = r'$\chi({R}^{\mathit{S\&P}})$',
                                legend_label = 'Cond. exp',
                                r_bar_call = r_bar_call_opt,
                                u = x_minus_xbar_opt)

# figure for linear prediction
fig_linls = plot_call_returns(title = 'Linear mean regression',
                              predictor = chi_beta,
                              pred_label = r'$\chi_{\beta}({R}^{\mathit{S\&P}})$',
                              legend_label = 'LS lin. approx.',
                              r_bar_call = r_bar_call_beta,
                              u = x_minus_xbar_beta)

# figure for CART prediction
fig_cart = plot_call_returns(title = 'CART mean regression',
                             predictor = chi_cart,
                             pred_label = r'$\chi_{\beta, \Delta^{\ast}}({R}^{\mathit{S\&P}})$',
                             legend_label = 'LS CART approx.',
                             r_bar_call = r_bar_call_cart,
                             u = x_minus_xbar_cart,
                             method = 'cart')
