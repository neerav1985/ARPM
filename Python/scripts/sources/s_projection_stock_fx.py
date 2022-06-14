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

# # s_projection_stock_fx [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_projection_stock_fx&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_fx).

# +
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from arpym.estimation.exp_decay_fp import exp_decay_fp
from arpym.statistics.meancov_sp import meancov_sp
from arpym.statistics.simulate_bm import simulate_bm
from arpym.tools.histogram_sp import histogram_sp
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_fx-parameters)

# +
j_ = 100  # number of scenarios
m_ = 10   # number of monitoring times (days)
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_fx-implementation-step00): Upload data

# +
stock_path = '~/databases/global-databases/equities/'
fx_path = '~/databases/global-databases/currencies/'
# import data
df_stocks = pd.read_csv(stock_path + 'db_stocks_SP500/db_stocks_sp.csv', index_col=0,
                        skiprows=[0])
# set timestamps
df_stocks = df_stocks.set_index(pd.to_datetime(df_stocks.index))
# select stock
df_stocks = df_stocks['AMZN']  # stock value
# select exchange rate
fx_df = pd.read_csv(fx_path + 'db_fx/data.csv', index_col=0, usecols=['date', 'GBP'],
                    parse_dates=['date'])
fx_df.dropna(inplace=True)
# joint time index
joint_ind =  df_stocks.index.intersection(fx_df.index)

# select data within the date range
t_ = 504  # length of time series
df_stocks = df_stocks.loc[joint_ind].tail(t_)  # stock value
fx_usd2gbp = fx_df.loc[joint_ind].tail(t_).values  # USD/GBP exchange rate
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_fx-implementation-step01): Compute the risk drivers

# +
x_stock = np.log(np.array(df_stocks))  # log-value
x_fx = np.log(fx_usd2gbp).reshape(-1)  # USD/GBP log-exchange rate
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_fx-implementation-step02): Compute HFP mean and covariance

# +
tau_hl = 180  # half-life (days)
# exponential decay probabilities
p = exp_decay_fp(t_ - 1, tau_hl)
# invariant past realizations
epsi_stock = np.diff(x_stock)
epsi_fx = np.diff(x_fx)
# HFP mean and covariance
mu_hat, sig2_hat = meancov_sp(np.r_[epsi_stock.reshape(1,-1),
                              epsi_fx.reshape(1,-1)].reshape((-1, 2)),
                              p)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_fx-implementation-step03): Generate Monte Carlo scenarios for the risk drivers process

# +
# Monte Carlo scenarios
delta_t_m = np.ones(m_)
x_stock_fx_0 = np.r_[x_stock[-1], x_fx[-1]]
x_tnow_thor = simulate_bm(x_stock_fx_0, delta_t_m, mu_hat,
                          sig2_hat, j_).squeeze()
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_projection_stock_fx-implementation-step04): Save databases

# +
t_m = np.append(0, np.cumsum(delta_t_m))
output = {'j_': j_,
          't_': t_,
          'm_': m_,
          'p': pd.Series(p.reshape(-1)),
          'mu_hat': pd.Series(mu_hat.reshape(-1)),
          'sig2_hat': pd.Series(sig2_hat.reshape(-1)),
          'x_tnow_thor': pd.Series(x_tnow_thor.reshape(-1))}
df = pd.DataFrame(output)
df.to_csv('~/databases/temporary-databases/db_stocks_fx_proj_bm.csv')
# -

# ## Plots

# +
plt.style.use('arpm')
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey

# plot that corresponds to step 4
num_plot = min(j_, 30)

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)


mu_thor = np.zeros((len(t_m), 2))
sig2_thor = np.zeros((len(t_m), 2, 2))
for t in range(len(t_m)):
    mu_thor[t], sig2_thor[t] = meancov_sp(x_tnow_thor[:, t, :])

# log-values
ax1 = fig.add_subplot(211)
ax1.set_xlim(0, t_m[-1]*1.35)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
i = 0
mu_stock_thor = np.zeros(m_ + 1)
sig_stock_thor = np.zeros(m_ + 1)
for m in range(0, m_ + 1):
    mu_stock_thor[m] = mu_thor[m, 0]
    sig_stock_thor[m] = np.sqrt(sig2_thor[m, 0, 0])

for j in range(num_plot):
    plt.plot(t_m, x_tnow_thor[j, :, i], lw=1, color=lgrey) 

f, xp = histogram_sp(x_tnow_thor[:, -1, i], k_=20*np.log(j_))
rescale_f = 0.3*f*t_m[-1]/np.max(f)
plt.barh(xp, rescale_f, height=xp[1]-xp[0], left=t_m[-1], facecolor=lgrey,
         edgecolor=lgrey,  label='horizon pdf')
plt.plot(rescale_f+t_m[-1], xp, color=dgrey, lw=1)
# mean plot
p_mu = plt.plot(t_m, mu_stock_thor, color='g',
                label='expectation', lw=1)
p_red_1 = plt.plot(t_m, mu_stock_thor + 2 * sig_stock_thor,
                   label='+ / - 2 st.deviation', color='r', lw=1)
p_red_2 = plt.plot(t_m, mu_stock_thor - 2 * sig_stock_thor,
                   color='r', lw=1)
plt.legend(fontsize=17)
plt.xlabel(r'$t_{\mathit{hor}}-t_{\mathit{now}}$ (days)', fontsize=17)
title = "Projection of log-value"
plt.title(title, fontsize=20, fontweight='bold')

# currency log-exchange rate
ax2 = fig.add_subplot(212)
ax2.set_xlim(0, t_m[-1]*1.35)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
i = 1
         
mu_fx_thor = np.zeros(m_ + 1)
sig_fx_thor = np.zeros(m_ + 1)
for m in range(0, m_ + 1):
    mu_fx_thor[m] = mu_thor[m, 1]
    sig_fx_thor[m] = np.sqrt(sig2_thor[m, 1, 1])
for j in range(num_plot):
    plt.plot(t_m, x_tnow_thor[j, :, i], lw=1, color=lgrey)   

f, xp = histogram_sp(x_tnow_thor[:, -1, i], k_=20*np.log(j_))
rescale_f = 0.3*f*t_m[-1]/np.max(f)
plt.barh(xp, rescale_f, height=xp[1]-xp[0], left=t_m[-1], facecolor=lgrey,
         edgecolor=lgrey)
plt.plot(rescale_f+t_m[-1], xp, color=dgrey, lw=1)
# mean plot
p_mu = plt.plot(t_m, mu_fx_thor, color='g',
                label='expectation', lw=1)
p_red_1 = plt.plot(t_m, mu_fx_thor+2*sig_fx_thor,
                   label='+ / - 2 st.deviation', color='r', lw=1)
p_red_2 = plt.plot(t_m, mu_fx_thor-2*sig_fx_thor,
                   color='r', lw=1)
plt.xlabel(r'$t_{\mathit{hor}}-t_{\mathit{now}}$ (days)', fontsize=17)
title = "Projection of currency log-exchange rate"
plt.title(title, fontsize=20, fontweight='bold')
add_logo(fig, set_fig_size=False)
fig.tight_layout()
