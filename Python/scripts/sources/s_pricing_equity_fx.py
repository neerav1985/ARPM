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

# # s_pricing_equity_fx [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_pricing_equity_fx&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-4-exch-equity-pl).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.tools.logo import add_logo
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_equity_fx-implementation-step00): Import data

# +
data = pd.read_csv('~/databases/temporary-databases//db_stocks_fx_proj_bm.csv')
j_ =  data['j_'][0]
m_ =  data['m_'][0]  # number of monitoring times (days)
mu =  data['mu_hat']
mu.dropna(inplace=True) 
mu = mu.values
sig2 =  data['sig2_hat']
sig2.dropna(inplace=True)
sig2 = sig2.values.reshape((2, 2))
mu_usd = mu[0]  # loc. par. for log-value in $
mu_fx = mu[1]  # loc. par. for log-exchange rate
sig2_usd = sig2[0, 0]  # disp. par. for log-value in $
sig2_fx = sig2[1, 1]  # disp. par. for log-exchange rate
sig2_usd_fx = sig2[0, 1]  # covariance between log-value in $ and log-exchange rate
x_tnow_thor = data['x_tnow_thor'].values  # Monte Carlo scenarios
x_tnow_thor = x_tnow_thor.reshape((j_, -1, 2))
# log-exchange rate at t_now
x_fx_tnow =  x_tnow_thor[0, 0, 1]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_equity_fx-implementation-step01): Scenarios for exchange rate and equity P&L's in both USD and GBP

# +
# current value
v_t_now = np.exp(x_tnow_thor[0, 0, 0])
# equity P&L in domestic currency
pl_gbp_tnow_thor = np.exp(x_tnow_thor[:, :, 0] +
                          x_tnow_thor[:, :, 1]) -\
                   v_t_now*np.exp(x_tnow_thor[0, 0, 1])

# equity P&L in foreign currencyy
pl_tnow_thor = v_t_now*(np.exp(x_tnow_thor[:, :, 0]-x_tnow_thor[0, 0, 0])-1)

# foreign exchange rate
fx_tnow_thor = np.exp(x_tnow_thor[:, :, 1])
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_equity_fx-implementation-step02): Analytical pdfs at the horizon

# +
l_ = 2000  # number of points
t_m = np.append(0, np.cumsum(np.ones(m_)))
pl_grid = np.linspace(np.min(pl_gbp_tnow_thor)-40,
                      np.max(pl_gbp_tnow_thor)+55, l_)
fx_grid = np.linspace(np.min(fx_tnow_thor)-0.1,
                      np.max(fx_tnow_thor)+0.2, l_)

# log-normal parameters of the horizon P&L in domestic currency
mu_pl_gbp_thor = np.log(v_t_now)+x_fx_tnow+(mu_usd+mu_fx)*t_m[-1]
sig_pl_gbp_thor = np.sqrt((sig2_usd+sig2_fx+2*sig2_usd_fx)*t_m[-1])
# analytical pdf
pl_gbp_grid = pl_grid + v_t_now*np.exp(x_fx_tnow)
pl_gbp_pdf_hor = np.exp(-(np.log(pl_gbp_grid)-mu_pl_gbp_thor)**2 /
                         (2*sig_pl_gbp_thor**2)) /\
                 (pl_gbp_grid * sig_pl_gbp_thor * np.sqrt(2 * np.pi))

# log-normal parameters of the horizon P&L in foreign currency
mu_pl_usd_thor = np.log(v_t_now)+mu_usd*t_m[-1]
sig_pl_usd_thor = np.sqrt(sig2_usd*t_m[-1])
# analytical pdf
pl_usd_pdf_hor = np.exp(-(np.log(pl_gbp_grid)-mu_pl_usd_thor)**2 /
                        (2*sig_pl_usd_thor**2)) /\
                (pl_gbp_grid * sig_pl_usd_thor * np.sqrt(2 * np.pi))

# log-normal parameters of FX rate
mu_fx_thor = x_fx_tnow+mu_fx*t_m[-1]
sig_fx_thor = np.sqrt(sig2_fx*t_m[-1])
# analytical pdf
fx_pdf_hor = np.exp(-(np.log(fx_grid)-mu_fx_thor)**2/(2*sig_fx_thor**2)) /\
             (fx_grid * sig_fx_thor * np.sqrt(2 * np.pi))
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_equity_fx-implementation-step03):  Projected expectations and standard deviations

# +
# moments of equity P&L in domestic currency
# log-normal base parameters
mu_gbp = mu_usd+mu_fx
sig2_gbp = sig2_usd+sig2_fx+2*sig2_usd_fx

mu_pl_gbp = v_t_now*np.exp(x_fx_tnow)*np.exp((mu_gbp+0.5*sig2_gbp)*t_m)-\
            v_t_now*np.exp(x_fx_tnow)
sig_pl_gbp = v_t_now*np.exp(x_fx_tnow)*(np.exp((mu_gbp+0.5*sig2_gbp)*t_m)) *\
              np.sqrt(np.exp(sig2_gbp*t_m)-1)

# moments of equity P&L in foreign currency
mu_pl_usd = v_t_now*(np.exp((mu_usd+0.5*sig2_usd)*t_m)-1)
sig_pl_usd = v_t_now*np.exp((mu_usd+0.5*sig2_usd)*t_m) * \
              np.sqrt(np.exp(sig2_usd*t_m)-1)

# moments of FX rate
mu_fx_m = np.exp(x_fx_tnow)*np.exp((mu_fx+0.5*sig2_fx)*t_m)
sig_fx_m = np.exp(x_fx_tnow)*(np.exp((mu_fx+0.5*sig2_fx)*t_m)) *\
         np.sqrt(np.exp(sig2_fx*t_m)-1)
# -

# ## Plots

# +
plt.style.use('arpm')
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey

num_plot = min(j_, 30)

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
# axes settings
s_ = 0
t_line = np.arange(0, t_m[-1] + 0.01, 0.01)
t = np.concatenate((np.arange(-s_, 0), t_m))
max_scale = t_m[-1] / 4

# foreign exchange rate
ax1 = fig.add_subplot(211)
ax1.set_xlim(0, t_m[-1]*1.35)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

scale = max_scale*0.96/np.max(fx_pdf_hor)

# simulated paths
for j in range(num_plot):
    plt.plot(t_m, fx_tnow_thor[j, :], lw=1, color=lgrey)

# expectation line
mu_line = fx_tnow_thor[0, 0]*np.exp((mu_fx+0.5*sig2_fx)*t_line)
plt.plot(t_line, mu_line, color='g',
         label='expectation', lw=2)
# standard deviation lines
num_sd = 2
sig_line = fx_tnow_thor[0, 0]*(np.exp((mu_fx+0.5*sig2_fx)*t_line)) *\
           np.sqrt(np.exp(sig2_fx*t_line)-1)

plt.plot(t_line, mu_line+num_sd*sig_line, color='r',
         label='+ / - %d st.deviation' % num_sd, lw=2)
plt.plot(t_line, mu_line-num_sd*sig_line, color='r', lw=2)

# analytical pdf at the horizon
for k, y in enumerate(fx_pdf_hor):
    plt.plot([t_m[-1], t_m[-1]+fx_pdf_hor[k]*scale],
             [fx_grid[k], fx_grid[k]],
             color=lgrey, lw=2)

plt.plot(t_m[-1] + fx_pdf_hor*scale, fx_grid,
         color=dgrey, label='horizon pdf', lw=1)

plt.legend(loc=2, fontsize=17)
plt.xlabel('time (days)', fontsize=18)
plt.ylabel('exchange rate', fontsize=18)
plt.yticks()
plt.grid(False)
title = "Equity P&L in domestic currency"
plt.title(title, fontsize=20, fontweight='bold')

#  P&L in domestic currency
ax2 = fig.add_subplot(212)
ax2.set_xlim(0, t_m[-1]*1.35)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

scale = max_scale*0.96/np.max(pl_gbp_pdf_hor)

# simulated paths
for j in range(num_plot):
    plt.plot(t_m, pl_gbp_tnow_thor[j, :], lw=1, color=lgrey)

# expectation line
mu_line = v_t_now*np.exp(x_fx_tnow)*(np.exp((mu_gbp+0.5*sig2_gbp)*t_line)-1)
plt.plot(t_line, mu_line, color='g',
         label='P&L in domestic currency expectation', lw=2)
# standard deviation lines
sig_line = v_t_now*np.exp(x_fx_tnow)*(np.exp((mu_gbp+0.5*sig2_gbp)*t_line)) *\
              np.sqrt(np.exp(sig2_gbp*t_line)-1)

plt.plot(t_line, mu_line+num_sd*sig_line, color='r',
         label='P&L in domestic currency + / - %d st.deviation' % num_sd, lw=2)
plt.plot(t_line, mu_line-num_sd*sig_line, color='r', lw=2)

# analytical pdf at the horizon
for k, y in enumerate(pl_gbp_pdf_hor):
    plt.plot([t_m[-1], t_m[-1]+pl_gbp_pdf_hor[k]*scale],
             [pl_grid[k], pl_grid[k]],
             color=lgrey, lw=2)

plt.plot(t_m[-1] + pl_gbp_pdf_hor*scale, pl_grid,
         color=dgrey, label='P&L in domestic currency horizon pdf', lw=1)

plt.xlabel('time (days)', fontsize=18)
plt.ylabel('P&L', fontsize=18)
plt.yticks()
plt.grid(False)


# expectation line
mu_line_n = np.exp(x_fx_tnow)*v_t_now*(np.exp((mu_usd+0.5*sig2_usd)*t_line)-1)

plt.plot(t_line, mu_line_n, linestyle='--',color='k',
         label='P&L in foreign currency (normalized) features', lw=2)
# standard deviation lines
plt.plot(t_line, mu_line_n+num_sd*sig_line,
         linestyle='--',color='k', lw=2)
plt.plot(t_line, mu_line_n-num_sd*sig_line,
         linestyle='--',color='k', lw=2)

mu_pl_norm_thor = np.log(v_t_now)+x_fx_tnow+mu_usd*t_m[-1]
sig_pl_norm_thor = np.sqrt(sig2_usd*t_m[-1])
# normalized analytical pdf at horizon
pl_norm_pdf_hor = np.exp(-(np.log(pl_gbp_grid)-mu_pl_norm_thor)**2 /
                         (2*sig_pl_norm_thor**2)) /\
                  (pl_gbp_grid * sig_pl_norm_thor * np.sqrt(2 * np.pi))

scale_n = max_scale*0.96/np.max(pl_norm_pdf_hor)
plt.plot(t_m[-1] + pl_norm_pdf_hor*scale_n, pl_grid,
         lw=1, linestyle='--',color='k')
plt.legend(loc=2, fontsize=17)
add_logo(fig, location=4, alpha=0.8, set_fig_size=False)
fig.tight_layout()
