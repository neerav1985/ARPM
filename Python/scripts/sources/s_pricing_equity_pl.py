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

# # s_pricing_equity_pl [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_pricing_equity_pl&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-4-equity-pl).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.tools.logo import add_logo
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_equity_pl-implementation-step00): Import data

# +
data = pd.read_csv('~/databases/temporary-databases/db_stocks_proj_bm.csv')
j_ =  data['j_'][0]
mu =  data['mu_hat'][0]
sig2 =  data['sig2_hat'][0]
# scenario-probability distribution of log-value
x_tnow_thor = data['x_tnow_thor'].values
x_tnow_thor = x_tnow_thor.reshape((j_, -1))
t_m = data['t_m']
t_m.dropna(inplace=True)
t_m = t_m.values
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_equity_pl-implementation-step01): Equity P&L

# +
x_t_now = x_tnow_thor[0, 0]
# current value of AMZN
v_t_now = np.exp(x_t_now)
# equity P&L
pl_tnow_thor = v_t_now*(np.exp(x_tnow_thor-x_t_now) - 1)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_equity_pl-implementation-step02): Analytical P&L (shifted log-normal) at the horizon

# +
l_ = 2000  # number of points
pl_grid_hor = np.linspace(np.min(pl_tnow_thor)-20,
                          np.max(pl_tnow_thor)+20, l_)
# log-normal parameters of the horizon P&L
mu_pl_thor = x_t_now+mu*t_m[-1]
sig_pl_thor = np.sqrt(sig2*t_m[-1])
# analytical pdf
v_thor = pl_grid_hor + v_t_now
y_pdf_hor = (np.exp(-(np.log(v_thor)-mu_pl_thor)**2/(2*sig_pl_thor**2))
            /(v_thor * sig_pl_thor * np.sqrt(2 * np.pi)))
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_equity_pl-implementation-step03): P&L expectations and standard deviations

# +
# expectation
mu_pl = v_t_now*(np.exp((mu+0.5*sig2)*t_m)-1)
# standard deviation
sig_pl = v_t_now*np.exp((mu+0.5*sig2)*t_m) * \
         np.sqrt(np.exp(sig2*t_m)-1)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_equity_pl-implementation-step04): Save databases

# +
output = {'v_t_now': v_t_now,
          'pl_tnow_thor': pd.Series(pl_tnow_thor.reshape(-1))
          }
df = pd.DataFrame(output)
df.to_csv('~/databases/temporary-databases/db_equity_pl.csv')
# -

# ## Plots

# +
# preliminary settings
plt.style.use('arpm')
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.2, 0.2, 0.2]  # dark grey

s_ = 0  # number of plotted observation before projecting time

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

# axes settings
t_line = np.arange(0, t_m[-1] + 0.01, 0.01)
t = np.concatenate((np.arange(-s_, 0), t_m))
max_scale = t_m[-1] / 4
scale = max_scale*0.96/np.max(y_pdf_hor)

plt.axis([t[0], t[-1] + max_scale, np.min(pl_tnow_thor)-20,
          np.max(pl_tnow_thor)+20])
plt.xlabel('time (days)', fontsize=18)
plt.ylabel('P&L', fontsize=18)
plt.yticks()
plt.grid(False)
plt.title('Equity P&L', fontsize=20, fontweight='bold')

# simulated paths
plt.plot(t_m, pl_tnow_thor.T, color=lgrey, lw=2)

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
    plt.plot([t_m[-1], t_m[-1]+y_pdf_hor[k]*scale],
             [pl_grid_hor[k], pl_grid_hor[k]],
             color=lgrey, lw=2)

plt.plot(t_m[-1] + y_pdf_hor*scale, pl_grid_hor,
         color=dgrey, label='horizon pdf', lw=1)

# legend
plt.legend(loc=2, fontsize=17)
add_logo(fig, location=4, alpha=0.8, set_fig_size=False)
