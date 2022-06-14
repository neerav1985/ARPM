#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # s_ewm_bf_statistics [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_ewm_bf_statistics&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-ewmanum-ex-copy-1).

# %%
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.estimation.exp_decay_fp import exp_decay_fp
from arpym.statistics.meancov_sp import meancov_sp
from arpym.statistics.quantile_sp import quantile_sp
from arpym.tools.logo import add_logo

# %% [markdown]
# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_ewm_bf_statistics-parameters)

# %%
t_ = 1799  # number of observations
tau_hl = 25  # half-life parameter
c = 0.05  # confidence level

# %% [markdown]
# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_ewm_bf_statistics-implementation-step00): Upload from database

# %%
# S&P 500 index value
spx_path = '~/databases/global-databases/equities/db_stocks_SP500/SPX.csv'
spx_all = pd.read_csv(spx_path, parse_dates=['date'])
spx = spx_all.loc[spx_all.index.max() - t_:spx_all.index.max(), :]
spx = spx.set_index(pd.to_datetime(spx.date))

# %% [markdown]
# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_ewm_bf_statistics-implementation-step01): Compute time series of S&P 500 compounded return

# %%
epsi = np.diff(np.log(spx.SPX_close))  # S&P 500 index compounded return

# %% [markdown]
# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_ewm_bf_statistics-implementation-step02): Compute exponential decay probabilities, backward/forward EWMA, EWM standard deviation and EWM quantile

# %%
_2ewma = np.zeros(t_)
_2ewm_cv = np.zeros(t_)
_2ewm_sd = np.zeros(t_)
_2ewm_q = np.zeros(t_)

for t in range(t_): 
    p_t = exp_decay_fp(t_, tau_hl, t_star = t)  # exponential decay probabilities
    _2ewma[t], _2ewm_cv[t] = meancov_sp(epsi, p_t)  # backward/forward EWM average and covariance
    _2ewm_q[t] = quantile_sp(c, epsi, p_t)  # backward/forward EWM quantile
    
_2ewm_sd = np.sqrt(_2ewm_cv)  # backward/forward EWM standard deviation

# %% [markdown]
# ## Plots

# %%
plt.style.use('arpm')

k_color = [33/255, 37/255, 41/255]
g_color = [71/255, 180/255, 175/255]
b_color = [13/255, 94/255, 148/255]
r_color = [227/255, 66/255, 52/255]

myFmt = mdates.DateFormatter('%d-%m-%Y')

mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)
plt.xlim(min(spx.index[1:]), max(spx.index[1:]))

plt.plot(spx.index[1:], epsi[0:], '.b', color=b_color, label=r'S&P 500 log-returns')
plt.plot(spx.index[1:], _2ewma, color=g_color, lw=1.5, label=r'B/F EWMA')
plt.plot(spx.index[1:], _2ewma + 2 * _2ewm_sd, color=r_color, label=r'+/- 2 B/F EWM std. dev. band')
plt.plot(spx.index[1:], _2ewma - 2 * _2ewm_sd, color=r_color)
plt.plot(spx.index[1:], _2ewm_q, color=k_color, label=r'B/F EWM 0.05-quantile')

plt.legend(loc=1)
plt.gca().xaxis.set_major_formatter(myFmt)

