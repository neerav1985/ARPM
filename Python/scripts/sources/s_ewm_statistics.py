#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # s_ewm_statistics [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_ewm_statistics&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-ewmanum-ex).

# %%
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.estimation.exp_decay_fp import exp_decay_fp
from arpym.statistics.ewm_meancov import ewm_meancov
from arpym.statistics.quantile_sp import quantile_sp
from arpym.tools.logo import add_logo

# %% [markdown]
# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_ewm_statistics-parameters)

# %%
t_ = 1799  # number of observations
tau_hl = 25  # half-life parameter
w = 200  # trailing window
c = 0.05  # confidence level

# %% [markdown]
# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_ewm_statistics-implementation-step00): Upload from database

# %%
# S&P 500 index value
spx_path = '~/databases/global-databases/equities/db_stocks_SP500/SPX.csv'
spx_all = pd.read_csv(spx_path, parse_dates=['date'])
spx = spx_all.loc[spx_all.index.max() - t_:spx_all.index.max(), :]
spx = spx.set_index(pd.to_datetime(spx.date))

# %% [markdown]
# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_ewm_statistics-implementation-step01): Compute time series of S&P 500 compounded return

# %%
epsi = np.diff(np.log(spx.SPX_close))  # S&P 500 index compounded return

# %% [markdown]
# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_ewm_statistics-implementation-step02): Compute EWMA and EWM standard deviation

# %%
ewma = np.zeros(t_ - w + 1)
ewm_cv = np.zeros(t_ - w + 1)
ewm_sd = np.zeros(t_ - w + 1)

for t in range(w, t_):
    ewma[t - w], ewm_cv[t - w] = ewm_meancov(epsi[t - w:t], tau_hl, w)  # EWM average and covariance

ewm_sd = np.sqrt(ewm_cv)  # EWM standard deviation

# %% [markdown]
# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_ewm_statistics-implementation-step03): Compute flexible probabilities and EWM quantile

# %%
ewm_q = np.zeros(t_ - w + 1)
p_s = exp_decay_fp(w, tau_hl)

for t in range(w, t_):
    ewm_q[t - w] = quantile_sp(c, epsi[t - w:t], p_s)  # EWM quantile

# %% [markdown]
# ## Plots

# %%
plt.style.use('arpm')

k_color = [33/255, 37/255, 41/255]
g_color = [71/255, 180/255, 175/255]
b_color = [13/255, 94/255, 148/255]
r_color = [227/255, 66/255, 52/255]

myFmt = mdates.DateFormatter('%d-%b-%Y')

mydpi = 72.0
f = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)
plt.xlim(np.min(spx.index[w:]), np.max(spx.index[w:]))

plt.plot(spx.index[w:], epsi[w-1:], '.b', color=b_color, label=r'S&P 500 log-returns')
plt.plot(spx.index[w:], ewma, color=g_color, lw=1.5, label=r'EWMA')
plt.plot(spx.index[w:], ewma + 2*ewm_sd, color=r_color, lw=1, label=r'+/- 2 EWM std. dev. band')
plt.plot(spx.index[w:], ewma - 2*ewm_sd, color=r_color, lw=1)
plt.plot(spx.index[w:], ewm_q, color=k_color, lw=1, label=r'EWM 0.05-quantile')

plt.legend(loc=1)
plt.gca().xaxis.set_major_formatter(myFmt)

