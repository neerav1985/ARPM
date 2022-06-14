#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_stock_long_horizon [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_stock_long_horizon&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerStockLong).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.tools.adjusted_value import adjusted_value
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_stock_long_horizon-parameters)

# select starting and ending date for the plot (format: day-month-year)
# fwd=True for forward adjusted value, fwd!=True for backward adjusted value
start_date = '25-2-2010'  # starting date
end_date = '17-7-2012'  # ending date
fwd = True  # indicator for forward of backward adjusted value

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_stock_long_horizon-implementation-step00): Load data

# +
# loading data from 03-01-1994 to 24-11-2017
path = '~/databases/global-databases/equities/db_stocks_SP500/'
df_nokia_stock = pd.read_csv(path + 'NOK_prices.csv',
                             header=0)
df_nok_dividends = pd.read_csv(path + 'NOK_dividends.csv',
                               header=0)

# convert column 'date' from string to datetime64
df_nokia_stock['date_tmstmp'] = pd.to_datetime(df_nokia_stock.date,
                                               dayfirst=True)
df_nok_dividends['date_tmstmp'] = pd.to_datetime(df_nok_dividends.date,
                                                 dayfirst=True)

t_start = pd.to_datetime(start_date, dayfirst=True)
t_end = pd.to_datetime(end_date, dayfirst=True)
# filter the data for the selected range
nok_stock_long = df_nokia_stock[(df_nokia_stock.date_tmstmp >= t_start) &
                                (df_nokia_stock.date_tmstmp < t_end)]
nok_dividends = df_nok_dividends[(df_nok_dividends.date_tmstmp >= t_start) &
                                 (df_nok_dividends.date_tmstmp < t_end)]
# extract values
dates = nok_stock_long.date_tmstmp.values
r = nok_dividends.date_tmstmp.values
cf_r = nok_dividends.dividends.values
v_stock = nok_stock_long.close.values
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_stock_long_horizon-implementation-step01): Dividend-adjusted values and dividend-adjusted log-values

v_tilde_stock = adjusted_value(v_stock, dates, cf_r, r, fwd)
ln_v_tilde_stock = np.log(v_tilde_stock)

# ## Plots

# +
number_of_xticks = 6
tick_array = np.linspace(0, dates.shape[0]-1, number_of_xticks, dtype=int)

plt.style.use('arpm')
fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('Value')
ax1.grid(True)

ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1, sharex=ax1)
plt.xlabel('Date')
plt.ylabel('ln(adjusted value)')
ax2.grid(True)

ax1.plot_date(dates, v_stock, 'b-', label='market value')
ax1.plot_date(dates, v_tilde_stock, 'r-', label='adjusted value')
ax1.plot([], [], linestyle='--', lw=1, c='k', label='ex-dividend date')
ax1.set_title('Market value')
[ax1.axvline(x=d, linestyle='--', lw=1, c='k') for d in r]
for d, v in zip(r, cf_r):
    ax1.axvline(x=d, linestyle='--', lw=1, c='k')
ax1.legend()

[ax2.axvline(x=d, linestyle='--', lw=1, c='k') for d in r]
ax2.plot_date(dates, ln_v_tilde_stock, '-', label='log-adjusted value', c='aqua')
ax2.set_title('Log-adjusted value')
ax2.legend()
plt.xticks(dates[tick_array], size=8)

add_logo(fig, location=3, set_fig_size=False)
plt.tight_layout()
