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

# # s_checklist_historical_step01 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_historical_step01&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-1-historical).

# +
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.pricing.bsm_function import bsm_function
from arpym.pricing.implvol_delta2m_moneyness import implvol_delta2m_moneyness
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step01-parameters)

# +
# set current time t_now
t_now = np.datetime64('2012-08-31')

# set start date for data selection
t_first = np.datetime64('2009-11-02')

# set initial portfolio construction date t_init
t_init = np.datetime64('2012-08-30')

# stocks - must include GE and JPM
stock_names = ['GE', 'JPM', 'A', 'AA', 'AAPL']  # stocks considered
# make sure stock names includes GE and JPM
stock_names = ['GE', 'JPM'] + [stock
                               for stock in stock_names
                               if stock not in ['GE', 'JPM']]
print('Stocks considered:', stock_names)

# options on S&P 500
k_strk = 1407  # strike value of options on S&P 500 (US dollars)
tend_option = np.datetime64('2013-08-26')  # options expiry date
y = 0.01  # level for yield curve (assumed flat and constant)
l_ = 9  # number of points on the m-moneyness grid

# index of risk driver to plot
d_plot = 1
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step01-implementation-step00): Import data

# +
# upload data
# stocks
stocks_path = '~/databases/global-databases/equities/db_stocks_SP500/'
db_stocks = pd.read_csv(stocks_path + 'db_stocks_sp.csv', skiprows=[0],
                        index_col=0)
db_stocks.index = pd.to_datetime(db_stocks.index)

# implied volatility of option on S&P 500 index
path = '~/databases/global-databases/derivatives/db_implvol_optionSPX/'
db_impliedvol = pd.read_csv(path + 'data.csv',
                            index_col=['date'], parse_dates=['date'])
implvol_param = pd.read_csv(path + 'params.csv', index_col=False)

# define the date range of interest
dates = db_stocks.index[(db_stocks.index >= t_first) &
                        (db_stocks.index <= t_now)]
dates = np.intersect1d(dates, db_impliedvol.index)
dates = dates.astype('datetime64[D]')

# length of the time series
t_ = len(dates)

# initialize temporary databases
db_risk_drivers = {}
v_tnow = {}
v_tinit = {}
risk_drivers_names = {}
v_tnow_names = {}

# implied volatility parametrized by time to expiry and delta-moneyness
tau_implvol = np.array(implvol_param.time2expiry)
tau_implvol = tau_implvol[~np.isnan(tau_implvol)]
delta_moneyness = np.array(implvol_param.delta)

implvol_delta_moneyness_2d = \
    db_impliedvol.loc[(db_impliedvol.index.isin(dates)),
                      (db_impliedvol.columns != 'underlying')]

k_ = len(tau_implvol)

# unpack flattened database (from 2d to 3d)
implvol_delta_moneyness_3d = np.zeros((t_, k_, len(delta_moneyness)))
for k in range(k_):
    implvol_delta_moneyness_3d[:, k, :] = \
        np.array(implvol_delta_moneyness_2d.iloc[:, k::k_])
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step01-implementation-step01): Stocks

# +
n_stocks = len(stock_names)  # number of stocks
d_stocks = n_stocks  # one risk driver for each stock

for d in range(d_stocks):
    # calculate time series of stock risk drivers
    db_risk_drivers[d] = np.log(np.array(db_stocks.loc[dates, stock_names[d]]))
    risk_drivers_names[d] = 'stock '+stock_names[d]+'_log_value'
    # stock value
    v_tnow[d] = db_stocks.loc[t_now, stock_names[d]]
    v_tinit[d] = db_stocks.loc[t_init, stock_names[d]]
    v_tnow_names[d] = 'stock '+stock_names[d]

# number of risk drivers, to be updated at every insertion
d_ = d_stocks
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step01-implementation-step02): S&P 500 Index

# +
# calculate risk driver of the S&P 500 index
db_risk_drivers[d_] = \
    np.log(np.array(db_impliedvol.loc[(db_impliedvol.index.isin(dates)),
                                      'underlying']))
risk_drivers_names[d_] = 'sp_index_log_value'

# value of the S&P 500 index
v_tnow[d_] = db_impliedvol.loc[t_now, 'underlying']
v_tinit[d_] = db_impliedvol.loc[t_init, 'underlying']
v_tnow_names[d_] = 'sp_index'

# update counter
d_ = d_+1
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step01-implementation-step03): Call and put options on the S&P 500 Index

# +
# from delta-moneyness to m-moneyness parametrization
implvol_m_moneyness_3d, m_moneyness = \
    implvol_delta2m_moneyness(implvol_delta_moneyness_3d, tau_implvol,
                              delta_moneyness, y*np.ones((t_, k_)),
                              tau_implvol, l_)

# calculate log implied volatility
log_implvol_m_moneyness_2d = \
    np.log(np.reshape(implvol_m_moneyness_3d,
                      (t_, k_*(l_)), 'F'))

# value of the underlying
s_tnow = v_tnow[d_stocks]
s_tinit = v_tinit[d_stocks]

# time to expiry (in years)
tau_option_tnow = np.busday_count(t_now, tend_option)/252
tau_option_tinit = np.busday_count(t_init, tend_option)/252

# moneyness
moneyness_tnow = np.log(s_tnow/k_strk)/np.sqrt(tau_option_tnow)
moneyness_tinit = np.log(s_tinit/k_strk)/np.sqrt(tau_option_tinit)

# grid points
points = list(zip(*[grid.flatten() for grid in np.meshgrid(*[tau_implvol,
                                                             m_moneyness])]))

# known values
values = implvol_m_moneyness_3d[-1, :, :].flatten('F')

# implied volatility (interpolated)
impl_vol_tnow = \
    interpolate.LinearNDInterpolator(points, values)(*np.r_[tau_option_tnow,
                                                            moneyness_tnow])
impl_vol_tinit = \
    interpolate.LinearNDInterpolator(points, values)(*np.r_[tau_option_tinit,
                                                            moneyness_tinit])

# compute call option value by means of Black-Scholes-Merton formula
v_call_tnow = bsm_function(s_tnow, y, impl_vol_tnow, moneyness_tnow, tau_option_tnow)
v_call_tinit = bsm_function(s_tinit, y, impl_vol_tinit, moneyness_tinit,
                            tau_option_tinit)

# compute put option value by means of the put-call parity
v_zcb_tnow = np.exp(-y*tau_option_tnow)
v_put_tnow = v_call_tnow - s_tnow + k_strk*v_zcb_tnow
v_zcb_tinit = np.exp(-y*tau_option_tinit)
v_put_tinit = v_call_tinit - s_tinit + k_strk*v_zcb_tinit

# store data
d_implvol = log_implvol_m_moneyness_2d.shape[1]
for d in np.arange(d_implvol):
    db_risk_drivers[d_+d] = log_implvol_m_moneyness_2d[:, d]
    risk_drivers_names[d_+d] = 'option_spx_logimplvol_mtau_' + str(d+1)

v_tnow[d_] = v_call_tnow
v_tinit[d_] = v_call_tinit
v_tnow_names[d_] = 'option_spx_call'
v_tnow[d_+1] = v_put_tnow
v_tinit[d_+1] = v_put_tinit
v_tnow_names[d_+1] = 'option_spx_put'

# update counter
d_ = len(db_risk_drivers)
n_ = len(v_tnow)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step01-implementation-step04): Save databases

# +
path = '~/databases/temporary-databases/'

# market risk drivers
out = pd.DataFrame({risk_drivers_names[d]: db_risk_drivers[d]
                    for d in range(len(db_risk_drivers))}, index=dates)
out = out[list(risk_drivers_names.values())]
out.index.name = 'dates'
out.to_csv(path+'db_riskdrivers_series_historical.csv')
del out

# values of all instruments at t_now
out = pd.DataFrame({v_tnow_names[n]: pd.Series(v_tnow[n])
                   for n in range(len(v_tnow))})
out = out[list(v_tnow_names.values())]
out.to_csv(path+'db_v_tnow_historical.csv',
           index=False)
del out

# values of all instruments at t_init
out = pd.DataFrame({v_tnow_names[n]: pd.Series(v_tinit[n])
                   for n in range(len(v_tinit))})
out = out[list(v_tnow_names.values())]
out.to_csv(path+'db_v_tinit_historical.csv',
           index=False)
del out

# additional variables needed for subsequent steps
out = {'n_stocks': pd.Series(n_stocks),
       'd_implvol': pd.Series(d_implvol),
       'tend_option': pd.Series(tend_option),
       'k_strk': pd.Series(k_strk),
       'l_': pd.Series(l_),
       'tau_implvol': pd.Series(tau_implvol),
       'y': pd.Series(y),
       'm_moneyness': pd.Series(m_moneyness),
       'd_': pd.Series(d_),
       't_now': pd.Series(t_now),
       't_init': pd.Series(t_init),
       't_first': pd.Series(t_first),
       'stock_names': pd.Series(stock_names)}
out = pd.DataFrame(out)
out.to_csv(path+'db_riskdrivers_tools_historical.csv',
           index=False)
del out
# -

# ## Plots 

# +
plt.style.use('arpm')

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
plt.plot(dates, db_risk_drivers[d_plot-1])
plt.title(risk_drivers_names[d_plot-1], fontweight='bold', fontsize=20)
plt.xlabel('time (days)', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([dates[0], dates[-1]])
add_logo(fig, set_fig_size=False)
fig.tight_layout()
