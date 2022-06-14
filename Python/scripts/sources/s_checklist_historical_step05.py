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

# # s_checklist_historical_step05 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_historical_step05&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-5-historical).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.pricing.bsm_function import bsm_function
from arpym.tools.histogram_sp import histogram_sp
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step05-parameters)

n_plot = 1  # index of instrument to plot

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step05-implementation-step00): Load data

# +
path = '~/databases/temporary-databases/'

# Risk drivers identification
# risk driver values
db_riskdrivers_series = pd.read_csv(path+'db_riskdrivers_series_historical.csv',
                                    index_col=0)
x = db_riskdrivers_series.values

# values at t_now
db_v_tnow = pd.read_csv(path+'db_v_tnow_historical.csv')
v_tnow = db_v_tnow.values[0]

# additional information
db_riskdrivers_tools = pd.read_csv(path+'db_riskdrivers_tools_historical.csv',
                                  parse_dates=True)
d_ = int(db_riskdrivers_tools['d_'].dropna())
n_stocks = int(db_riskdrivers_tools['n_stocks'].dropna())
n_ = n_stocks+3
d_implvol = int(db_riskdrivers_tools['d_implvol'].dropna())
tend_option = np.datetime64(db_riskdrivers_tools['tend_option'][0], 'D')
k_strk = db_riskdrivers_tools['k_strk'][0]
l_ = int(db_riskdrivers_tools['l_'].dropna())
m_moneyness = db_riskdrivers_tools['m_moneyness'].values[:l_]
tau_implvol = db_riskdrivers_tools['tau_implvol'].values
y = db_riskdrivers_tools['y'][0]
t_now = np.datetime64(db_riskdrivers_tools.t_now[0], 'D')
# index of risk drivers for options
idx_options = np.array(range(n_stocks+1, n_stocks+d_implvol+1))

# Projection
# projected risk driver paths
db_projection_riskdrivers = \
    pd.read_csv(path+'db_projection_bootstrap_riskdrivers.csv')

# additional information
db_projection_tools = \
    pd.read_csv(path+'db_projection_bootstrap_tools.csv')
j_ = int(db_projection_tools['j_'][0])
t_hor = np.datetime64(db_projection_tools['t_hor'][0], 'D')

# projected scenarios probabilities
db_scenario_probs = pd.read_csv(path+'db_scenario_probs_bootstrap.csv')
p = db_scenario_probs['p'].values
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step05-implementation-step01): Calculate number of business days between t_now and t_hor

# +
# business days between t_now and t_hor
m_ = np.busday_count(t_now, t_hor)
# date of next business day (t_now + 1)
t_1 = np.busday_offset(t_now, 1, roll='forward')

# projected scenarios
x_proj = db_projection_riskdrivers.values.reshape(j_, m_+1, d_)

# initialize output arrays
pi_tnow_thor = np.zeros((j_, n_))
pi_oneday = np.zeros((j_, n_))
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step05-implementation-step02): Stocks

for n in range(n_stocks):
    pi_tnow_thor[:, n] = v_tnow[n] * (np.exp(x_proj[:, -1, n] - x[-1, n])-1)
    pi_oneday[:, n] = v_tnow[n] * (np.exp(x_proj[:, 1, n] - x[-1, n])-1)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step05-implementation-step03): S&P index

pi_tnow_thor[:, n_stocks] = v_tnow[n_stocks]*(np.exp(x_proj[:, -1, n_stocks] -
                                               x[-1, n_stocks])-1)
pi_oneday[:, n_stocks] = v_tnow[n_stocks]*(np.exp(x_proj[:, 1, n_stocks] -
                                               x[-1, n_stocks])-1)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step05-implementation-step04): Options

# +
# time to expiry of the options at the horizon t_hor
tau_opt_thor = np.busday_count(t_hor, tend_option)/252
# time to expiry of the options after one day
tau_opt_oneday = np.busday_count(t_1, tend_option)/252

# underlying and moneyness at the horizon
s_thor = np.exp(x_proj[:, -1, n_stocks])
mon_thor = np.log(s_thor/k_strk)/np.sqrt(tau_opt_thor)
# underlying and moneyness after one day
s_oneday = np.exp(x_proj[:, 1, n_stocks])
mon_oneday = np.log(s_oneday/k_strk)/np.sqrt(tau_opt_oneday)

# log-implied volatility at the horizon
logsigma_thor = x_proj[:, -1, idx_options].reshape(j_, -1, l_)
# log-implied volatility after one day
logsigma_oneday = x_proj[:, 1, idx_options].reshape(j_, -1, l_)

# interpolate log-implied volatility
logsigma_interp = np.zeros(j_)
logsigma_interp_oneday = np.zeros(j_)
for j in range(j_):
    # grid points
    points = list(zip(*[grid.flatten()
                        for grid in np.meshgrid(*[tau_implvol, m_moneyness])]))
    # known values
    values = logsigma_thor[j, :, :].flatten()
    values_oneday = logsigma_oneday[j, :, :].flatten()
    # interpolation
    moneyness_thor = min(max(mon_thor[j], min(m_moneyness)), max(m_moneyness))
    moneyness_oneday = min(max(mon_oneday[j], min(m_moneyness)), max(m_moneyness))
    # log-implied volatility at the horizon
    logsigma_interp[j] =\
        interpolate.LinearNDInterpolator(points, values)(*np.r_[tau_opt_thor,
                                                                moneyness_thor])
    # log-implied volatility after one day
    logsigma_interp_oneday[j] =\
        interpolate.LinearNDInterpolator(points, values_oneday)(*np.r_[tau_opt_oneday,
                                                                       moneyness_oneday])

# compute call option value by means of Black-Scholes-Merton formula
v_call_thor = bsm_function(s_thor, y, np.exp(logsigma_interp), moneyness_thor,
                           tau_opt_thor)
v_call_oneday = bsm_function(s_oneday, y, np.exp(logsigma_interp_oneday), 
                             moneyness_oneday, tau_opt_oneday)

# compute put option value using put-call parity
v_zcb_thor = np.exp(-y*tau_opt_thor)
v_put_thor = v_call_thor - s_thor + k_strk*v_zcb_thor
v_zcb_oneday = np.exp(-y*tau_opt_oneday)
v_put_oneday = v_call_oneday - s_oneday + k_strk*v_zcb_oneday

# compute P&L of the call option
pi_tnow_thor[:, n_stocks+1] = v_call_thor - v_tnow[n_stocks+1]
pi_oneday[:, n_stocks+1] = v_call_oneday - v_tnow[n_stocks+1]
# compute P&L of the put option
pi_tnow_thor[:, n_stocks+2] = v_put_thor - v_tnow[n_stocks+2]
pi_oneday[:, n_stocks+2] = v_put_oneday - v_tnow[n_stocks+2]
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step05-implementation-step05): Save database

# +
# ex-ante performance over [t_now, t_hor)
out = {db_v_tnow.columns[n]: pi_tnow_thor[:, n]
       for n in range(n_)}
names = [db_v_tnow.columns[n] for n in range(n_)]
out = pd.DataFrame(out)
out = out[list(names)]
out.to_csv(path+'db_pricing_historical.csv',
           index=False)
del out

# ex-ante performance over one day
out = {db_v_tnow.columns[n]: pi_oneday[:, n]
       for n in range(n_)}
names = [db_v_tnow.columns[n] for n in range(n_)]
out = pd.DataFrame(out)
out = out[list(names)]
out.to_csv(path+'db_oneday_pl_historical.csv', index=False)
del out
# -

# ## Plots

# +
plt.style.use('arpm')
# instrument P&L plot
fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
f, xp = histogram_sp(pi_tnow_thor[:, n_plot-1], p=p, k_=30)

plt.bar(xp, f, width=xp[1]-xp[0], fc=[0.7, 0.7, 0.7],
        edgecolor=[0.5, 0.5, 0.5])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('P&L', fontsize=17)
plt.title('Ex-ante P&L: '+db_v_tnow.columns[n_plot-1], fontsize=20, fontweight='bold')

add_logo(fig, set_fig_size=False)
plt.show()
