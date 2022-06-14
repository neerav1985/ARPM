#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_attribution_hedging [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_attribution_hedging&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-hedging-bs-vs-fod-appl).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from arpym.tools.histogram_sp import histogram_sp
from arpym.tools.logo import add_logo
from arpym.pricing.bsm_function import bsm_function
from arpym.statistics.meancov_sp import meancov_sp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_hedging-parameters)

m = 90  # index of the hedging horizon within the monitoring time vector {t_m} 

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_hedging-implementation-step00): Load data

# +
# (generated by script s_pricing_call_option_value.py)

path = '~/databases/temporary-databases/'

db = pd.read_csv(path + 'db_call_data.csv', index_col=0)

m_ = int(np.array(db['m_'].iloc[0]))  # upper limit of monitoring horizon
t_m = np.array(pd.to_datetime(db['t_m'].values.reshape(-1)), dtype='datetime64[D]')
t_end = pd.to_datetime(db['t_end'].iloc[0])  # call option expiry date
t_end = np.datetime64(t_end, 'D')
k_strike = int(np.array(db['k_strike'].iloc[0]))  # call option strike
y = np.array(db['y_rf'].iloc[0])  # risk-free yield for selected hor.'s
log_sigma = db['log_sigma_atm'].iloc[0] #log-implied volatility at the money
j_ = int(np.array(db['j_'].iloc[0]))  # number of simulations
# j_ simulated realizations of call option value over m_ monitoring times
vcall_m = np.array(db['v_call_thor'].iloc[:j_*(m_+1)]).reshape((j_, (m_+1)))
# value of underlying for m_ monitoring times
v_stock_m = np.exp(np.array(db['log_s'].iloc[:j_*(m_+1)]).reshape((j_, (m_+1))))

# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_hedging-implementation-step01): Compute the Black-Scholes-Merton delta

delta_t = np.busday_count(t_m[0], t_end)/252
sigma = np.exp(log_sigma)
d1 = (np.log(v_stock_m[0, 0]/k_strike) + (y+sigma**2/2)*delta_t) /\
                            (sigma * np.sqrt(delta_t))
delta_bms = norm.cdf(d1)  # BMS delta
beta_bms = delta_bms*v_stock_m[0, 0]/vcall_m[0, 0]  # BMS beta

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_hedging-implementation-step02): Compute the factors on demand delta at the selected horizon

# call option linear return
r_call = (vcall_m[:, [m]] / vcall_m[:, [0]]) - 1
# stock linear return
r_stock = (v_stock_m[:, [m]] / v_stock_m[:, [0]]) - 1
rr = np.r_['-1', r_call, r_stock]
expectation, cov = meancov_sp(rr)
beta_fod = cov[0, 1]/cov[1, 1]  # FOD beta
delta_fod = beta_fod*vcall_m[0, 0]/v_stock_m[0, 0]  # FOD delta

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_hedging-implementation-step04): Compute the return of the hedged portfolio

# BMS return of hedged portfolio
r_bms = r_call - beta_bms*r_stock
# FOD return of hedged portfolio
r_fod = r_call - beta_fod*r_stock

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_attribution_hedging-implementation-step05): Compute the Black-Scholes-Merton curve and payoff

# +
s = np.arange(600, 1801)  # range of values for underlying
l_ = len(s)
bs_curve = np.zeros(l_)
bs_payoff = np.zeros(l_)

for l in range(l_):
    moneyness = np.log(s[l]/k_strike)/np.sqrt(delta_t)
    # range of call option values
    bs_curve[l] = bsm_function(s[l], y, sigma, moneyness,
                               delta_t)
    # range of the payoff values
    bs_payoff[l] = (s[l] - k_strike)*int(k_strike <= s[l]) +\
        0*int(k_strike >= s[l])

# current value of the call option
moneyness = np.log(v_stock_m[0,0]/k_strike)/np.sqrt(delta_t)
bs_curve_current = bsm_function(v_stock_m[0, 0], y, sigma, moneyness, delta_t)
# -

# ## Plots

# +
plt.style.use('arpm')

lgray = [0.8, 0.8, 0.8]  # light grey
dgray = [0.4, 0.4, 0.4]  # dark grey

fig = plt.figure()
plt.plot(np.squeeze(v_stock_m[:, m]), np.squeeze(vcall_m[:, m]),
         markerSize=3, color=dgray, marker='.', linestyle='none',
         label='call option values')
plt.plot(s, bs_curve.flatten(), color='k',
         label='BMS call option value')
plt.plot(v_stock_m[0, 0], bs_curve_current, color='r', marker='.',
         markersize=15, label='BMS call option current value')
plt.plot(s, np.squeeze(bs_curve_current + delta_bms*(s - v_stock_m[0, 0])),
         color='r', label='BMS hedge')
plt.plot(s, np.squeeze(bs_curve_current + delta_fod*(s - v_stock_m[0, 0])),
         color='b', label='FoD hedge')
plt.plot(s, bs_payoff.flatten(), color='k')
plt.legend()
plt.ylabel('call option value')
plt.xlabel('underlying')
plt.title('Time to horizon: '+str(m_)+' days')

add_logo(fig)
plt.tight_layout()

fig = plt.figure()
f_hist, x_hist = histogram_sp(r_call.flatten(), k_=100)
plt.bar(x_hist, f_hist.flatten(), (max(x_hist)-min(x_hist))/(len(x_hist)-1),
        color=lgray, edgecolor=lgray, linewidth=2)
f1_hist, x1_hist = histogram_sp(r_bms.flatten(), k_=100)
plt.plot(x1_hist, f1_hist.flatten(), color='r', label='BMS hedged pdf')
f2_hist, x2_hist = histogram_sp(r_fod.flatten(), k_=100)
plt.plot(x2_hist, f2_hist.flatten(), color='b', label='FoD hedged pdf')
plt.legend()
plt.title('Repriced call option return')

add_logo(fig)
plt.tight_layout()
