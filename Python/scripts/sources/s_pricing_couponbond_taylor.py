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

# # s_pricing_couponbond_taylor [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_pricing_couponbond_taylor&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-coupon-bond-taylor-approx).

# +
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from arpym.pricing.bond_value import bond_value
from arpym.statistics.moments_mvou import moments_mvou
from arpym.statistics.meancov_sp import meancov_sp
from arpym.statistics.saddle_point_quadn import saddle_point_quadn
from arpym.tools.histogram_sp import histogram_sp
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond_taylor-parameters)

deltat = 3  # time to horizon (in months)
j_ = 1000  # number of scenarios
c = 0.04  # annualized coupons (percentage of the face value)
dy = 10 ** -4   # numerical differentiation step (duration and convexity)

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond_taylor-implementation-step00): Upload data

# +
# import database generated by script s_projection_var1_yields
path = '~/databases/temporary-databases'

df = pd.read_csv(path + '/db_proj_scenarios_yield.csv', header=0)
tau = np.array(list(map(int, df.columns)))  # times to maturity
d_ = tau.shape[0]
df1 = pd.read_csv(path + '/db_proj_scenarios_yield_par.csv', header=0)
theta = np.array(df1['theta'].iloc[:d_ ** 2].values.reshape(d_, d_))
mu_mvou = np.array(df1['mu_mvou'].iloc[:d_])
sig2_mvou = np.array(df1['sig2_mvou'].iloc[:d_ ** 2].values.reshape(d_, d_))
df2 = pd.read_csv(path + '/db_proj_dates.csv', header=0, parse_dates=True)
t_m = df2.values
t_m = np.array(pd.to_datetime(df2.values.reshape(-1)), dtype='datetime64[D]')
m_ = t_m.shape[0]-1
deltat_m = np.busday_count(t_m[0], t_m[1])

if deltat > m_:
    print(" Projection doesn't have data until given horizon!!! Horizon lowered to ", m_)
    deltat = m_
# number of monitoring times
m_ = deltat
t_m = t_m[:m_+1]
t_now = t_m[0]

j_m_, _ = df.shape
x_t_now_t_hor = np.array(df).reshape(j_, int(j_m_/j_), d_)
x_t_now_t_hor = x_t_now_t_hor[:j_, :m_+1, :]
x_t_now = x_t_now_t_hor[0, 0, :].reshape(1, -1)
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond_taylor-implementation-step01): Record dates and coupons of the bond

# +
t_end = np.datetime64('2025-12-22')  # time of maturity

# number of coupons
k_ = int(np.busday_count(t_now, t_end)/252)

# record dates
r = np.busday_offset(t_now, np.arange(1, k_+1)*252)

# coupons
coupon = c * np.ones(k_)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond_taylor-implementation-step02): Scenarios for bond value path

# +
v_t_m = np.array([bond_value(t, x_t_now_t_hor[:, m, :], tau, coupon,
                              r, 'y')
                   for m, t in enumerate(t_m)]).T
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond_taylor-implementation-step03): Scenarios for normalized bond P&L

# +
v_t_now = v_t_m[0, 0]
r_t_now_t_m = (v_t_m - v_t_now) / v_t_now
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond_taylor-implementation-step04): Scenario-probability expectations and standard deviations

# +
mu_r_t_m = np.zeros(m_+1)
sig_r_t_m = np.zeros(m_+1)

for m in range(len(t_m)):
    mu_r_t_m[m], sig_2_r_t_m = meancov_sp(r_t_now_t_m[:, m].reshape(-1, 1))
    sig_r_t_m[m] = np.sqrt(sig_2_r_t_m)
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond_taylor-implementation-step05): Numerical yield

# +
def y_bond_tnow(y):
    eq = bond_value(t_now, y*np.ones((1,d_)), tau, coupon, r, 'y') - v_t_now
    return eq

y_bond = fsolve(y_bond_tnow, 0)
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond_taylor-implementation-step06): Effective key rate durations

# +
y_t_now = x_t_now_t_hor[[0], 0, :]

dy_vec = dy * np.eye(d_)
dur_hat_d = np.zeros((d_, 1))

for d in np.arange(d_):
    v_bond_d1 = bond_value(t_now, y_t_now + dy_vec[d, :], tau, coupon, r, 'y')
    v_bond_d2 = bond_value(t_now, y_t_now - dy_vec[d, :], tau, coupon, r, 'y')
    dur_hat_d[d] = - ((v_bond_d1 - v_bond_d2 ).reshape(-1) / (2 * dy * v_t_now))
# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond_taylor-implementation-step07): Effective convexity

# +
v_bond_y1 = bond_value(t_now, y_t_now + dy, tau, coupon, r, 'y')
v_bond_y2 = bond_value(t_now, y_t_now - dy, tau, coupon, r, 'y')
conv_hat = (v_bond_y1 - 2 * v_t_now + v_bond_y2) / (dy**2 * v_t_now)
# -

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond_taylor-implementation-step08): Moments of yield increment

# +
# moments of yield increment
_, mu_y_dt, sig2_y_dt = moments_mvou(y_t_now.reshape(-1),
                                     [deltat*21],
                                     theta, mu_mvou, sig2_mvou)
mu_dt = mu_y_dt - y_t_now.reshape(-1)
sig2_dt = sig2_y_dt
# -

# ## [Step 9](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond_taylor-implementation-step09): Distribution of first order approximation

# +
# parameters of normal distribution
mu_r_1st = y_bond * deltat/12 - dur_hat_d.T @ mu_dt
sig2_r_1st = dur_hat_d.T @ sig2_dt @ dur_hat_d

# normal pdf
n_points = 500
x_grid = np.linspace(mu_r_t_m[-1] - 10*sig_r_t_m[-1],
                   mu_r_t_m[-1] + 10*sig_r_t_m[-1], n_points)
norm_pdf = stats.norm.pdf(x_grid, mu_r_1st,
                          np.sqrt(sig2_r_1st[0, 0]))
# -

# ## [Step 10](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_couponbond_taylor-implementation-step10): Distribution of second order approximation

# +
# parameters of quadratic normal distribution
gamma = np.ones((d_, d_)) * conv_hat / d_**2

# quad normal pdf via saddle point approximation
_, quadn_pdf = saddle_point_quadn(x_grid, y_bond * deltat/12, -dur_hat_d.reshape(-1),
                                  gamma, mu_dt, sig2_dt)
# -

# ## Plots

# +
plt.style.use('arpm')
lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey
lblue = [0.27, 0.4, 0.9]  # light blue
orange = [0.94, 0.35, 0]  # orange
j_sel = 35  # selected MC simulations
scale = (np.busday_count(np.min(t_m), np.max(t_m))/252) * 0.015

# simulated path, mean and standard deviation

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

t_axis = np.busday_count(t_now, t_m)/252
plt.plot(t_axis.reshape(-1, 1), r_t_now_t_m[:j_sel, :].T, color=lgrey, lw=1)
plt.ylabel('Bond return')
plt.xlabel('horizon')
l2 = plt.plot(t_axis, mu_r_t_m + sig_r_t_m, color='r')
plt.plot(t_axis, mu_r_t_m - sig_r_t_m, color='r')
l1 = plt.plot(t_axis, mu_r_t_m, color='g')
plt.grid(False)

# empirical pdf
p = np.ones(j_) / j_
y_hist, x_hist = histogram_sp(r_t_now_t_m[:, -1], k_=10*np.log(j_))
y_hist = y_hist * scale  # adapt the hist height to the current xaxis scale
shift_y_hist = deltat/12 + y_hist

emp_pdf = plt.barh(x_hist, y_hist, left=t_axis[-1],
                   height=x_hist[1]-x_hist[0], facecolor=lgrey,
                   edgecolor=lgrey)

plt.plot(shift_y_hist, x_hist, color=dgrey, lw=1)
plt.plot([t_axis[-1], t_axis[-1]], [x_hist[0], x_hist[-1]], color=dgrey,
         lw=0.5)

# normal approximation
shift_norm_pdf = np.array([t_axis[-1]+i for i in (norm_pdf * scale)])
l3 = plt.plot(shift_norm_pdf, x_grid, color=lblue, lw=1)

# quadn approximation
shift_quadn_pdf = np.array([t_axis[-1]+i for i in (quadn_pdf * scale)])
l4 = plt.plot(shift_quadn_pdf, x_grid, color=orange, lw=1)

# axis lim
plt.xlim([np.min(t_axis), np.max([np.max(t_axis), np.max(shift_quadn_pdf),
          np.max(shift_norm_pdf),
          t_axis[0]+np.max(shift_y_hist)]) +
          np.min(np.diff(t_axis))])

# legend
plt.legend(handles=[l1[0], l2[0], emp_pdf[0], l3[0], l4[0]],
           labels=['mean', ' + / - st.deviation', 'emp. pdf',
                   'first order approx.', 'second order approx'])
plt.title('Taylor approximation P&L of a coupon bond')
add_logo(fig, location=4, set_fig_size=False)
plt.tight_layout()
