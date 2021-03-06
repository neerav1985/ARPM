#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # S_PricingDefaultCouponBond [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=S_PricingDefaultCouponBond&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-coup-bear-bond-credit-risk).

# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))
from collections import namedtuple

import numpy as np
from numpy import arange, array, ones, zeros, std, where, round, mean, log, exp, tile, r_, newaxis

from scipy.stats import binom
from scipy.io import loadmat
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, xlim, xlabel, title

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot
from intersect_matlab import intersect
from HistogramFP import HistogramFP
from SimVAR1MVOU import SimVAR1MVOU
from VAR1toMVOU import VAR1toMVOU
from FitVAR1 import FitVAR1
from BondPrice import BondPrice
from InverseCallTransformation import InverseCallTransformation
from PerpetualAmericanCall import PerpetualAmericanCall
from CashFlowReinv import CashFlowReinv
# -

# ## Up

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_SwapParRates'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_SwapParRates'), squeeze_me=True)

Rates = db['Rates']

try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_DefaultCoupon'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_DefaultCoupon'),
                 squeeze_me=True)  # generated by S_ProjDiscreteMarkovChain setting tau=1/25[1:]1/25[1:]3

p_tau = db['p_tau']
# -

# ## Select the key rates and recover the historical series of the shadow rates

# +
t_end = 3
fPaym = .5
coup_pay_t = arange(.5, t_end+fPaym,fPaym).reshape(1,-1)
t_ = coup_pay_t.shape[1]
dt = 1 / 252
horiz_u = arange(0,t_end+dt,dt)
u_ = len(horiz_u)

# match the db
[Dates, i_u, i_t] = intersect(horiz_u, coup_pay_t)

if len(i_u) != t_:
    raise ValueError('Setup a suitable dt')

timeStep = 1
pick = range(7)
tau_d = array([[1, 2, 5, 7, 10, 15, 30]]).T
y = Rates[pick, ::timeStep]
eta = 0.013
invcy = InverseCallTransformation(y, {1:eta})  # shadow rates
# -

# ## Fit the MVOU to the historical series of the shadow rates

#dinvcy = diff(invcy, 1, 2)
[alpha, b, sig2_U] = FitVAR1(invcy)
# [alpha, b, sig2_U] = FitVAR1(dinvcy, invcy(:,1:-1))
mu, theta, sigma2,_ = VAR1toMVOU(alpha, b, sig2_U, timeStep*1 / 252)
# [mu, theta, sigma2] = FitVAR1MVOU(dinvcy, invcy(:,1:-1), timeStep@1/252)

# ## Project the shadow rates and the Bernoulli variable by using the default probabilities

# +
j_ = 1000
x_0 = tile(invcy[:,[-1]], (1, j_))  # initial setup
X_u = SimVAR1MVOU(x_0, horiz_u[1:].reshape(1,-1), theta, mu, sigma2, j_)
X_u = r_['-1',x_0[...,np.newaxis], X_u]

# Bernoulli variable
idx_i = 6-1  # rating "B"
p_default = zeros(len(horiz_u)-1)
for k in range(len(horiz_u) - 1):
    p_default[k] = p_tau[k][idx_i, -1]

I = zeros((j_, u_-1))
for i in range(len(p_default)):
    if i == 0:
        I[:,[i]] = binom.rvs(1, p_default[i], size=(j_, 1))
    else:
        I[:,[i]] = binom.rvs(1, p_default[i] - p_default[i - 1], size=(j_, 1))
# -

# ## Compute the value of the bond, the reinvested cash-flows and P&L with credit risk

# +
# Coupon-bearing bond value
V_bond_u = zeros((j_, u_))

for u in range(u_):
    time = coup_pay_t[0,coup_pay_t[0] >= horiz_u[u]]-horiz_u[u]  # time interval between the current time and the coupon payment dates
    if all(time == 0):
        coupon = array([[0.04]])
    else:
        coupon = tile(0.04, (1, len(time)))

    V_bond_u[:, u] = BondPrice(X_u[:,:, u], tau_d, coupon, time, 1, 'shadow rates', {'eta':eta})

b_0 = V_bond_u[:, 0]
V_bond_u = V_bond_u[:, 1:]
Mu_V_bond_u = mean(V_bond_u,axis=0,keepdims=True)
Sigma_V_bond_u = std(V_bond_u,axis=0,keepdims=True)

# Reinvested cash-flow stream
Reinv_tk_u = zeros((1, j_, u_))
Reinv_tk_u[0,:, 0] = 0
interp = interp1d(tau_d.flatten(), invcy[:,-1],fill_value='extrapolate')
y_0 = interp(0)
cty_0 = PerpetualAmericanCall(y_0, {'eta':eta})
Reinv_tk_u[0,:, 1] = exp(dt*cty_0)

for k in arange(2,u_):
    interp = interp1d(tau_d.flatten(), X_u[:,:,k],axis=0, fill_value='extrapolate')
    Y_0 = interp(0)
    ctY_0 = PerpetualAmericanCall(Y_0, {'eta':eta})
    Reinv_tk_u[0,:,k] = exp(dt*ctY_0)

# Reinvested cash-flow value
c = ones((t_, 1))*0.04
Cf_u = zeros((j_, u_))
for j in range(j_):
    Cf_u[j,:] = CashFlowReinv(horiz_u.reshape(1,-1), coup_pay_t, i_u, Reinv_tk_u[[0], j,:], c)

cf_0 = Cf_u[:, 0]
Cf_u = Cf_u[:, 1:]
MuCF_u = mean(Cf_u,axis=0,keepdims=True)
SigmaCF_u = std(Cf_u,axis=0,keepdims=True)

# Compute the value of the coupon-bearing bond with credit risk
r_D = 0.7  #
reinv_factor = Reinv_tk_u[[0], j,:].T
reinv_factor = reinv_factor[1:]
for j in range(j_):
    def_idx = where(I[j,:] == 1)[0]
    if def_idx.size == 0:
        pass
    else:
        I[j, def_idx[0]:] = 1
        if def_idx[0] - 1 == 0:  # if default at the first future horizon
            V_bond_u[j, :] = b_0[0]
            Cf_u[j, :] = cf_0[0]
        else:
            V_bond_u[j, def_idx[0]:] = V_bond_u[j, def_idx[0] - 1]  # take the last value before default
            Cf_u[j, def_idx[0]:] = Cf_u[j, def_idx[0] - 1]*reinv_factor[def_idx[0]:].flatten()  # take the last value before default

V_u = r_D*I*V_bond_u + (1 - I)*V_bond_u
# Compute the P&L
PL_u = V_u - tile(b_0[...,newaxis], (1, V_u.shape[1])) + Cf_u
# ## Plot a few paths of the bond, the reinvested cash-flows and the P&L and their histograms at future horizons
pp_ = ones((1, j_)) / j_  # outcomes probabilities
i = 252*2-1
scen1 = where(I[:,-1] == 1)[0]
scen2 = where(I[:,-1] == 0)[0]
n_scens = r_[scen1[:int(round(p_default[-1]*30))],
             scen2[:int(round(1 - p_default[-1])*30)]]

lgrey = [0.8, 0.8, 0.8]  # light grey
dgrey = [0.4, 0.4, 0.4]  # dark grey
f1 = figure()

plot(horiz_u[1:i+1], V_u[n_scens, :i].T, color = lgrey)
# histogram
option = namedtuple('option', 'n_bins')
option.n_bins = round(10*log(j_))
y_hist, x_hist = HistogramFP(V_u[:,[i]].T, pp_, option)
y_hist = y_hist*.02  # adapt the hist height to the current xaxis scale
shift_y_hist = horiz_u[i] + y_hist
# empirical pdf

emp_pdf = plt.barh(x_hist[:-1], shift_y_hist[0]-horiz_u[i], left=horiz_u[i],height=x_hist[1]-x_hist[0],
                   facecolor=lgrey, edgecolor= lgrey)
# border
plot(shift_y_hist[0], x_hist[:-1], color=dgrey)
title('Coupon-bearing bond value')
xlabel('time (years)')
xlim([0, 3]);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

f2 = figure()

plot(horiz_u[1:i+1], Cf_u[n_scens, :i].T, color = lgrey)
# histogram
[y_hist, x_hist] = HistogramFP(Cf_u[:,[i]].T, pp_, option)
y_hist = y_hist*.0025  # adapt the hist height to the current xaxis scale
shift_y_hist = horiz_u[i] + y_hist

# empirical pdf
emp_pdf = plt.barh(x_hist[:-1], shift_y_hist[0]-horiz_u[i], left=horiz_u[i], height=x_hist[1]-x_hist[0],
                   facecolor=lgrey, edgecolor= lgrey)
# border
plot(shift_y_hist[0], x_hist[:-1], color=dgrey)
title('Cash-flows (coupon)')
xlabel('time (years)')
xlim([0, 3]);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

f3 = figure()
plot(horiz_u[1:i+1], PL_u[n_scens, :i].T, color = lgrey)
# histogram
y_hist, x_hist = HistogramFP(PL_u[:,[i]].T, pp_, option)
y_hist = y_hist*.02  # adapt the hist height to the current xaxis scale
shift_y_hist = horiz_u[i] + y_hist
# empirical pdf

emp_pdf = plt.barh(x_hist[:-1], shift_y_hist[0]-horiz_u[i], left=horiz_u[i], height=x_hist[1]-x_hist[0],
                   facecolor=lgrey, edgecolor= lgrey)
# border
plot(shift_y_hist[0], x_hist[:-1], color=dgrey)
title('Coupon-bearing bond and cash-flows P&L')
xlabel('time (years)')
xlim([0, 3]);
# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])

