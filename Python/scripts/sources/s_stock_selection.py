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

# # s_stock_selection [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_stock_selection&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.statistics.meancov_sp import meancov_sp
from arpym.estimation.exp_decay_fp import exp_decay_fp
from arpym.tools.transpose_square_root import transpose_square_root
from arpym.portfolio.obj_tracking_err import obj_tracking_err
from arpym.tools.naive_selection import naive_selection
from arpym.tools.forward_selection import forward_selection
from arpym.tools.backward_selection import backward_selection
from arpym.tools.enet_selection import enet_selection
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-parameters)

n_ = 48  # number of stocks
t_ = 1008  # length of the time series
t_now = '2012-01-01'  # current time
tau_hl = 180  # half life parameter

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-implementation-step00): Upload data

# +
path = '~/databases/global-databases/equities/db_stocks_SP500/'
spx = pd.read_csv(path + 'SPX.csv', index_col=0, parse_dates=['date'])
stocks = pd.read_csv(path + 'db_stocks_sp.csv', skiprows=[0], index_col=0)
# merging datasets
spx_stocks = pd.merge(spx, stocks, left_index=True, right_index=True)
# select data within the date range
spx_stocks = spx_stocks.loc[spx_stocks.index <= t_now].tail(t_)
# remove the stocks with missing values
spx_stocks = spx_stocks.dropna(axis=1, how='any')
date = spx_stocks.index
# upload stocks values
v_stock = np.array(spx_stocks.iloc[:, 2:2+n_])

# upload S&P500 index value
v_sandp = np.array(spx_stocks.SPX_close)
t_ = v_stock.shape[0]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-implementation-step01): Compute linear returns of both benchmark and securities

# stocks return
r_stock = np.diff(v_stock, axis=0)/v_stock[:-1, :]
# S&P500 index return
r_sandp = np.diff(v_sandp, axis=0)/v_sandp[:-1]

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-implementation-step02): Cov. matrix of the joint vector of stocks and bench. returns

# +
# exponential decay probabilities
p = exp_decay_fp(t_ - 1, tau_hl)

# HFP covariance
_, s2_r_stock_r_sandp = meancov_sp(np.concatenate((r_stock, r_sandp.reshape(-1, 1)), axis=1), p)
cv_r_stock = s2_r_stock_r_sandp[:n_, :n_]
cv_r_stock_r_sandp = s2_r_stock_r_sandp[:n_, -1]
cv_r_sandp = s2_r_stock_r_sandp[-1, -1]
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-implementation-step03): Objective function

optim = lambda s: obj_tracking_err(s2_r_stock_r_sandp, s)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-implementation-step04): Portfolio selection via naive routine

w_naive, te_w_naive, s_naive = naive_selection(optim, n_)

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-implementation-step05): Portfolio selection via forward stepwise routine

w_fwd, te_w_fwd, s_fwd = forward_selection(optim, n_)

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-implementation-step06): Portfolio selection via backward stepwise routine

w_bwd, te_w_bwd, s_bwd = backward_selection(optim, n_)

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_stock_selection-implementation-step07): Portfolio selection via elastic nets heuristics

# +
a_eq = np.ones((1, r_stock.shape[1]))
s = 0.01
for p in range(a_eq.shape[0]):
    for n in range(a_eq.shape[1]):
        a_eq[p, n] = (1 + (-s)**(p+n)/np.linalg.norm(a_eq))*a_eq[p, n]
b_eq = np.ones((1, 1))
a_ineq = -np.eye(r_stock.shape[1])
b_ineq = np.zeros((r_stock.shape[1], 1))

q2 = cv_r_stock
q = transpose_square_root(q2, method='Cholesky')
qinv = np.linalg.solve(q, np.eye(n_))
c = -np.atleast_2d(cv_r_stock_r_sandp).T
u = np.sqrt(2*n_)*q.T
v = -np.sqrt(n_/2)*qinv@c

w_enet, _, s_enet, k_lam, lam_vec = enet_selection(v, u, alpha=10**-5,
                                                         a_eq=a_eq, b_eq=b_eq,
                                                         a_ineq=a_ineq, b_ineq=b_ineq,
                                                         a=100,
                                                         eps=10**-9,
                                                         thr=10**-8)
te_w_enet = np.zeros(w_enet.shape[0])
for h in range(w_enet.shape[0]):  # rescale weights
    #weights
    w_enet[h] = w_enet[h]/np.sum(w_enet[h])
    # tracking error
    te_w_enet[h] = np.sqrt(w_enet[h].T@cv_r_stock@w_enet[h]-2*cv_r_stock_r_sandp.T@w_enet[h]+cv_r_sandp)
# -

# ## Plots

# +
plt.style.use('arpm')

mydpi = 72.0 # set these dpi
f = plt.figure(figsize=(1280.0/mydpi,720.0/mydpi),dpi=mydpi)
h3 = plt.plot(np.arange(1, n_+1), np.abs(te_w_naive), color=[.5, .5, .5], lw=2,
              label='naive')
h1 = plt.plot(np.arange(1, n_ + 1), np.abs(te_w_fwd), 'b',
              lw=2, label='forward stepwise')
h2 = plt.plot(np.arange(1, n_ + 1), np.abs(te_w_bwd),
              color=[0.94, 0.3, 0], lw=2,
              label='backward stepwise')
h4 = plt.plot(k_lam[::-1], np.abs(te_w_enet[::-1]), lw=2,
              label='elastic net')
plt.legend(handles=[h3[0], h1[0], h2[0], h4[0]], loc='best')
plt.xlabel('Number of stocks')
ticks = np.arange(0, 10 * (n_ // 10 + 1), 10)
plt.xticks(np.append(1, np.append(ticks, n_)))
plt.xlim([0.5, n_+1])
plt.ylabel('Tracking error')
plt.title('n-choose-k routines comparison', fontweight='bold')

mydpi = 72.0 # set these dpi
f = plt.figure(figsize=(1280.0/mydpi,720.0/mydpi),dpi=mydpi)

plt.ylabel('Weights')
plt.xlabel('Log-lambda')
for n in range(w_enet.shape[1]):
    plt.plot(np.log(lam_vec[lam_vec>0]), w_enet[:, n][lam_vec>0], lw=2)
plt.ylim([0, 1])
