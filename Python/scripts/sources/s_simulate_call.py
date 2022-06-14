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

# # s_simulate_call [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_simulate_call&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-simeucall).

# +
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

from arpym.pricing.bsm_function import bsm_function
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_call-parameters)

# +
j_ = 30  # number of scenarios (=number of basis call options)
delta_t = 60  # time to horizon
s_low = 77.66  # lower bound for the underlying grid
delta_s = 2.9  # tick-size of underlying/strikes at expiry
s_tnow = 120  # underlying current value
r = 2 * 1e-4  # risk-free interest rate
sigma2 = 0.01  # volatility of the underlying
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_call-implementation-step01): Underlying scenarios at horizon and strikes' calls

# +
s_omega_j = s_low + np.arange(1, j_+1, 1).reshape(1, -1)*delta_s
k_j = (s_omega_j - delta_s).T
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_call-implementation-step02): Payoff matrix of basis call options

# +
v_call_pay = delta_s*np.triu(toeplitz(np.arange(1, j_+1)))
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_call-implementation-step03): Current values basis call options

# +
v_call = np.zeros(j_)
for n in range(j_):
    m_n = np.log(s_tnow/k_j[n])/np.sqrt(delta_t)  # # moneynesses
    v_call[n] = bsm_function(s_tnow, r, sigma2, m_n, delta_t)  # current values basis call options
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_simulate_call-implementation-step04): Save databases

# +
out = np.c_[s_omega_j.reshape(-1,1), k_j, v_call]
col = ['s_omega_j', 'k_j', 'v_call']
out = pd.DataFrame(out, columns=col)
out.to_csv('~/databases/temporary-databases/db_simcall.csv')
del out
out = {'s_tnow': pd.Series(s_tnow),
       'delta_s': pd.Series(delta_s),
       'delta_t': pd.Series(delta_t),
       'r': pd.Series(r),
       'sigma2': pd.Series(sigma2)}
out = pd.DataFrame(out)
out.to_csv('~/databases/temporary-databases/db_simcall_tools.csv')
del out
# -

# ## Plots

# +
s_up = s_omega_j[0, -1]  # upper bound for the underlying/strike grid
plt.style.use('arpm')

tick_size = int(j_/6)
col_unit = int(150/j_)

f = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

gs0 = gridspec.GridSpec(1, 1)
gs00 = gridspec.GridSpecFromSubplotSpec(504, 288, subplot_spec=gs0[0],
                                        wspace=0, hspace=1)
ax1 = plt.Subplot(f, gs00[0:469, 10:160])
f.add_subplot(ax1)
ax1.imshow(v_call_pay, cmap=cm.jet, aspect='auto')
plt.title(r'$\mathbf{\mathcal{V}}^{\mathit{call.pay}}$', fontsize = 36)
plt.xlabel('Scenario', fontsize = 24,  labelpad=10)
plt.ylabel('Instrument', fontsize = 24,  labelpad=10)
plt.xticks(np.arange(4, j_+1, tick_size), np.arange(5, j_+1, tick_size), fontsize = 18)
plt.yticks(np.arange(4, j_+1, tick_size), np.arange(5, j_+1, tick_size), fontsize = 18)

ax11 = plt.Subplot(f, gs00[:469, 210:210+col_unit])
f.add_subplot(ax11)
ax11.imshow(v_call.reshape(-1, 1), vmin=0, vmax=s_up - s_low, cmap=cm.jet, aspect='auto')
plt.title(r'$\mathbf{v}^{\mathit{call}}$', fontsize = 36)
plt.xticks([])
plt.yticks(np.arange(4, j_+1, tick_size), np.arange(5, j_+1, tick_size), fontsize=18)
plt.grid(False)

ax12 = plt.Subplot(f, gs00[:469, 270:270+col_unit])
f.add_subplot(ax12)
cbar = np.floor((np.flipud(s_omega_j.T - s_omega_j[0,0])) * 100) / 100
plt.imshow(cbar, cmap=cm.jet, aspect='auto')
plt.title('Scale', fontsize = 36)
plt.xticks([])
plt.yticks([0, tick_size, 2*tick_size, 3*tick_size, 4*tick_size, 5*tick_size, j_-1], 
           [i[0] for i in cbar[[0, tick_size, 2*tick_size, 3*tick_size, 4*tick_size, 5*tick_size, -1]] ],
           fontsize = 18)
plt.grid(False)

add_logo(f, axis=ax1, location=3, size_frac_x=1/12, set_fig_size=False)
