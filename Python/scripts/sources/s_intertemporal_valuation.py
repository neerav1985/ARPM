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

# # s_intertemporal_valuation [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_intertemporal_valuation&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_intertemporal_valuation).

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.stats import chi, multivariate_normal, lognorm
from scipy.integrate import nquad
from mpl_toolkits import mplot3d
from matplotlib.collections import PolyCollection

from arpym.pricing.numeraire_mre import numeraire_mre
from arpym.statistics.simulate_normal import simulate_normal
from arpym.tools.histogram_sp import histogram_sp
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_intertemporal_valuation-parameters)

# +
j_ = 15  # number of element of the partition
mu = 0.05  # drift of underlying BM
sig2 = 0.2  # variance of underlying BM
r_min = 0.001  # lower bound for risk-free rate
r_max = 0.03  # upper bound for risk-free rate
v_tnow = 20  # current value
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_intertemporal_valuation-implementation-step01): Brownian motion parameters

# +
# expectation vector
mu_v = np.array([np.log(v_tnow)+mu, np.log(v_tnow)+2*mu])
# covariance matrix
sig2_v = np.array([[sig2, sig2], [sig2, 2*sig2]])
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_intertemporal_valuation-implementation-step02): Risk-free numeraire

# +
# risk-free rates
r_01 =  np.random.uniform(low=r_min, high=r_max, size=1)
r_12 =  np.random.uniform(low=r_min, high=r_max, size=j_)
# risk-free numeraire
v_rn_1 = np.ones(j_)*(1+r_01)
v_rn_2 = np.ones((j_,j_))*(1+r_01)
for j in np.arange(j_):
    v_rn_2[j,:]= v_rn_2[j,:]*(1+r_12[j])
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_intertemporal_valuation-implementation-step03): Partition

# +
# partition for the log price
x_j = np.append(np.append(-np.inf, 
                      np.linspace(np.log(v_tnow)+2*mu-2*np.sqrt(2*sig2),
                                  np.log(v_tnow)+2*mu+2*np.sqrt(2*sig2),j_-1)),
                      np.inf)
# partition for the value
v_j = np.exp(x_j)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_intertemporal_valuation-implementation-step04): Scenarios and probabilities for the adapted value process

# +
p_1 = np.zeros(j_)
v_delta_1 = np.zeros(j_)
p_2 = np.zeros((j_, j_))
v_delta_2 = np.zeros((j_, j_))

# scenarios for Monte Carlo integration
l_ = 100000
u = simulate_normal(mu_v, sig2_v, l_)
u = np.exp(u)

# set functions for numerical integration to solve importance sampling issues
def scen_num_1(x1,x2):
    return (1/x2)*multivariate_normal.pdf(np.array([np.log(x1),np.log(x2)]), mu_v, sig2_v)
def scen_num_2(x1,x2):
    return (1/x1)*multivariate_normal.pdf(np.array([np.log(x1),np.log(x2)]), mu_v, sig2_v)
def scen_den(x1,x2):
    return (1/x1)*(1/x2)*multivariate_normal.pdf(np.array([np.log(x1),np.log(x2)]), mu_v, sig2_v)

for j1 in np.arange(j_):
    # nodes and probabilities for apdated process at t = 1
    count = (u[:,0]>=v_j[j1])*(u[:,0]<v_j[j1+1])
    p_1[j1] = np.sum(count)/l_
    if p_1[j1]!= 0:
    # node for set v^j1 via Monte-Carlo intregration
        v_delta_1[j1] = np.sum(u[:,0]*count)/(l_*p_1[j1])
    else: 
    # node for set v^j1 via numerical quadrature
        p_1[j1] = nquad(scen_den,[[v_j[j1],v_j[j1+1]],[0,np.inf]])[0]
        v_delta_1[j1] = nquad(scen_num_1,[[v_j[j1],v_j[j1+1]],[0,np.inf]])[0]/p_1[j1]
    for j2 in np.arange(j_):
        # nodes and probabilities for apdated process at t = 2
        count = count*(u[:,1]>=v_j[j2])*(u[:,1]<v_j [j2+1])
        p_2[j1,j2] = np.sum(count)/l_
        if p_2[j1,j2] != 0:      
            # node for set v^j1 x v^j2 via Monte-Carlo intregration
            v_delta_2[j1,j2] = np.sum(u[:,1]*count)/(l_*p_2[j1,j2]) 
        else:
        # use numerical quadrature if importance sampliing
            p_2[j1,j2] = nquad(scen_den,[[v_j [j1],v_j[j1+1]],[v_j [j2],v_j[j2+1]]])[0]
            v_delta_2[j1,j2] = nquad(scen_num_2,[[v_j[j1],v_j [j1+1]],[v_j [j2],v_j[j2+1]]])[0]/p_2[j1,j2]
                
                
p_2 = p_2/np.sum(p_2)
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_intertemporal_valuation-implementation-step05): Cumilative SDF via minimum relative entropy

# +
# cumulative SDF process at time t=1
v_pay = np.append(v_rn_1.reshape(j_,1), v_delta_1.reshape(j_, 1), axis=1)
v_t = np.array([1, v_tnow])

# cumulative stochastic discount factor at t = 2
_, sdf_delta_1 = numeraire_mre(v_pay, v_t, p=p_1, k=0)

sdf_delta_2 = np.ones((j_, j_))
sdf_delta_12 = np.ones((j_, j_))
for j in np.arange(j_):
    v_pay_2 = np.append(v_rn_2[j, :].reshape(j_, 1), v_delta_2[j, :].reshape(j_, 1), axis=1)
    v_t_2 = np.array([v_rn_1[j], v_delta_1[j]])
    _, sdf_delta_12[j, :] = numeraire_mre(v_pay_2, v_t_2, p=p_2[j,:]/p_1[j], k=0)

    # cumulative stochastic discount factor at t = 2
    sdf_delta_2[j, :] = sdf_delta_1[j]*sdf_delta_12[j, :]
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_intertemporal_valuation-implementation-step06): Numeraire probabilities for risk-free numeraire

# +
pnum_1 = np.zeros(j_);
pnum_2 = np.ones((j_,j_))
d_2 = np.ones((j_,j_))
d_1 = np.ones(j_)

for j in np.arange(j_):
    # scenarios for Radon-Nikodym numeraire
    d_1[j]=sdf_delta_1[j]*v_rn_1[j]
    d_2[j,:]=sdf_delta_2[j,:]*v_rn_2[j,:]
    # num probabilities at t = 1
    pnum_1[j] = d_1[j]*p_1[j]
    # num probabilities at t = 2
    pnum_2[j,:] = d_2[j,:]*p_2[j,:]   
# -

# ## Plots

# +
plt.style.use('arpm')
u_color = [60/255, 149/255, 145/255]

# set figure specification
fig1 = plt.figure(1, figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

ax1_1 = fig1.add_axes([0.10, 0.5, 0.40, 0.45])

# histograms
v_t1 = v_delta_1*sdf_delta_1
v_t2 = v_delta_2.reshape(j_**2)*sdf_delta_2.reshape(j_**2)
v_max = max(np.max(v_t1), np.max(v_t2)) 
v_min = min(np.min(v_t1), np.min(v_t2)) 

f_t1, x_t1 = histogram_sp(v_t1, p=p_1, k_=j_)
f_t2, x_t2 = histogram_sp(v_t2, p=p_2.reshape(j_**2), k_=int(j_**2/5))

x_lim = max(np.max(f_t1), np.max(f_t2))*3.3
y_lim = (v_max-v_tnow)*1.05

ax1_1.set_title(r'$\mathit{Sdf}^{\Delta}_{t}V_{t}^{\Delta}$ w.r.t. $\mathbb{P}$', fontsize=20)

# v_tnow
ax1_1.scatter([0], [v_tnow], marker='o', color=u_color, label=r'$v_{t_{\mathit{now}}}$')

# at time t = t_now + 1
ax1_1.barh(x_t1, f_t1, left=x_lim/5,  height=x_t1[1]-x_t1[0], facecolor=u_color, edgecolor=[0, 0, 0])
ax1_1.plot(x_lim/5*np.ones(100), np.linspace(0,v_tnow+y_lim, 100), color='black', linewidth=0.5)

# at time t = t_now + 2
ax1_1.barh(x_t2, f_t2, left=3*x_lim/5, height=x_t2[1]-x_t2[0], facecolor=u_color, edgecolor=[0, 0, 0])
ax1_1.plot(3*x_lim/5*np.ones(100), np.linspace(0, v_tnow+y_lim, 100), color='black', linewidth=0.5)

# expectation
x = np.linspace(0, x_lim, 100)
ax1_1.plot(x, v_tnow*np.ones(100), '-r', linewidth=2, label='expectation')

ax1_1.set_ylim([0, v_tnow+y_lim])
ax1_1.set_xlim([0, x_lim])
plt.xticks([0, x_lim/5, 3*x_lim/5], [r'$t_{\mathit{now}}$', r'$t_{\mathit{now}}+1$', r'$t_{\mathit{now}}+2$'], fontsize = 14)
plt.legend()

# check martingality

v_mart_1 = np.ones(j_)
for j in np.arange(j_):
    p_cond = p_2[j,:]/p_1[j]
    v_mart_1[j] = np.sum(v_delta_2[j,:]*sdf_delta_2[j, :]*p_cond)

ax2_1 = fig1.add_axes([0.55, 0.5, 0.45, 0.45])
plt.gca().set_aspect('equal', adjustable='box')
plt.title('One-period linear pricing equation identity', fontsize = 20, pad = 10)
plt.plot([np.min(v_mart_1)*0.9, np.max(v_mart_1)*1.1], [np.min(v_mart_1)*0.9, np.max(v_mart_1)*1.1], 'r')
plt.scatter(v_delta_1*sdf_delta_1, v_mart_1, marker='o')
plt.axis([np.min(v_mart_1)*0.9, np.max(v_mart_1)*1.1, np.min(v_mart_1)*0.9, np.max(v_mart_1)*1.1])
plt.ylabel(r'$v_{t_{\mathit{now}}+1}^{j_{t_{\mathit{now}}}, j_{t_{\mathit{now}}} + 1}$', fontsize = 18)
plt.xlabel(r'$\mathbb{E}_{t_{\mathit{now}}+1}\{ \mathit{Sdf}^{\Delta}_{t_{\mathit{now}}+1 \rightarrow t_{\mathit{now}}+2}V_{ t_{\mathit{now}}+2}^{\Delta} \}(\omega^{j_{t_{\mathit{now}}}, j_{t_{\mathit{now}}} + 1})$', fontsize = 18)
plt.legend(['identity line'])
add_logo(fig1, ax2_1, location=4, alpha=0.8, set_fig_size=False)

# heatmaps for values
v_delta_max = max(np.max(v_delta_1), np.max(v_delta_2)) # upper bound for the values to be displayed
v_delta_min = min(np.min(v_delta_1), np.min(v_delta_2))  # lower bound for the values to be displayed

# at time t = t_now
left, width = -.15, .8
top = left + width

ax3_1 = fig1.add_axes([0.1, 0.295, 0.36, 0.05])
ax3_1.imshow(np.array([v_tnow]).reshape((1, 1)), cmap=cm.jet, vmin=v_delta_min , vmax=v_delta_max, aspect='auto')
ax3_1.autoscale(False)
plt.xticks(np.linspace(0, 1, 1, dtype=int), np.linspace(1, 1, 1, dtype=int))
plt.title(r'$V^{\Delta}_{t}$', fontsize = 20, pad = 10)
plt.xticks(np.linspace(0, 1, 1, dtype=int), np.linspace(1, 1, 1, dtype=int))
plt.yticks([])
plt.grid(False)
ax3_1.text(left-0.03, top, r'$t_{\mathit{now}}$',
           horizontalalignment='center',
           verticalalignment='center',
           rotation=0,
           transform=ax3_1.transAxes,
           fontsize = 14)

# at time t = t_now + 1
v_t1_trunc = v_delta_1.copy()
v_t1_trunc[v_t1_trunc < v_delta_min] = v_delta_min 
v_t1_trunc[v_t1_trunc > v_delta_max] = v_delta_max 

ax4_1 = fig1.add_axes([0.1, 0.18, 0.36, 0.05])
ax4_1.imshow(v_t1_trunc.reshape((1, j_)), cmap=cm.jet, vmin=v_delta_min , vmax=v_delta_max, aspect='auto')
ax4_1.autoscale(False)
plt.yticks([])
plt.xticks(np.linspace(0, j_ - 1, 3, dtype=int), np.linspace(1, j_, 3, dtype=int))
plt.yticks([])
ax4_1.text(left, top, r'$t_{\mathit{now}}+1$',
           horizontalalignment='center',
           verticalalignment='center',
           rotation=0,
           transform=ax4_1.transAxes,
           fontsize = 14)

# at time t = t_now + 2
v_t2_trunc = v_delta_2.copy()
v_t2_trunc[v_t2_trunc < v_delta_min] = v_delta_min 
v_t2_trunc[v_t2_trunc > v_delta_max] = v_delta_max 

ax5_1 = fig1.add_axes([0.1, 0.065, 0.36, 0.05])
ax5_1.imshow(v_t2_trunc.reshape((1, j_**2)), cmap=cm.jet, vmin=v_delta_min , vmax=v_delta_max, aspect='auto')
ax5_1.autoscale(False)
plt.xlabel('Scenario', fontsize = 14,  labelpad=7)
plt.xticks(np.linspace(0, j_ **2- 1, 3, dtype=int), np.linspace(1, j_ **2, 3, dtype=int))
plt.yticks([])
plt.grid(False)
ax5_1.text(left, top, r'$t_{\mathit{now}}+2$',
           horizontalalignment='center',
           verticalalignment='center',
           rotation=0,
           transform=ax5_1.transAxes,
           fontsize = 14)

# scale
ax6_1 = fig1.add_axes([0.5, 0.065, 0.005, 0.28])
cbar = np.linspace(v_delta_max, v_delta_min, j_**2)
plt.imshow(cbar.reshape(-1, 1), cmap=cm.jet, vmin=v_delta_min, vmax=v_delta_max, aspect='auto')
ax6_1.autoscale(False)
plt.xticks([])
tick = np.linspace(0, j**2-1, 6, dtype=int)
plt.yticks(tick, np.round(cbar[tick], decimals=1))
plt.title('Scale', fontsize = 20, pad = 10)

# heatmaps for rnd derivative
sdf_max = max(np.max(sdf_delta_1), np.max(sdf_delta_2)) # upper bound for the values to be displayed
sdf_min = min(np.min(sdf_delta_1), np.min(sdf_delta_2))  # lower bound for the values to be displayed

# at time t = t_now
sdf_tnow = np.array([1])
ax7_1 = fig1.add_axes([0.55, 0.295, 0.36, 0.05])
ax7_1.set_title(r'$\mathit{Sdf}^{\Delta}_{t}$', fontsize = 20, pad = 10)
ax7_1.imshow(sdf_tnow.reshape((1, 1)), cmap=cm.jet, vmin=sdf_min, vmax=sdf_max, aspect='auto')
#ax7_1.autoscale(False)
plt.xticks(np.linspace(0, 1, 1, dtype=int), np.linspace(1, 1, 1, dtype=int))
plt.yticks([])
plt.grid(False)

# at time t = t_now + 1
sdf_1_trunc = sdf_delta_1.copy()
sdf_1_trunc[sdf_1_trunc < sdf_min] = sdf_min
sdf_1_trunc[sdf_1_trunc > sdf_max] = sdf_max

sdf_2_trunc = sdf_delta_2.copy()
sdf_2_trunc[sdf_2_trunc < sdf_min] = sdf_min
sdf_2_trunc[sdf_2_trunc > sdf_max] = sdf_max

ax8_1 = fig1.add_axes([0.55, 0.18, 0.36, 0.05])
ax8_1.imshow(sdf_1_trunc.reshape((1, j_)), cmap=cm.jet, vmin=sdf_min, vmax=sdf_max, aspect='auto')
ax8_1.autoscale(False)
plt.xticks(np.linspace(0, j_ - 1, 3, dtype=int), np.linspace(1, j_, 3, dtype=int))
plt.yticks([])
plt.grid(False)

# at time t = t_now + 2
ax9_1 = fig1.add_axes([0.55, 0.065, 0.36, 0.05])
ax9_1.imshow(sdf_2_trunc.reshape((1, j_**2)), cmap=cm.jet, vmin=sdf_min, vmax=sdf_max, aspect='auto')
ax9_1.autoscale(False)
plt.xticks(np.linspace(0, j_**2- 1, 3, dtype=int), np.linspace(1, j_ **2, 3, dtype=int))
plt.yticks([])
plt.xlabel('Scenario', fontsize = 14,  labelpad=7)
plt.grid(False)

# scale
ax10_1 = fig1.add_axes([0.95, 0.065, 0.005, 0.28])
cbar = np.linspace(sdf_max, sdf_min, j_**2)
plt.imshow(cbar.reshape(-1, 1), cmap=cm.jet, vmin=sdf_min, vmax=sdf_max, aspect='auto')
ax10_1.autoscale(False)
plt.xticks([])
tick = np.linspace(0, j**2-1, 6, dtype=int)
plt.yticks(tick, np.round(cbar[tick], decimals=1))
plt.title('Scale', fontsize = 20, pad = 10)


# +
# set figure specification
fig2 = plt.figure(1, figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

ax1 = fig2.add_axes([0.10, 0.5, 0.40, 0.45])

# histograms
v_rn_t1 = v_delta_1/v_rn_1
v_rn_t2 = (v_delta_2/v_rn_2).reshape(j_**2)
v_rn_max = max(np.max(v_rn_t1), np.max(v_rn_t2)) # upper bound for the values to be displayed
v_rn_min = min(np.min(v_rn_t1), np.min(v_rn_t2))  # lower bound for the values to be displayed

f_rn_t1, x_rn_t1 = histogram_sp(v_rn_t1, p=pnum_1, k_=j_)
f_rn_t2, x_rn_t2 = histogram_sp(v_rn_t2, p=pnum_2.reshape(j_**2), k_=int(j_**2/5))
x_rn_lim = max(np.max(f_rn_t1), np.max(f_rn_t2))*3.3
y_rn_lim = (v_rn_max-v_tnow)*1.05

plt.title(r'$V_{t}^{\Delta}/V^{\mathit{num}, \Delta}_{t}$ w.r.t. $\mathbb{P}^{\mathit{num}}$', fontsize=20)

# v_tnow
ax1.scatter([0], [v_tnow], marker='o', color=u_color, label=r'$v_{t_{\mathit{now}}}$')

# at time t = t_now + 1
ax1.barh(x_rn_t1, f_rn_t1, left=x_rn_lim/5,  height=x_rn_t1[1]-x_rn_t1[0], facecolor=u_color, edgecolor=[0, 0, 0])
ax1.plot(x_rn_lim/5*np.ones(100), np.linspace(0,v_tnow+y_rn_lim, 100), color='black', linewidth=0.5)

# at time t = t_now + 2
plt.barh(x_rn_t2, f_rn_t2, left=3*x_rn_lim/5, height=x_rn_t2[1]-x_rn_t2[0], facecolor=u_color, edgecolor=[0, 0, 0])
plt.plot(3*x_rn_lim/5*np.ones(100), np.linspace(0, v_tnow+y_rn_lim, 100), color='black', linewidth=0.5)

# expectation
x = np.linspace(0, x_rn_lim, 100)
plt.plot(x, v_tnow*np.ones(100), '-r', linewidth=2, label='expectation')

plt.ylim([0, v_tnow+y_rn_lim])
plt.xlim([0, x_rn_lim])
plt.xticks([0, x_rn_lim/5, 3*x_rn_lim/5], [r'$t_{\mathit{now}}$', r'$t_{\mathit{now}}+1$', r'$t_{\mathit{now}}+2$'], fontsize = 14)
plt.legend()

# check martingality
v_mart_num_1 = np.ones(j_)
for j in np.arange(j_):
    p_cond = pnum_2[j,:]/pnum_1[j]
    v_mart_num_1[j] = np.sum(v_delta_2[j,:]*p_cond/v_rn_2[j,:])

ax2 = fig2.add_axes([0.55, 0.5, 0.45, 0.45])
plt.gca().set_aspect('equal', adjustable='box')
plt.title('One-period linear pricing equation identity', fontsize = 20, pad = 10)
plt.plot([np.min(v_mart_num_1)*0.9, np.max(v_mart_num_1)*1.1], [np.min(v_mart_num_1)*0.9, np.max(v_mart_num_1)*1.1], 'r')
plt.scatter(v_mart_num_1, v_delta_1/v_rn_1, marker='o')
plt.axis([np.min(v_mart_num_1)*0.9, np.max(v_mart_num_1)*1.1, np.min(v_mart_num_1)*0.9, np.max(v_mart_num_1)*1.1])
plt.ylabel(r'$v_{t_{\mathit{now}}+1}^{j_{t_{\mathit{now}}}, j_{t_{\mathit{now}}} + 1}/v^{\mathit{num}, \Delta}_{t_{\mathit{now}}+1}$', fontsize = 18)
plt.xlabel(r'$\mathbb{E}^{\mathit{num}}_{t_{\mathit{now}}+1}\{ V_{ t_{\mathit{now}}+2}^{\Delta}/V^{\mathit{num}, \Delta}_{ t_{\mathit{now}}+2} \}(\omega^{j_{t_{\mathit{now}}}, j_{t_{\mathit{now}}} + 1})$', fontsize = 18)
plt.legend(['identity line'])
add_logo(fig2, ax2, location=4, alpha=0.8, set_fig_size=False)

# heatmaps for values
# at time t = t_now
left, width = -.15, .8
top = left + width

ax3 = fig2.add_axes([0.1, 0.295, 0.36, 0.05])
ax3.imshow(np.array([v_tnow]).reshape((1, 1)), cmap=cm.jet, vmin=v_rn_min, vmax=v_rn_max, aspect='auto')
ax3.autoscale(False)
plt.xticks(np.linspace(0, 1, 1, dtype=int), np.linspace(1, 1, 1, dtype=int))
plt.title(r'$V^{\Delta}_{t}/V^{\mathit{num}, \Delta}_{t}$', fontsize = 20, pad = 10)
plt.xticks(np.linspace(0, 1, 1, dtype=int), np.linspace(1, 1, 1, dtype=int))
plt.yticks([])
plt.grid(False)
ax3.text(left-0.03, top, r'$t_{\mathit{now}}$',
         horizontalalignment='center',
         verticalalignment='center',
         rotation=0,
         transform=ax3.transAxes,
         fontsize = 14)

# at time t = t_now + 1
v_rn_t1_trunc = v_rn_t1.copy()
v_rn_t1_trunc[v_rn_t1_trunc < v_rn_min] = v_rn_min
v_rn_t1_trunc[v_rn_t1_trunc > v_rn_max] = v_rn_max

ax4 = fig2.add_axes([0.1, 0.18, 0.36, 0.05])
ax4.imshow(v_rn_t1_trunc.reshape((1, j_)), cmap=cm.jet, vmin=v_rn_min, vmax=v_rn_max, aspect='auto')
ax4.autoscale(False)
plt.yticks([])
plt.xticks(np.linspace(0, j_ - 1, 3, dtype=int), np.linspace(1, j_, 3, dtype=int))
plt.yticks([])
ax4.text(left, top, r'$t_{\mathit{now}}+1$',
         horizontalalignment='center',
         verticalalignment='center',
         rotation=0,
         transform=ax4.transAxes,
         fontsize = 14)

# at time t = t_now + 2
v_rn_t2_trunc = (v_rn_t2).copy()
v_rn_t2_trunc[v_rn_t2_trunc < v_rn_min] = v_rn_min
v_rn_t2_trunc[v_rn_t2_trunc > v_rn_max] = v_rn_max

ax5 = fig2.add_axes([0.1, 0.065, 0.36, 0.05])
ax5.imshow(v_rn_t2_trunc.reshape((1, j_**2)), cmap=cm.jet, vmin=v_rn_min, vmax=v_rn_max, aspect='auto')
ax5.autoscale(False)
plt.xlabel('Scenario', fontsize = 14,  labelpad=7)
plt.xticks(np.linspace(0, j_ **2- 1, 3, dtype=int), np.linspace(1, j_ **2, 3, dtype=int))
plt.yticks([])
plt.grid(False)
ax5.text(left, top, r'$t_{\mathit{now}}+2$',
         horizontalalignment='center',
         verticalalignment='center',
         rotation=0,
         transform=ax5.transAxes,
         fontsize = 14)
# scale
ax6 = fig2.add_axes([0.5, 0.065, 0.005, 0.28])
cbar = np.linspace(v_rn_max, v_rn_min, j_**2)
plt.imshow(cbar.reshape(-1, 1), cmap=cm.jet, vmin=v_rn_min, vmax=v_rn_max, aspect='auto')
ax6.autoscale(False)
plt.xticks([])
tick = np.linspace(0, j**2-1, 6, dtype=int)
plt.yticks(tick, np.round(cbar[tick], decimals=0))
plt.title('Scale', fontsize = 20, pad = 10)

# heatmaps for rnd derivative
d_max = max(np.max(d_1), np.max(d_2)) # upper bound for the values to be displayed
d_min = min(np.min(d_1), np.min(d_2))  # lower bound for the values to be displayed

# at time t = t_now
d_tnow = np.array([1])
ax7 = fig2.add_axes([0.55, 0.295, 0.36, 0.05])
ax7.set_title(r'$D^{\Delta}_{t}$', fontsize = 20, pad = 10)
ax7.imshow(d_tnow.reshape((1, 1)), cmap=cm.jet, vmin = d_min, vmax = d_max, aspect='auto')
ax7.autoscale(False)
plt.xticks(np.linspace(0, 1, 1, dtype=int), np.linspace(1, 1, 1, dtype=int))
plt.yticks([])
plt.grid(False)

# at time t = t_now + 1
d_1_trunc = d_1.copy()
d_1_trunc[d_1_trunc < 0] = 0
d_1_trunc[d_1_trunc > d_max] = d_max

d_2_trunc = d_2.copy()
d_2_trunc[d_2_trunc < 0] = 0
d_2_trunc[d_2_trunc > d_max] = d_max

ax8 = fig2.add_axes([0.55, 0.18, 0.36, 0.05])
ax8.imshow(d_1_trunc.reshape((1, j_)), cmap=cm.jet,  vmin = d_min, vmax = d_max, aspect='auto')
ax8.autoscale(False)
plt.xticks(np.linspace(0, j_ - 1, 3, dtype=int), np.linspace(1, j_, 3, dtype=int))
plt.yticks([])
plt.grid(False)

# at time t = t_now + 2
ax9 = fig2.add_axes([0.55, 0.065, 0.36, 0.05])
ax9.imshow(d_2_trunc.reshape((1, j_**2)), cmap=cm.jet, vmin = d_min, vmax = d_max,  aspect='auto')
ax9.autoscale(False)
plt.xticks(np.linspace(0, j_**2- 1, 3, dtype=int), np.linspace(1, j_ **2, 3, dtype=int))
plt.yticks([])
plt.xlabel('Scenario', fontsize = 14,  labelpad=7)
plt.grid(False)

# scale
ax10 = fig2.add_axes([0.95, 0.065, 0.005, 0.28])
cbar = np.linspace(d_max, d_min, j_**2)
plt.imshow(cbar.reshape(-1, 1), cmap=cm.jet, vmin = d_min, vmax = d_max, aspect='auto')
ax10.autoscale(False)
plt.xticks([])
tick = np.linspace(0, j_**2-1, 6, dtype=int)
plt.yticks(tick, np.round(cbar[tick], decimals=1))
plt.title('Scale', fontsize = 20, pad = 10)


# +
v_1_grid = np.linspace(np.min(v_delta_1), np.max(v_delta_1), 100)
v_2_grid = np.linspace(np.min(v_delta_2), np.max(v_delta_2), 100)

f_logn_t1, x_logn_t1 = histogram_sp(v_delta_1, p=p_1, k_=int(j_))
f_logn_t2, x_logn_t2 = histogram_sp(v_delta_2.reshape(j_**2), p=p_2.reshape(j_**2), k_=40)

pdf_logn_t1 = lognorm.pdf(v_1_grid, np.sqrt(sig2_v[0,0]), scale=np.exp(mu_v[0]))

pdf_logn_t2 = lognorm.pdf(v_2_grid,  np.sqrt(sig2_v[1,1]), scale=np.exp(mu_v[1]))

fig3 = plt.figure(1, figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

ax3_1 = fig3.add_axes([0.08, 0.2, 0.40, 0.65])

plt.bar(x_logn_t1, f_logn_t1, width=x_logn_t1[1]-x_logn_t1[0], facecolor=u_color, edgecolor=[0, 0, 0])
plt.plot(v_1_grid, pdf_logn_t1, color='b', lw=1.5, label='analytical pdf')
plt.xlabel(r'$v_{t_{\mathit{now}}+1}^{\Delta}$', fontsize = 18)
plt.legend(fontsize = 18)

ax3_2 = fig3.add_axes([0.55, 0.2, 0.40, 0.65])
plt.bar(x_logn_t2, f_logn_t2, width=x_logn_t2[1]-x_logn_t2[0], facecolor=u_color, edgecolor=[0, 0, 0])
plt.plot(v_2_grid, pdf_logn_t2, color='b', lw=1.5, label='analytical pdf')
plt.xlabel(r'$v_{t_{\mathit{now}}+2}^{\Delta}$', fontsize = 18)
plt.legend(fontsize = 18)
add_logo(fig3, ax3_2, location=4, alpha=0.8, set_fig_size=False)
