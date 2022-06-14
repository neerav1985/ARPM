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

# # s_default_merton_model [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_default_merton_model&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-merton-struct-model).

# +
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import norminvgauss
import matplotlib.pyplot as plt

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_default_merton_model-parameters)

v_asset_t = 10  # initial asset value
mu_asset = 0  # "percentage" drift of the GBM
sigma_asset = 0.3  # "percentage" volatility of the GBM
j_ = 5  # number of trajectories for the plot
v_liab_t = 6  # initial value of the liabilities
r = 0.1  # liabilities growth coefficient
n_steps = 252  # number of time steps between t and t+1
l_thresholds = np.array([-0.895, -0.800, -0.680, -0.565, -0.430, -0.285, 0, float('Inf')])  # log-leverage thresholds

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_default_merton_model-implementation-step01): Monte Carlo scenarios for the path of the asset value

delta_tm = 1/n_steps
epsi = (mu_asset - 0.5 * sigma_asset ** 2) * delta_tm +\
       sigma_asset * np.sqrt(delta_tm) * stats.norm.rvs(size=(n_steps, j_))  # simulation of normal shocks
epsi[0, 0] = 0
v_asset = v_asset_t * np.exp(np.cumsum(epsi, axis=0))  # simulation of asset value

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_default_merton_model-implementation-step02): Distribution of the assets value at time t+1

n_grid = 100
lognscale = np.exp(np.log(v_asset_t) + (mu_asset - 0.5 * sigma_asset ** 2))
x_grid = np.linspace(stats.lognorm.ppf(.01, sigma_asset, scale=lognscale),
                     stats.lognorm.ppf(.99, sigma_asset, scale=lognscale),
                     n_grid)
f_vasset_tp1 = stats.lognorm.pdf(x_grid, sigma_asset, scale=lognscale)

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_default_merton_model-implementation-step03): Liabilities evolution

t_plot = np.linspace(0, 1, n_steps)
v_liab = v_liab_t * np.exp(t_plot * r)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_default_merton_model-implementation-step04): Log-leverage and probability of default

l_t = np.log(v_liab_t / v_asset_t)  # log-leverage
mu_l = -mu_asset + 0.5 * sigma_asset ** 2 + r  # log-leverage mean
sigma_l = sigma_asset  # log-leverage variance
dd_t = (l_t + mu_l) / sigma_l  # distance to default
p_def_l = stats.norm.cdf(dd_t)  # probability of default

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_default_merton_model-implementation-step05): Map the log-leverage thresholds

# +
v_asset_thresholds_t = v_liab_t * np.exp(-l_thresholds)  # asset value thresholds at time t
v_asset_thresholds_tp1 = v_liab[-1] * np.exp(-l_thresholds)  # asset value thresholds at time t+1

c_t = np.digitize(v_asset_t, v_asset_thresholds_t, right=True)  # initial rating
c_tp1 = np.digitize(v_asset[-1,:], v_asset_thresholds_tp1, right=True)  # final rating for each trajectory
# -

# ## Plots

# +
plt.style.use('arpm')

fig = plt.figure()

lblue = [0.58, 0.80, 0.87]  # light blue
lgreen = [0.76, 0.84, 0.61]  # light green
lpurple = [0.70, 0.64, 0.78]  # light purple
rat_col = np.array([[0/255, 166/255, 0/255],  #AAA
            [75/255, 209/255, 29/255],  #AA
            [131/255, 213/255, 32/255],  #A
            [188/255, 217/255, 34/255], #BBB
            [221/255, 195/255, 36/255],  #BB
            [225/255, 144/255, 38/255],  #B
            [229/255, 92/255, 40/255],  #CCC
            [233/255, 42/255, 47/255]])  #D

color_t = rat_col[c_t,:]
color_tp1 = rat_col[c_tp1[-1],:] # color based on the final rating of the first trajectory

# balance sheet at time t
ax1 = plt.subplot2grid((3, 3), (0, 0))
plt.bar(0, v_asset_t, width=1, color=color_t)
plt.bar(1, np.max([v_asset_t - v_liab_t, 0]), width=1, facecolor=lgreen)
plt.bar(1, v_liab_t, bottom=[np.max(v_asset_t - v_liab_t, 0)], width=1,
        facecolor=lpurple)
plt.axis([0, 3, 0, x_grid[-1]])
ax1.xaxis.set_visible(False)
plt.title('Balance sheet at time t')
ax1.legend(['Assets', 'Equities', 'Liabilities'], loc='best')

# balance sheet at time t+1
ax3 = plt.subplot2grid((3, 3), (0, 2))
plt.bar(0, v_asset[-1, -1], width=1, color=color_tp1)
plt.bar(1, np.max([v_asset[-1, -1] - v_liab[-1], 0]), width=1,
        facecolor=lgreen)
plt.bar(1, v_liab[-1], bottom=np.max([v_asset[-1, -1] - v_liab[-1], 0]),
        width=1, facecolor=lpurple)
plt.axis([0, 3, 0., x_grid[-1]])
ax3.xaxis.set_visible(False)
plt.title('Balance sheet at time t+1')
ax3.legend(['Assets', 'Equities', 'Liabilities'], loc='best')

# probability of default at time t+1
ax2 = plt.subplot2grid((3, 3), (0, 1))
plt.plot(l_t, p_def_l, 'r.')
plt.ylim([0, 1])
plt.title('Probability of default at time t+1')
plt.xlabel('Log-leverage')
plt.ylabel('Probability')

ax4 = plt.subplot2grid((3, 3), (1, 0), colspan=3, rowspan=2)
plt.xticks([0, 1], ['t', 't+1'])
plt.ylim([x_grid[0], x_grid[-1]])

#ax4.set_yticklabels([])

# liabilities
plt.plot(t_plot, v_liab, color=lpurple, lw=1.5)

# assets
plt.plot(t_plot, v_asset[:, -1], color=lblue, lw=2)
for j in range(j_):
    plt.plot(t_plot, v_asset[:, j], color=lblue, lw=0.75)
plt.plot(0, v_asset_t, '.', color=lblue, markersize=20)

# assets distribution at time t+1
idx_def7 = np.where(x_grid <= v_asset_thresholds_tp1[6])[0]
idx_def6 = np.where(x_grid <= v_asset_thresholds_tp1[5])[0]
idx_def5 = np.where(x_grid <= v_asset_thresholds_tp1[4])[0]
idx_def4 = np.where(x_grid <= v_asset_thresholds_tp1[3])[0]
idx_def3 = np.where(x_grid <= v_asset_thresholds_tp1[2])[0]
idx_def2 = np.where(x_grid <= v_asset_thresholds_tp1[1])[0]
idx_def1 = np.where(x_grid <= v_asset_thresholds_tp1[0])[0]

# highlight solvent vs default areas under the pdf
ax4.fill_betweenx(x_grid[idx_def7], np.ones(len(idx_def7)), 1+f_vasset_tp1[idx_def7],
                  color=rat_col[7], zorder = 35)
ax4.fill_betweenx(x_grid[idx_def6], np.ones(len(idx_def6)), 1+f_vasset_tp1[idx_def6],
                  color=rat_col[6], zorder = 30)
ax4.fill_betweenx(x_grid[idx_def5], np.ones(len(idx_def5)), 1+f_vasset_tp1[idx_def5],
                  color=rat_col[5], zorder = 25)
ax4.fill_betweenx(x_grid[idx_def4], np.ones(len(idx_def4)), 1+f_vasset_tp1[idx_def4],
                  color=rat_col[4], zorder = 20)
ax4.fill_betweenx(x_grid[idx_def3], np.ones(len(idx_def3)), 1+f_vasset_tp1[idx_def3],
                  color=rat_col[3], zorder = 15)
ax4.fill_betweenx(x_grid[idx_def2], np.ones(len(idx_def2)), 1+f_vasset_tp1[idx_def2],
                  color=rat_col[2], zorder = 10)
ax4.fill_betweenx(x_grid[idx_def1], np.ones(len(idx_def1)), 1+f_vasset_tp1[idx_def1],
                  color=rat_col[1], zorder = 5)
ax4.fill_betweenx(x_grid, np.ones(n_grid), 1+f_vasset_tp1, color=rat_col[0], zorder = 0)

# rating lines
rat = np.zeros(len(v_asset_thresholds_t)-1)
for k in range(0, len(v_asset_thresholds_t)-1):
    rat[k] = (v_asset_thresholds_t[k]-x_grid[0])/(x_grid[-1]-x_grid[0])
    
ax4.axvline(x = 0, ymin = 0, ymax = rat[6], color=rat_col[7], lw=4, zorder = 0)
ax4.axvline(x = 0, ymin = rat[6], ymax = rat[5], color=rat_col[6], lw=4, zorder = 0)
ax4.axvline(x = 0, ymin = rat[5], ymax = rat[4], color=rat_col[5], lw=4, zorder = 0)
ax4.axvline(x = 0, ymin = rat[4], ymax = rat[3], color=rat_col[4], lw=4, zorder = 0)
ax4.axvline(x = 0, ymin = rat[3], ymax = rat[2], color=rat_col[3], lw=4, zorder = 0)
ax4.axvline(x = 0, ymin = rat[2], ymax = rat[1], color=rat_col[2], lw=4, zorder = 0)
ax4.axvline(x = 0, ymin = rat[1], ymax = rat[0], color=rat_col[1], lw=4, zorder = 0)
ax4.axvline(x = 0, ymin = rat[0], ymax = x_grid[-1], color=rat_col[0], lw=4, zorder = 0)

ax4.text(-0.03, 5.8, 'D', fontsize=13)
ax4.text(-0.05, 14.5, 'AAA', fontsize=13)

#pdf_border_inf = plt.plot(np.ones(n_grid), x_grid, color='k')
plt.legend(['Liabilities', 'Assets'])

add_logo(fig)
plt.tight_layout()
