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

# # s_checklist_historical_step03 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_historical_step03&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-3-historical).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.estimation.conditional_fp import conditional_fp
from arpym.estimation.effective_num_scenarios import effective_num_scenarios
from arpym.estimation.exp_decay_fp import exp_decay_fp
from arpym.statistics.scoring import scoring
from arpym.statistics.smoothing import smoothing
from arpym.tools.colormap_fp import colormap_fp
from arpym.tools.histogram_sp import histogram_sp
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step03-parameters)

# +
# flexible probabilities parameters
tau_hl_prior = 4*252  # half-life parameter for time conditioning (days)
tau_hl_smooth = 21  # half-life parameter for VIX smoothing (days)
tau_hl_score = 5*21  # half-life parameter for VIX scoring (days)
alpha = 0.7  # proportion of obs. included in range for state conditioning

# modeled invariant to plot
i_plot = 1
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step03-implementation-step00): Load data

# +
path = '~/databases/temporary-databases/'

# VIX (used for time-state conditioning)
vix_path = '~/databases/global-databases/derivatives/db_vix/data.csv'
db_vix = pd.read_csv(vix_path, usecols=['date', 'VIX_close'],
                     index_col=0, parse_dates=True)

# Quest for invariance
# invariant series
db_invariants_series = pd.read_csv(path+'db_invariants_series_historical.csv',
                                   index_col=0, parse_dates=True)
epsi = db_invariants_series.values
dates = db_invariants_series.index
t_, i_ = np.shape(epsi)
risk_drivers_names = db_invariants_series.columns
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step03-implementation-step01): Define market state indicator

# +
# time and state conditioning on smoothed and scored VIX returns
# state indicator: VIX compounded return realizations
db_vix['c_vix'] = np.log(db_vix).diff()

# extract data for analysis period
c_vix = db_vix.c_vix[dates].values

# smoothing
z_smooth = smoothing(c_vix, tau_hl_smooth)

# scoring
z = scoring(z_smooth, tau_hl_score)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step03-implementation-step02): Set the flexible probabilities

# +
# target value
z_star = z[-1]
# prior probabilities
p_prior = exp_decay_fp(t_, tau_hl_prior)
# posterior probabilities
p = conditional_fp(z, z_star, alpha, p_prior)
# effective number of scenarios
ens = effective_num_scenarios(p)

print('Effective number of scenarios is', int(round(ens)))
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step03-implementation-step03): Save databases

# +
# flexible probabilities
out = pd.DataFrame({'dates' : pd.Series(dates), 'p': pd.Series(p)})
out.to_csv(path+'db_estimation_flexprob.csv', index=None)
del out

# market indicator for flexible probabilities
out = pd.DataFrame({'z': z}, index=dates)
out.index.name= 'dates'
out.to_csv(path+'db_estimation_z.csv')
del out
# -

# ## Plots

# +
plt.style.use('arpm')

# VIX
myFmt = mdates.DateFormatter('%d-%b-%Y')
date_tick = np.arange(0, t_-1, 200)
fig1 = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
ax1 = fig1.add_subplot(311)
plt.plot(dates, z, color=[0, 0, 0], lw=1.15)
plt.title('Market state', fontweight='bold', fontsize=20)
plt.xticks(dates[date_tick], fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([min(dates), max(dates)])
ax1.xaxis.set_major_formatter(myFmt)
plt.plot(dates, z_star*np.ones(len(dates)), color='red', lw=1.25)
plt.legend(['Market state', 'Target value'], fontsize=17)

# flexible probabilities
ax2 = fig1.add_subplot(312)
plt.bar(dates, p.flatten(), color='gray')
plt.xlim([min(dates), max(dates)])
plt.title('Time and state conditioning flexible probabilities',
          fontweight='bold', fontsize=20)
plt.xticks(dates[date_tick], fontsize=14)
plt.yticks([], fontsize=14)
plt.xlim([min(dates), max(dates)])
ax2.xaxis.set_major_formatter(myFmt)

# flexible probabilities scatter for invariant i_plot
ax3 = fig1.add_subplot(313)
grey_range = np.r_[np.arange(0, 0.6 + 0.01, 0.01), .85]
[color_map, p_colors] = colormap_fp(p, np.min(p),
                                    np.max(p), grey_range,
                                    0, 10, [10, 0])
p_colors = p_colors.T

plt.xticks(dates[date_tick], fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([min(dates), max(dates)])
plt.scatter(dates, epsi[:, i_plot-1], s=30, c=p_colors, marker='.',
            cmap=color_map)
plt.title(risk_drivers_names[i_plot-1] + ' observation weighting',
          fontweight='bold', fontsize=20)
ax3.xaxis.set_major_formatter(myFmt)
add_logo(fig1, location=1, set_fig_size=False)
fig1.tight_layout()

# marginal distributions
n_bins = 10 * np.log(t_)

hfp = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
ax = hfp.add_subplot(111)

# HFP histogram
f_eps, x_eps = histogram_sp(epsi[:, i_plot-1],
                            p=p,
                            k_=n_bins)
bar_width = x_eps[1] - x_eps[0]
plt.bar(x_eps, f_eps.flatten(), width=bar_width, fc=[0.7, 0.7, 0.7],
        edgecolor=[0.5, 0.5, 0.5])

plt.title(risk_drivers_names[i_plot-1] + ' invariant distribution',
         fontweight='bold', fontsize=20)
plt.xlabel('Invariant', fontsize=17)
add_logo(hfp, location=1, set_fig_size=False)
hfp.tight_layout()
