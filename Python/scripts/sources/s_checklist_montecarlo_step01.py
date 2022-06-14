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

# # s_checklist_montecarlo_step01 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_montecarlo_step01&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-1).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.pricing.bootstrap_nelson_siegel import bootstrap_nelson_siegel
from arpym.tools.aggregate_rating_migrations import aggregate_rating_migrations
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step01-parameters)

# +
# corporate bonds
# expiry date of the GE coupon bond to extract
tend_ge = np.datetime64('2013-09-16')
# expiry date of the JPM coupon bond to extract
tend_jpm = np.datetime64('2014-01-15')

# starting ratings following the table:
# "AAA" (0), "AA" (1), "A" (2), "BBB" (3), "BB" (4), "B" (5),
# "CCC" (6), "D" (7)
ratings_tnow = np.array([5,   # initial credit rating for GE (corresponding to B)
                         3])  # initial credit rating for JPM  (corresponding to BBB)

# start of period for aggregate credit risk drivers
tfirst_credit = np.datetime64('1995-01-01')
# end of period for aggregate credit risk drivers
tlast_credit = np.datetime64('2004-12-31')

# index of market risk driver to plot
d_plot = 1
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step01-implementation-step00): Import data

# +
path = '~/databases/temporary-databases/'

# data for stocks, S&P and options
# supporting information
db_riskdrivers_tools = pd.read_csv(path+'db_riskdrivers_tools_historical.csv')
t_first = np.datetime64(db_riskdrivers_tools.t_first[0], 'D')
t_now = np.datetime64(db_riskdrivers_tools.t_now[0], 'D')
t_init = np.datetime64(db_riskdrivers_tools.t_init[0], 'D')
d_ = int(db_riskdrivers_tools.d_.dropna())
d_implvol = int(db_riskdrivers_tools.d_implvol.dropna())
n_stocks = int(db_riskdrivers_tools.n_stocks.dropna())
d_stocks = n_stocks

# risk driver series
db_riskdrivers_series = pd.read_csv(path+'db_riskdrivers_series_historical.csv',
                                    index_col=0, parse_dates=True)
dates = pd.to_datetime(np.array(db_riskdrivers_series.index))
t_ = len(dates)
risk_drivers_names = dict(zip(range(d_), db_riskdrivers_series.columns.values))
db_risk_drivers = {}
for d in range(d_):
    db_risk_drivers[d] = np.array(db_riskdrivers_series.iloc[:, d])

# values at t_now
db_v_tnow = pd.read_csv(path+'db_v_tnow_historical.csv')
n_ = db_v_tnow.shape[1]
v_tnow = dict(zip(range(n_), db_v_tnow.values.squeeze()))
v_tnow_names = dict(zip(range(n_), db_v_tnow.columns.values))

# values at t_init
db_v_tinit = pd.read_csv(path+'db_v_tinit_historical.csv')
v_tinit = dict(zip(range(n_), db_v_tinit.values.squeeze()))

# corporate bonds: GE and JPM
jpm_path = \
    '~/databases/global-databases/fixed-income/db_corporatebonds/JPM/'
db_jpm = pd.read_csv(jpm_path + 'data.csv',
                     index_col=['date'], parse_dates=['date'])
jpm_param = pd.read_csv(jpm_path + 'params.csv',
                        index_col=['expiry_date'], parse_dates=['expiry_date'])
jpm_param['link'] = ['dprice_'+str(i) for i in range(1, jpm_param.shape[0]+1)]

ge_path = '~/databases/global-databases/fixed-income/db_corporatebonds/GE/'
db_ge = pd.read_csv(ge_path + 'data.csv',
                    index_col=['date'], parse_dates=['date'])
ge_param = pd.read_csv(ge_path + 'params.csv',
                       index_col=['expiry_date'], parse_dates=['expiry_date'])
ge_param['link'] = ['dprice_'+str(i) for i in range(1, ge_param.shape[0]+1)]
# select the bond dates
ind_dates_bonds = np.where((db_ge.index >= t_first) &
                           (db_ge.index <= t_now))
dates_bonds = np.intersect1d(db_ge.index[ind_dates_bonds], db_jpm.index)
dates_bonds = dates_bonds.astype('datetime64[D]') 
t_bonds = len(dates_bonds)

# ratings
rating_path = '~/databases/global-databases/credit/db_ratings/'
db_ratings = pd.read_csv(rating_path+'data.csv', parse_dates=['date'])
# ratings_param represents all possible ratings i.e. AAA, AA, etc.
ratings_param = pd.read_csv(rating_path+'params.csv', index_col=0)
ratings_param = np.array(ratings_param.index)
c_ = len(ratings_param)-1
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step01-implementation-step01): Corporate bonds

# +
n_bonds = 2

# GE bond

# extract coupon
coupon_ge = ge_param.loc[tend_ge, 'coupons']/100

# rescaled dirty prices of GE bond
v_bond_ge = db_ge.loc[db_ge.index.isin(dates_bonds)]/100

# computation of Nelson-Siegel parameters for GE bond
theta_ge = np.zeros((t_bonds, 4))
theta_ge = bootstrap_nelson_siegel(v_bond_ge.values, dates_bonds,
                                   np.array(ge_param.coupons/100),
                                   ge_param.index.values.astype('datetime64[D]'))

# risk drivers for bonds are Nelson-Siegel parameters
for d in np.arange(4):
    if d == 3:
        db_risk_drivers[d_+d] = np.sqrt(theta_ge[:, d])
    else:
        db_risk_drivers[d_+d] = theta_ge[:, d]
    risk_drivers_names[d_+d] = 'ge_bond_nel_sieg_theta_' + str(d+1)

# store dirty price of GE bond
# get column variable name in v_bond_ge that selects bond with correct expiry
ge_link = ge_param.loc[tend_ge, 'link']
v_tnow[n_] = v_bond_ge.loc[t_now, ge_link]
v_tinit[n_] = v_bond_ge.loc[t_init, ge_link]
v_tnow_names[n_] = 'ge_bond'

# update counter
d_ = len(db_risk_drivers)
n_ = len(v_tnow_names)

# JPM bond

# extract coupon
coupon_jpm = jpm_param.loc[tend_jpm, 'coupons']/100

# rescaled dirty prices of JPM bond
v_bond_jpm = db_jpm.loc[db_ge.index.isin(dates_bonds)]/100

# computation of Nelson-Siegel parameters for JPM bond
theta_jpm = np.zeros((t_bonds, 4))
theta_jpm = bootstrap_nelson_siegel(v_bond_jpm.values, dates_bonds,
                                   np.array(jpm_param.coupons/100),
                                   jpm_param.index.values.astype('datetime64[D]'))

# risk drivers for bonds are Nelson-Siegel parameters
for d in np.arange(4):
    if d == 3:
        db_risk_drivers[d_+d] = np.sqrt(theta_jpm[:, d])
    else:
        db_risk_drivers[d_+d] = theta_jpm[:, d]
    risk_drivers_names[d_+d] = 'jpm_bond_nel_sieg_theta_'+str(d+1)

# store dirty price of JPM bond
# get column variable name in v_bond_ge that selects bond with correct expiry
jpm_link = jpm_param.loc[tend_jpm, 'link']
v_tnow[n_] = v_bond_jpm.loc[t_now, jpm_link]
v_tinit[n_] = v_bond_jpm.loc[t_init, jpm_link]
v_tnow_names[n_] = 'jpm_bond'

# update counter
d_ = len(db_risk_drivers)
n_ = len(v_tnow)

# fill the missing values with nan's
for d in range(d_stocks+1+d_implvol,
               d_stocks+1+d_implvol+n_bonds*4):
    db_risk_drivers[d] = np.concatenate((np.zeros(t_-t_bonds),
                                         db_risk_drivers[d]))
    db_risk_drivers[d][:t_-t_bonds] = np.NAN
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step01-implementation-step02): Credit 

# +
# extract aggregate credit risk drivers
dates_credit, n_obligors, n_cum_trans, *_ = \
    aggregate_rating_migrations(db_ratings, ratings_param, tfirst_credit,
                                tlast_credit)

# number of obligors in each rating at each t
t_credit = len(dates_credit)  # length of the time series
credit_types = {}
credit_series = {}
for c in np.arange(c_+1):
    credit_types[c] = 'n_oblig_in_state_'+ratings_param[c]
    credit_series[c] = n_obligors[:, c]

d_credit = len(credit_series)

# cumulative number of migrations up to time t for each pair of rating buckets
for i in np.arange(c_+1):
    for j in np.arange(c_+1):
        if i != j:
            credit_types[d_credit] = \
                'n_cum_trans_'+ratings_param[i]+'_'+ratings_param[j]
            credit_series[d_credit] = n_cum_trans[:, i, j]
            d_credit = len(credit_series)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step01-implementation-step03): Save databases

# +
path = '~/databases/temporary-databases/'

# market risk drivers
out = pd.DataFrame({risk_drivers_names[d]: db_risk_drivers[d]
                    for d in range(len(db_risk_drivers))}, index=dates)
out = out[list(risk_drivers_names.values())]
out.index.name = 'dates'
out.to_csv(path+'db_riskdrivers_series.csv')
del out

# values of all instruments at t_now
out = pd.DataFrame({v_tnow_names[n]: pd.Series(v_tnow[n])
                   for n in range(len(v_tnow))})
out = out[list(v_tnow_names.values())]
out.to_csv(path+'db_v_tnow.csv',
           index=False)
del out

# values of all instruments at t_init
out = pd.DataFrame({v_tnow_names[n]: pd.Series(v_tinit[n])
                   for n in range(len(v_tinit))})
out = out[list(v_tnow_names.values())]
out.to_csv(path+'db_v_tinit.csv',
           index=False)
del out

# aggregate credit risk drivers
out = pd.DataFrame({credit_types[d]: credit_series[d]
                    for d in range(d_credit)},
                   index=dates_credit)
out = out[list(credit_types.values())]
out.index.name = 'dates'
out.to_csv(path+'db_riskdrivers_credit.csv')
del out

out = {'n_bonds': pd.Series(n_bonds),
       'tend_ge': pd.Series(tend_ge),
       'coupon_ge': pd.Series(coupon_ge),
       'tend_jpm': pd.Series(tend_jpm),
       'coupon_jpm': pd.Series(coupon_jpm),
       'c_': pd.Series(c_),
       'tlast_credit': pd.Series(tlast_credit),
       'd_credit': pd.Series(d_credit),
       'ratings_tnow': pd.Series(ratings_tnow),
       'ratings_param': pd.Series(ratings_param)}
out = pd.DataFrame(out)
out = pd.concat([db_riskdrivers_tools, out], axis=1, sort=False)
out['d_']=pd.Series(d_)
out.to_csv(path+'db_riskdrivers_tools.csv',
           index=False)
del out
# -

# ## Plots

# +
plt.style.use('arpm')

# plot market risk driver
fig1 = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
plt.plot(dates, db_risk_drivers[d_plot-1])
plt.title(risk_drivers_names[d_plot-1], fontweight='bold', fontsize=20)
plt.xlabel('time (days)', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([dates[0], dates[-1]])
add_logo(fig1, set_fig_size=False)
fig1.tight_layout()

# plot credit risk drivers
# create colormaps
colors = ['#FFFFFF', '#3C9591']
arpm_cmap = LinearSegmentedColormap.from_list('arpm_cmap', colors)
grey_cmap = ListedColormap('#E6E6E6')

fig2 = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
gs = gridspec.GridSpec(1, 4)

same = np.full((c_+1, c_+1), np.nan)
for c in range(c_+1):
    same[c,c] = 1

# number of obligors
ax1 = fig2.add_subplot(gs[0, 0])
max_color=np.max(n_obligors)
plt.imshow(n_obligors[-1, :].reshape(-1, 1),
           cmap=arpm_cmap, vmin=0, vmax=max_color)
plt.grid(False)
plt.title('Number of obligors', fontsize=20, fontweight='bold')
plt.ylabel('Rating class', fontsize=17)
plt.yticks(range(c_+1), ratings_param, fontsize=14)
plt.xticks([])
plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False)

# cumulative transitions
ax2 = fig2.add_subplot(gs[0, 1:])
plt.imshow(n_cum_trans[-1,:,:], cmap=arpm_cmap,
           vmin=0, vmax=max_color)
add_logo(fig2, location=1, set_fig_size=False)
plt.colorbar()
plt.imshow(same, cmap=grey_cmap)
plt.grid(False)

plt.title('Cumulative rating transitions', fontsize=20, fontweight='bold')
plt.xlabel('To', fontsize=17)
plt.ylabel('From', fontsize=17)
plt.xticks(range(c_+1), ratings_param, fontsize=14)
plt.yticks(range(c_+1), ratings_param, fontsize=14)
plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False)
