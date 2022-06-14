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

# # s_checklist_montecarlo_step02 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_montecarlo_step02&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-2).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.estimation.fit_trans_matrix_credit import fit_trans_matrix_credit
from arpym.estimation.fit_var1 import fit_var1
from arpym.statistics.invariance_test_copula import invariance_test_copula
from arpym.statistics.invariance_test_ellipsoid import invariance_test_ellipsoid
from arpym.statistics.invariance_test_ks import invariance_test_ks
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step02-parameters)

tau_hl_credit = 5  # half-life parameter for credit fit (years)
i_plot = 1  # select the invariant to be tested (i = 1,...,i_)
lag_ = 5  # lag used in invariance tests

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step02-implementation-step00): Load data

# +
path = '~/databases/temporary-databases/'

# invariants for stocks, S&P and options
db_invariants_series = pd.read_csv(path+'db_invariants_series_historical.csv',
                                   index_col=0, parse_dates=True)
dates = pd.to_datetime(np.array(db_invariants_series.index))
t_ = len(dates)
i_historical = db_invariants_series.shape[1]
db_invariants = {}
for i in range(i_historical):
    db_invariants[i] = np.array(db_invariants_series.iloc[:, i])

# next step models for stocks, S&P and options
db_invariants_nextstep = pd.read_csv(path+'db_invariants_nextstep_historical.csv')
db_nextstep = dict(zip(range(i_historical), db_invariants_nextstep.values.squeeze()))

# market risk drivers
db_riskdrivers_series = pd.read_csv(path+'db_riskdrivers_series.csv',
                                    index_col=0, parse_dates=True)
x = db_riskdrivers_series.values
risk_drivers_names = db_riskdrivers_series.columns.values

# credit risk drivers
db_riskdrivers_credit = pd.read_csv(path+'db_riskdrivers_credit.csv',
                                    index_col=0, parse_dates=True)
dates_credit = np.array(db_riskdrivers_credit.index).astype('datetime64[D]')

# additional information
db_riskdrivers_tools = pd.read_csv(path+'db_riskdrivers_tools.csv')
n_stocks = int(db_riskdrivers_tools.n_stocks.dropna())
d_implvol = int(db_riskdrivers_tools.d_implvol.dropna())
n_bonds = int(db_riskdrivers_tools.n_bonds.dropna())
tlast_credit = np.datetime64(db_riskdrivers_tools.tlast_credit[0], 'D')
c_ = int(db_riskdrivers_tools.c_.dropna())
ratings_param = db_riskdrivers_tools.ratings_param.dropna()

i_bonds = n_bonds*4  # 4 NS parameters x n_bonds
ind_ns_bonds = np.arange(n_stocks+1+d_implvol,
                         n_stocks+1+d_implvol+i_bonds)

# number of obligors
n_obligors = db_riskdrivers_credit.iloc[:, :c_+1]

# cumulative number of transitions
n_cum_trans = db_riskdrivers_credit.iloc[:, c_+1:(c_+1)**2]
from_to_index = pd.MultiIndex.from_product([ratings_param, ratings_param],
                                           names=('rating_from', 'rating_to'))
mapper = {}
for col in n_cum_trans:
    (rating_from, _, rating_to) = col[12:].partition('_')
    mapper[col] = (rating_from, rating_to)
n_cum_trans = n_cum_trans.rename(columns=mapper) \
                                     .reindex(columns=from_to_index).fillna(0)

del db_riskdrivers_tools, db_riskdrivers_credit
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step02-implementation-step01): AR(1) fit of Nelson-Siegel parameters

# +
# initialize temporary database
db_ar1_param = {}

# the fit is performed only on non-nan entries
t_bonds = np.sum(np.isfinite(x[:, ind_ns_bonds[0]]))

x_obligor = np.zeros((t_bonds, i_bonds))
epsi_obligor = np.zeros((t_bonds-1, i_bonds))

b_ar_obligor = np.zeros(i_bonds)  # initialize AR(1) parameter
for i in range(i_bonds):
    # risk driver (non-nan entries)
    x_obligor[:, i] = x[t_-t_bonds+1:, ind_ns_bonds[i]]
    # fit parameter
    b_ar_obligor[i], _, _ = fit_var1(x_obligor[:, i])
    # invariants
    epsi_obligor[:, i] = x_obligor[1:, i]-b_ar_obligor[i]*x_obligor[:-1, i]

# store the next-step function and the extracted invariants
k = 0
for i in ind_ns_bonds:
    db_invariants[i] = np.r_[np.full(t_-t_bonds+1, np.nan),
                             epsi_obligor[:, k]]
    db_nextstep[i] = 'AR(1)'
    db_ar1_param[i] = {'b': b_ar_obligor[k]}
    k = k+1
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step02-implementation-step02): Credit migrations: time-homogeneous Markov chain

# +
# array format
n_cum_trans = n_cum_trans.values.reshape((-1, c_+1, c_+1), order='C')

# annual credit transition matrix
p_credit = fit_trans_matrix_credit(dates_credit,
                                   n_obligors.values,
                                   n_cum_trans, tau_hl_credit)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step02-implementation-step03): Save databases

# +
# all market invariants
out = pd.DataFrame({risk_drivers_names[i]: db_invariants[i]
                    for i in range(len(db_invariants))}, index=dates)
out = out[list(risk_drivers_names[:len(db_invariants)])]
out.index.name = 'dates'
out.to_csv(path+'db_invariants_series.csv')
del out

# next-step models for all invariants
out = pd.DataFrame({risk_drivers_names[i]: db_nextstep[i]
                    for i in range(len(db_nextstep))}, index=[''])
out = out[list(risk_drivers_names[:len(db_nextstep)])]
out.to_csv(path+'db_invariants_nextstep.csv',
           index=False)
del out

# parameters in AR(1) models
out = pd.DataFrame({risk_drivers_names[i]: db_ar1_param[i]
                    for i in ind_ns_bonds})
out.to_csv(path+'db_invariants_ar1_param.csv')
del out

# annual credit transition matrix
out = pd.DataFrame({'p_credit': pd.Series(p_credit.reshape(-1))})
out.to_csv(path+'db_invariants_p_credit.csv',
           index=None)
del out
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step02-implementation-step04): Perform invariance tests

# +
plt.style.use('arpm')

invar = db_invariants[i_plot-1][~np.isnan(db_invariants[i_plot-1])]

_ = invariance_test_ellipsoid(invar, lag_)
fig_ellipsoid = plt.gcf()
fig_ellipsoid.set_dpi(72.0)
fig_ellipsoid.set_size_inches(1280.0/72.0, 720.0/72.0)
add_logo(fig_ellipsoid, set_fig_size=False)
plt.show()

invariance_test_ks(invar)
fig_ks = plt.gcf()
fig_ks.set_dpi(72.0)
fig_ks.set_size_inches(1280.0/72.0, 720.0/72.0)
add_logo(fig_ks, set_fig_size=False)
plt.tight_layout()

_ = invariance_test_copula(invar, lag_)
fig_cop = plt.gcf()
fig_cop.set_dpi(72.0)
fig_cop.set_size_inches(1280.0/72.0, 720.0/72.0)
plt.tight_layout()
