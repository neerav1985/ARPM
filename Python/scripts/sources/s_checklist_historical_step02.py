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

# # s_checklist_historical_step02 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_historical_step02&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-2-historical).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.estimation.fit_garch_fp import fit_garch_fp
from arpym.statistics.invariance_test_copula import invariance_test_copula
from arpym.statistics.invariance_test_ellipsoid import invariance_test_ellipsoid
from arpym.statistics.invariance_test_ks import invariance_test_ks
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step02-parameters)

i_plot = 1  # select the invariant to be tested (i = 1,...,i_)
lag_ = 5  # lag used in invariance tests

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step02-implementation-step00): Load data

# +
path = '~/databases/temporary-databases/'

# market risk drivers
db_riskdrivers_series = pd.read_csv(path+'db_riskdrivers_series_historical.csv',
                                    index_col=0, parse_dates=True)
x = db_riskdrivers_series.values
dates = pd.to_datetime(np.array(db_riskdrivers_series.index))
risk_drivers_names = db_riskdrivers_series.columns.values

# additional information
db_riskdrivers_tools = pd.read_csv(path+'db_riskdrivers_tools_historical.csv')
n_stocks = int(db_riskdrivers_tools.n_stocks.dropna())
d_implvol = int(db_riskdrivers_tools.d_implvol.dropna())

del db_riskdrivers_tools

t_ = len(dates)-1  # length of the invariants time series

# initialize temporary databases
db_invariants = {}
db_nextstep = {}
db_garch_param = {}
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step02-implementation-step01): GARCH(1,1) fit on stocks log-values

for i in range(n_stocks):
    # time series of risk driver increment
    dx = np.diff(x[:, i])
    # fit parameters
    par, sig2, epsi = fit_garch_fp(dx)
    # store next-step function and invariants
    db_invariants[i] = np.array(epsi)
    db_garch_param[i] = dict(zip(['a', 'b', 'c', 'mu'] + \
                                 ['sig2_'+str(t).zfill(3) for t in range(t_)],
                                 np.append(par, sig2)))
    db_nextstep[i] = 'GARCH(1,1)'

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step02-implementation-step02): GARCH(1,1) fit on S&P index log-values

# time series of risk driver increment
dx = np.diff(x[:, n_stocks])
# fit parameters
par, sig2, epsi = fit_garch_fp(dx)
# store next-step function and invariants
db_invariants[n_stocks] = np.array(epsi)
db_garch_param[n_stocks] = dict(zip(['a', 'b', 'c', 'mu'] + \
                             ['sig2_'+str(t).zfill(3) for t in range(t_)],
                             np.append(par, sig2)))
db_nextstep[n_stocks] = 'GARCH(1,1)'

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step02-implementation-step03): Random walk fit on options log-implied volatility

for i in range(n_stocks+1, n_stocks+1+d_implvol):
    db_invariants[i] = np.diff(x[:, i])
    db_nextstep[i] = 'Random walk'

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step02-implementation-step04): Save databases

# +
dates = dates[1:]

# all market invariants
out = pd.DataFrame({risk_drivers_names[i]: db_invariants[i]
                    for i in range(len(db_invariants))}, index=dates)
out = out[list(risk_drivers_names[:len(db_invariants)])]
out.index.name = 'dates'
out.to_csv(path+'db_invariants_series_historical.csv')
del out

# next-step models for all invariants
out = pd.DataFrame({risk_drivers_names[i]: db_nextstep[i]
                    for i in range(len(db_nextstep))}, index=[''])
out = out[list(risk_drivers_names[:len(db_nextstep)])]
out.to_csv(path+'db_invariants_nextstep_historical.csv',
           index=False)
del out

# parameters in GARCH(1,1) models
out = pd.DataFrame({risk_drivers_names[i]: db_garch_param[i]
                    for i in range(len(db_garch_param))})
out = out[list(risk_drivers_names[:len(db_garch_param)])]
out.to_csv(path+'db_invariants_garch_param.csv')
del out
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step02-implementation-step05): Perform invariance tests

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
