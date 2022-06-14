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

# # s_checklist_historical_step04 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_historical_step04&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-4-historical).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.statistics.bootstrap_hfp import bootstrap_hfp
from arpym.tools.histogram_sp import histogram_sp
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step04-parameters)

m_ = 1  # number of days to project
j_ = 10000  # number of projection scenarios
d_plot = 1  # index of projected risk driver to plot

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step04-implementation-step00): Load data

# +
path = '~/databases/temporary-databases/'

# Risk drivers identification
# market risk drivers
db_riskdrivers_series = pd.read_csv(path+'db_riskdrivers_series_historical.csv',
                                    index_col=0, parse_dates=True)
x = db_riskdrivers_series.values
risk_drivers_names = db_riskdrivers_series.columns

# risk driver information
db_riskdrivers_tools = pd.read_csv(path+'db_riskdrivers_tools_historical.csv')
d_ = int(db_riskdrivers_tools.d_.dropna())
t_now = np.datetime64(db_riskdrivers_tools.t_now[0], 'D')

# Quest for invariance
# invariant series
db_invariants_series = pd.read_csv(path+'db_invariants_series_historical.csv',
                                   index_col=0, parse_dates=True)
epsi = db_invariants_series.dropna().values
dates = db_invariants_series.dropna().index
t_, i_ = np.shape(epsi)

# next step models
db_invariants_nextstep = pd.read_csv(path+'db_invariants_nextstep_historical.csv')

# next step model parameters
db_invariants_garch_param = pd.read_csv(path+'db_invariants_garch_param.csv',
                                        index_col=0)

# Estimation
# flexible probabilities
db_estimation_flexprob = pd.read_csv(path+'db_estimation_flexprob.csv',
                                     index_col=0, parse_dates=True)
p = db_estimation_flexprob.loc[:, 'p'].values
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step04-implementation-step01): Projection of invariants (bootstrap)

t_hor = np.busday_offset(t_now, m_)
epsi_proj = np.zeros((j_, m_, d_))
for m in range(m_):
    epsi_proj[:, m, :] = bootstrap_hfp(epsi, p, j_)
p_scenario = np.ones(j_)/j_  # projection scenario probabilities

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step04-implementation-step02): Projection of risk drivers

# +
dx_proj = np.empty((j_, m_+1, d_))
x_proj = np.empty((j_, m_+1, d_))
sig2_garch = np.empty((j_, m_+1, d_))

# risk drivers at time t_0=t_now
x_proj[:, 0, :] = x[-1, :]

# initialize parameters for GARCH(1,1) projection
d_garch = [d for d in range(d_)
           if db_invariants_nextstep.iloc[0, d] =='GARCH(1,1)']
for d in d_garch:
    a_garch = db_invariants_garch_param.loc['a'][d]
    b_garch = db_invariants_garch_param.loc['b'][d]
    c_garch = db_invariants_garch_param.loc['c'][d]
    mu_garch = db_invariants_garch_param.loc['mu'][d]
    sig2_garch[:, 0, d] = db_invariants_garch_param.iloc[-1, d]
    dx_proj[:, 0, d] = x[-1, d] - x[-2, d]

# project risk drivers
for m in range(1, m_+1):
    for d in range(d_):
        # risk drivers modeled as random walk
        if db_invariants_nextstep.iloc[0, d] == 'Random walk':
            x_proj[:, m, d] = x_proj[:, m-1, d] + epsi_proj[:, m-1, d]

        # risk drivers modeled as GARCH(1,1)
        elif db_invariants_nextstep.iloc[0, d] == 'GARCH(1,1)':
            a_garch = db_invariants_garch_param.loc['a'][d]
            b_garch = db_invariants_garch_param.loc['b'][d]
            c_garch = db_invariants_garch_param.loc['c'][d]
            mu_garch = db_invariants_garch_param.loc['mu'][d]
            sig2_garch[:, m, d] = c_garch + b_garch*sig2_garch[:, m-1, d] +\
                a_garch*(dx_proj[:, m-1, d] - mu_garch)**2
            dx_proj[:, m, d] = mu_garch +\
                np.sqrt(sig2_garch[:, m, d])*epsi_proj[:, m-1, d]
            x_proj[:, m, d] = x_proj[:, m-1, d] + dx_proj[:, m, d]
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_historical_step04-implementation-step03): Save databases

# +
# projected risk drivers
out = pd.DataFrame({risk_drivers_names[d]:
                   x_proj[:, :, d].reshape((j_*(m_+1),))
                   for d in range(d_)})
out = out[list(risk_drivers_names[:d_].values)]
out.to_csv(path+'db_projection_bootstrap_riskdrivers.csv', index=None)
del out

# additional information
out = pd.DataFrame({'j_': pd.Series(j_),
                    't_hor': pd.Series(t_hor)})
out.to_csv(path+'db_projection_bootstrap_tools.csv', index=None)
del out

# projected scenario probabilities
out = pd.DataFrame({'p': pd.Series(p_scenario)})
out.to_csv(path+'db_scenario_probs_bootstrap.csv', index=None)
del out
# -

# ## Plots

# +
plt.style.use('arpm')

# marginal distributions
n_bins = 10 * np.log(t_)

# projected risk driver distribution
proj_dist = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
ax = proj_dist.add_subplot(111)
f_eps, x_eps = histogram_sp(x_proj[:, m_, d_plot-1],
                            p=p_scenario,
                            k_=n_bins)
bar_width = x_eps[1] - x_eps[0]
plt.bar(x_eps, f_eps.flatten(), width=bar_width, fc=[0.7, 0.7, 0.7],
        edgecolor=[0.5, 0.5, 0.5])

plt.title(db_riskdrivers_series.columns[d_plot-1] + \
          ' projected risk driver distribution',
         fontweight='bold', fontsize=20)
plt.xlabel('Projected risk driver', fontsize=17)
add_logo(proj_dist, location=1, set_fig_size=False)
proj_dist.tight_layout()
