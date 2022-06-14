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

# # s_checklist_montecarlo_step04 [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_checklist_montecarlo_step04&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ex-vue-4-copmarg).

# +
import numpy as np
import pandas as pd
from scipy.stats import t as tstu
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from arpym.statistics.meancov_sp import meancov_sp
from arpym.statistics.quantile_sp import quantile_sp
from arpym.statistics.simulate_markov_chain_multiv import simulate_markov_chain_multiv
from arpym.statistics.simulate_t import simulate_t
from arpym.statistics.project_trans_matrix import project_trans_matrix
from arpym.tools.histogram_sp import histogram_sp
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step04-parameters)

# t_now is 31-Aug-2012. Set t_hor>t_now
t_hor = np.datetime64('2012-10-26')  # the future investment horizon
j_ = 5000  # number of scenarios
d_plot = 1 # projected risk driver to plot

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step04-implementation-step00): Load data

# +
path = '~/databases/temporary-databases/'

# Risk drivers identification
# realizations of risk drivers up to and including time t_now
db_riskdrivers_series = pd.read_csv(path+'db_riskdrivers_series.csv',
                                    index_col=0, parse_dates=True)
x = db_riskdrivers_series.values
risk_drivers_names = db_riskdrivers_series.columns

# additional information
db_riskdrivers_tools = pd.read_csv(path+'db_riskdrivers_tools.csv')
d_ = int(db_riskdrivers_tools.d_.dropna())
n_stocks = int(db_riskdrivers_tools.n_stocks.dropna())
d_implvol = int(db_riskdrivers_tools.d_implvol.dropna())
n_bonds = int(db_riskdrivers_tools.n_bonds.dropna())
i_bonds = n_bonds*4  # 4 NS parameters x n_bonds
t_now = np.datetime64(db_riskdrivers_tools.t_now[0], 'D')
d_credit = int(db_riskdrivers_tools.d_credit.dropna())
c_ = int(db_riskdrivers_tools.c_.dropna())
ratings_tnow = np.array(db_riskdrivers_tools.ratings_tnow.dropna())

# Quest for invariance
# values of invariants
db_invariants_series = pd.read_csv(path+'db_invariants_series.csv',
                                   index_col=0, parse_dates=True)
epsi = db_invariants_series.values
t_, i_ = np.shape(epsi)

# next step models
db_invariants_nextstep = pd.read_csv(path+'db_invariants_nextstep.csv')

# parameters for GARCH(1,1) next-step models
db_invariants_garch_param = pd.read_csv(path+'db_invariants_garch_param.csv',
                                        index_col=0)

# parameters for AR(1) next-step models
db_invariants_ar1_param = pd.read_csv(path+'db_invariants_ar1_param.csv',
                                        index_col=0)

# estimated annual credit transition matrix 
p_credit = pd.read_csv(path +
                       'db_invariants_p_credit.csv').values.reshape(c_+1, c_+1)

# Estimation
# parameters for invariants modeled using Student t distribution 
db_estimation_parametric = pd.read_csv(path+'db_estimation_parametric.csv',
                                       index_col=0)

# estimated probabilities for nonparametric distributions
db_estimation_nonparametric = pd.read_csv(path+'db_estimation_nonparametric.csv',
                                          index_col=False)
p_marginal = db_estimation_nonparametric.values

# parameters for estimated Student t copula
db_estimation_copula = pd.read_csv(path+'db_estimation_copula.csv')
nu_copula = int(db_estimation_copula['nu'].iloc[0])
rho2_copula = np.array(db_estimation_copula['rho2']).reshape(i_, i_)

# parameters for the credit copula
db_estimation_credit_copula = pd.read_csv(path+'db_estimation_credit_copula.csv')
rho2_credit = db_estimation_credit_copula.rho2_credit.values.reshape(2, 2)
nu_credit = db_estimation_credit_copula.nu_credit[0]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step04-implementation-step01): Determine number of projection steps and scenario probabilities

# number of monitoring times
m_ = np.busday_count(t_now, t_hor)
# projection scenario probabilities
p = np.ones(j_)/j_
# invariants modeled parametrically
ind_parametric = np.arange(n_stocks+1+d_implvol,
                         n_stocks+1+d_implvol+i_bonds)
# invariants modeled nonparametrically
ind_nonparametric = list(set(range(i_))-set(ind_parametric))

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step04-implementation-step02): Projection of invariants

# +
epsi_proj = np.zeros((j_, m_, i_))

for m in range(m_):
    # copula scenarios
    # simulate standardized invariants scenarios for copula
    epsi_tilde_proj = simulate_t(np.zeros(i_), rho2_copula, nu_copula, j_)
    
    # generate invariants scenarios
    # invariants modeled nonparametrically
    for i in ind_nonparametric:
        # project t-copula standardized invariants scenarios
        u_proj = tstu.cdf(epsi_tilde_proj[:, i], nu_copula)
        epsi_proj[:, m, i] = quantile_sp(u_proj, epsi[:, i], p_marginal[:, i])
    # invariants modeled parametrically (estimated as Student t distributed)
    for i in ind_parametric:
        # project t-copula standardized invariants scenarios
        u_proj = tstu.cdf(epsi_tilde_proj[:, i], nu_copula)
        mu_marg = db_estimation_parametric.loc['mu', str(i)]
        sig2_marg = db_estimation_parametric.loc['sig2', str(i)]
        nu_marg = db_estimation_parametric.loc['nu', str(i)]
        epsi_proj[:, m, i] = mu_marg + np.sqrt(sig2_marg)*tstu.ppf(u_proj, nu_marg)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step04-implementation-step03): Projection of risk drivers

# +
x_proj = np.zeros((j_, m_+1, d_))
dx_proj = np.zeros((j_, m_+1 , d_))
sig2_garch = np.zeros((j_, m_+1, d_))

a_garch = db_invariants_garch_param.loc['a'].values
b_garch = db_invariants_garch_param.loc['b'].values
c_garch = db_invariants_garch_param.loc['c'].values
mu_garch = db_invariants_garch_param.loc['mu'].values

# risk drivers at time t_now are the starting values for all scenarios
x_proj[:, 0, :] = db_riskdrivers_series.iloc[-1, :]

# initialize parameters for GARCH(1,1) projection
d_garch = [d for d in range(d_)
           if db_invariants_nextstep.iloc[0, d] =='GARCH(1,1)']  
for d in d_garch:
    sig2_garch[:, 0, d] = db_invariants_garch_param.iloc[-1, d]
    dx_proj[:, 0, d] = x[-1, d] - x[-2, d]

# project daily scenarios
for m in range(1, m_+1):
    for d in range(d_):
        risk_driver = risk_drivers_names[d]
        # risk drivers modeled as random walk
        if db_invariants_nextstep.loc[0, risk_driver] == 'Random walk':
            x_proj[:, m, d] = x_proj[:, m-1, d] + epsi_proj[:, m-1, d]

        # risk drivers modeled as GARCH(1,1)
        elif db_invariants_nextstep.loc[0, risk_driver] == 'GARCH(1,1)':
            sig2_garch[:, m, d] = c_garch[d] + \
                b_garch[d]*sig2_garch[:, m-1, d] +\
                a_garch[d]*(dx_proj[:, m-1, d]-mu_garch[d])**2
            dx_proj[:, m, d] = mu_garch[d] +\
                np.sqrt(sig2_garch[:, m, d])*epsi_proj[:, m-1, d]
            x_proj[:, m, d] = x_proj[:, m-1, d] + dx_proj[:, m, d]

        # risk drivers modeled as AR(1)
        elif db_invariants_nextstep.loc[0, risk_driver] == 'AR(1)':
            b_ar1 = db_invariants_ar1_param.loc['b', risk_driver]
            x_proj[:, m, d] = b_ar1*x_proj[:, m-1, d] + epsi_proj[:, m-1, d]
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step04-implementation-step04): Projection of credit ratings

# +
# compute the daily credit transition matrix
p_credit_daily = project_trans_matrix(p_credit, 1/252, credit=True)

# project ratings
ratings_proj = simulate_markov_chain_multiv(ratings_tnow, p_credit_daily,
                                            m_, rho2=rho2_credit,
                                            nu=nu_credit, j_=j_)
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_checklist_montecarlo_step04-implementation-step05): Save databases

# +
# delete big files
del dx_proj, sig2_garch

# projected risk drivers
out = pd.DataFrame({risk_drivers_names[d]:
                   x_proj[:, :, d].reshape((j_*(m_+1),))
                   for d in range(d_)})
out = out[list(risk_drivers_names[:d_].values)]
out.to_csv(path+'db_projection_riskdrivers.csv', index=None)
del out

# projected credit ratings
out = pd.DataFrame({'GE': ratings_proj[:, :, 0].reshape((j_*(m_+1),)),
                   'JPM': ratings_proj[:, :, 1].reshape((j_*(m_+1),))})
out.to_csv(path+'db_projection_ratings.csv', index=None)
del out

# number of scenarios and future investment horizon
out = pd.DataFrame({'j_': pd.Series(j_),
                    't_hor': pd.Series(t_hor)})
out.to_csv(path+'db_projection_tools.csv', index=None)
del out

# projected scenario probabilities
out = pd.DataFrame({'p': pd.Series(p)})
out.to_csv(path+'db_scenario_probs.csv', index=None)
del out
# -

# ## Plots

# +
plt.style.use('arpm')

# colors
arpm_orange = '#FF9900'
arpm_green = '#3C9591'
arpm_blue = '#0D5E94'

# number of paths to plot
num_plot = min(j_, 20)

# market risk driver path
fig1 = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

# mean and standard deviation at each projection time step
mu_thor = np.zeros(m_ + 1)
sig_thor = np.zeros(m_ + 1)
for m in range(0, m_ + 1):
    mu_thor[m], sig2_thor = meancov_sp(x_proj[:, m, d_plot-1].reshape(-1, 1))
    sig_thor[m] = np.sqrt(sig2_thor)

# plot historical series
f1 = plt.plot(np.arange(t_-2, t_+1), db_riskdrivers_series.iloc[-3:, d_plot-1],
              lw=1, color=[0.7, 0.7, 0.7], zorder=10)
f1 = plt.scatter(np.arange(t_-2, t_+1), db_riskdrivers_series.iloc[-3:, d_plot-1],
                color=arpm_blue, zorder=40)
# plot projected series
for j in range(num_plot):
    f1 = plt.plot(np.arange(t_, t_+m_+1), x_proj[j, :, d_plot-1],
                  lw=1, color=[0.7, 0.7, 0.7], zorder=0)
# plot mean and std deviation
p_mu = plt.plot(np.arange(t_, t_+m_+1), mu_thor, color=arpm_green,
                label='expectation', lw=2, zorder=30)
p_red_1 = plt.plot(np.arange(t_, t_+m_+1), mu_thor + 2 * sig_thor,
                   label='+ / - 2 st.deviation', color=arpm_orange, lw=2,
                   zorder=20)
p_red_2 = plt.plot(np.arange(t_, t_+m_+1), mu_thor - 2 * sig_thor,
                   color=arpm_orange, lw=2, zorder=20)

f, xp = histogram_sp(x_proj[:, -1, d_plot-1], k_=10*np.log(j_))
f1 = plt.barh(xp, f, height=xp[1]-xp[0], left=t_+m_,
              fc=[0.7, 0.7, 0.7], edgecolor=[0.5, 0.5, 0.5],
              label='horizon pdf')
plt.title('Projected path: ' + risk_drivers_names[d_plot-1],
          fontweight='bold', fontsize=20)
plt.xlabel('t (days)', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=17)
add_logo(fig1, set_fig_size=False)
fig1.tight_layout()

# plot projected ratings
# select paths with rating changes
ind_j_plot_GE = np.zeros(1)
ind_j_plot_GE[0] = 0
k = 0
while k < num_plot:
    k = k+1
    for j in range(j_):
        if (j not in ind_j_plot_GE and
                ratings_proj[j, -1, 0] != ratings_proj[k, -1, 0]):
            ind_j_plot_GE = np.append(ind_j_plot_GE, j)
            break

ind_j_plot_JPM = np.zeros(1)
ind_j_plot_JPM[0] = 0
k = 0
while k < num_plot:
    k = k+1
    for j in range(j_):
        if (j not in ind_j_plot_JPM and
                ratings_proj[j, -1, 1] != ratings_proj[k, -1, 1]):
            ind_j_plot_JPM = np.append(ind_j_plot_JPM, j)
            break

fig2, ax = plt.subplots(2, 1, figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
plt.sca(ax[0])
for j in ind_j_plot_GE:
    f5 = plt.plot(np.arange(m_+1), ratings_proj[int(j), :, 0]+1)
    plt.title('Projected rating GE', fontweight='bold', fontsize=20)
plt.yticks(np.arange(10), fontsize=14)
ax[0].set_yticklabels(['', 'AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D', ''])
plt.gca().invert_yaxis()
plt.sca(ax[1])

for j in ind_j_plot_JPM:
    plt.plot(np.arange(m_+1), ratings_proj[int(j), :, 1]+1)
    plt.title('Projected rating JPM', fontweight='bold', fontsize=20)
plt.yticks(np.arange(10), fontsize=14)
ax[1].set_yticklabels(['', 'AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D', ''])
plt.gca().invert_yaxis()
add_logo(fig2, set_fig_size=False)
fig2.tight_layout()
