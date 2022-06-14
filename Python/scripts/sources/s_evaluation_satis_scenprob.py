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

# # s_evaluation_satis_scenprob [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_evaluation_satis_scenprob&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBEvalNumericalExample).

# +
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

from arpym.portfolio.spectral_index import spectral_index
from arpym.statistics.meancov_sp import meancov_sp
from arpym.statistics.quantile_sp import quantile_sp
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-parameters)

alpha = 0.01  # threshold probability
lam_evar = 0.25  # parameter for α-expectile
alpha_prop_haz = 0.5  # parameter for proportional hazards expectation
lam_buhlmann = 2  # parameter for Esscher expectation
lam_mv = 0.5  # parameter for mean-variance and mean-semideviation trade-off
lam_ut = 2  # parameter for certainty-equivalent (exponential function)
r = 0.0001  # target for omega ratio
theta = -0.1  # parameter for Wang expectation
z = np.array([-0.0041252, -0.00980853,  -0.00406089,  0.02680999])  # risk factor

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step00): Import data

# +
path = '~/databases/temporary-databases/db_aggregation_scenario_numerical.csv'

j_ = pd.read_csv(path, usecols=['j_'], nrows=1).values[0, 0].astype(int)
n_ = pd.read_csv(path, usecols=['n_'], nrows=1).values[0, 0].astype(int)
# joint scenarios-probabilities
pi = pd.read_csv(path, usecols=['pi']).values.reshape(j_, n_)
p = pd.read_csv(path, usecols=['p']).iloc[:j_].values.reshape(j_, )
# holdings
h_tilde = pd.read_csv(path, usecols=['h_tilde']).iloc[:n_].values.reshape(n_, )
# budgets
v_h = pd.read_csv(path, usecols=['v_h'], nrows=1).values[0, 0].astype(int)
v_b = pd.read_csv(path, usecols=['v_b'], nrows=1).values[0, 0].astype(int)
# returns
r_h = pd.read_csv(path, usecols=['r_h']).iloc[:j_].values.reshape(j_, )
pi_b_resc = pd.read_csv(path, usecols=['pi_b_resc']).iloc[:j_].values.reshape(j_, )
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step01):  Performance expectation, variance, negative standard deviation

mu_r_h, s2_r_h = meancov_sp(r_h, p)  # performance expectation
s2_satis = - s2_r_h  # performance variance
std_satis = -np.sqrt(s2_r_h)  # negative standard deviation

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step02): Performance mean-variance trade-off

# performance mean-variance trade-off by definition
mv_r_h = mu_r_h-lam_mv/2*s2_r_h
mu_pi, s2_pi = meancov_sp(pi, p)  # instruments P&L's exp. and cov.
# performance mean-variance trade-off by quadratic form
mv_r_h_quad = h_tilde@mu_pi-lam_mv/2*h_tilde@s2_pi@h_tilde

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step03): Certainty-equivalent

mu_ut = -np.exp(-lam_ut*r_h)@p  # expected utility
ceq_r_h = -(1/lam_ut)*np.log(-mu_ut)  # certainty-equivalent

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step04): Quantile (VaR) satisfaction measure

q_r_h = quantile_sp(alpha, r_h, p=p,  method='kernel_smoothing')

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step05): Expected shortfall/sub-quantile satisfaction measure

# +
sort_r_h_j = np.argsort(r_h)  # sorted indices
r_h_sort = np.sort(r_h)  # sorted scenarios
p_sort = p[sort_r_h_j]  # sorted probabilities
u_sort = np.r_[0, np.cumsum(p_sort)]  # cumulative sum of ordered probs.

j_alpha  = [i for i, x in enumerate(u_sort) if u_sort[i-1]<alpha and
            u_sort[i]>=alpha][0]
# weights
weight_j = np.zeros(j_)
if j_alpha  == 1:
    weight_j[0] = 1
elif j_alpha  >1:
    weight_j[j_alpha-1] = 1 - u_sort[j_alpha-1]/alpha
    for j in range(j_alpha-1):
        weight_j[j] = p_sort[j]/alpha
# negative expected shortfall/sub-quantile
q_sub_r_h = r_h_sort@weight_j
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step06): Wang expectation

wang_expectation_r_h = r_h_sort@np.append(norm.cdf(norm.ppf(u_sort[1:])-theta)[0],
                                          np.diff(norm.cdf(norm.ppf(u_sort[1:])
                                                           -theta)))

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step07): Proportional hazard expectation

prop_haz_expectation_r_h = r_h_sort@ np.diff(u_sort**alpha_prop_haz)

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step08): Mean-semideviation trade-off, lower partial moments

semiv_r_h = sum(((r_h[r_h <= mu_r_h] - mu_r_h) ** 2)
                * p[r_h <= mu_r_h])  # semivariance
semid_r_h = (semiv_r_h) ** (0.5)  # semideviation
# mean-semideviation trade-off
msemid_r_h = mu_r_h - lam_mv * semid_r_h
# first order lower partial moment
lpm_1_r_h = np.maximum(r - r_h, 0)@p
# second oerder lower partial moment
lpm_2_r_h = (np.maximum(r - r_h, 0) ** 2)@p

# ## [Step 9](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step09): Expectile

# +
def expectile_f(x, p, lam):
    return lam * np.sum(p * np.maximum(r_h - x, 0)) + \
        (1 - lam) * (np.sum(p * np.minimum(r_h - x, 0)))


# expectile
expectile_r_h = fsolve(expectile_f, -0.01, args=(p, lam_evar))
# -

# ## [Step 10](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step10): Information ratio, Sortino ratio and omega ratio

# information ratio
info_ratio_r_h = mu_r_h /np.sqrt(s2_r_h) 
# Sortino ratio
sortino_ratio_r_h = (mu_r_h - r) / np.sqrt(lpm_2_r_h)
# omega ratio by definition
omega_ratio_r_h = (np.maximum(r_h - r, 0)@p) / lpm_1_r_h
# omega ratio by equivalent formulation
omega_ratio_1_r_h = (r_h@p - r) / lpm_1_r_h + 1

# ## [Step 11](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step11): Scenario-probability distribution of factor Z, beta, correlation

mu_z, s2_z = meancov_sp(z, p)  # variance of z
cv_yz = (r_h * z)@p - mu_r_h * mu_z  # covariance of r_h and z
beta_r_h_z = - cv_yz / s2_z  # opposite of beta
# correlation satisfaction measure
cr_r_h_z = - cv_yz / (np.sqrt(s2_r_h) * np.sqrt(s2_z))

# ## [Step 12](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step12): Buhlmann expectation and Esscher expectation

# +
bulhmann_expectation_r_h, _ = meancov_sp(np.exp(-lam_buhlmann*pi_b_resc)*r_h, p)[0] / \
                              meancov_sp(np.exp(-lam_buhlmann*pi_b_resc), p)

esscher_expectation_r_h, _ = meancov_sp(np.exp(-lam_buhlmann*r_h)*r_h, p)[0] / \
                             meancov_sp(np.exp(-lam_buhlmann*r_h), p)
# -

# ## [Step 13](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_scenprob-implementation-step13): Save the data

# +
output = {'s2_satis': pd.Series(s2_satis),
          'std_satis': pd.Series(std_satis),
          'wang_expectation_r_h': pd.Series(wang_expectation_r_h),
          'prop_haz_expectation_r_h': pd.Series(prop_haz_expectation_r_h),
          'expectile_r_h': pd.Series(expectile_r_h),
          'bulhmann_expectation_r_h': pd.Series(bulhmann_expectation_r_h),
          'esscher_expectation_r_h': pd.Series(esscher_expectation_r_h)
          }

df = pd.DataFrame(output)
df.to_csv('~/databases/temporary-databases/db_evaluation_scenprob.csv')
