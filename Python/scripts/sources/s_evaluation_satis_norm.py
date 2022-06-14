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

# # s_evaluation_satis_norm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_evaluation_satis_norm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=EBSatisfNormalNumerical).

import numpy as np
import pandas as pd
from scipy.special import erfinv

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_norm-parameters)

lam = 1/4  # risk aversion parameter
alpha = 0.05  # threshold probability

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_norm-implementation-step00): Import data

# +
path = '~/databases/temporary-databases/'
db = pd.read_csv(path + 'db_aggregation_normal.csv', index_col=0)

n_ = int(np.array(db['n_'].iloc[0]))
# parameters of portfolio P&L distribution
mu_pi_h = np.array(db['mu_h'].iloc[0])
sig2_pi_h = np.array(db['sig2_h'].iloc[0])
# holdings
h = np.array(db['h'].iloc[:n_]).reshape(-1)
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_norm-implementation-step01): Standard deviation satisfaction measure

sig_pi_h = np.sqrt(sig2_pi_h)
sd_satis = -sig_pi_h

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_norm-implementation-step02): Performance mean-variance trade-off

mv_pi_h = mu_pi_h-lam/2*sig2_pi_h

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_norm-implementation-step03): Certainty-equivalent

ceq_pi_h = mu_pi_h-lam/2*sig2_pi_h

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_norm-implementation-step04): Quantile (VaR) satisfaction measure

# quantile (VaR) measure
q_pi_h = mu_pi_h+sig_pi_h*np.sqrt(2)*erfinv(2*alpha-1)
# Cornish-Fisher approximation
q_pi_h_cf = (mu_pi_h+sig_pi_h*(-1.64))

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_norm-implementation-step05): Expected shortfall/sub-quantile satisfaction measurel 

q_sub_pi_h = mu_pi_h+sig_pi_h/alpha * \
            (-1/(np.sqrt(2*np.pi))*np.exp(-erfinv(2*alpha-1)**2))

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_norm-implementation-step06): Information ratio

info_ratio_pi_h = mu_pi_h/sig_pi_h

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_evaluation_satis_norm-implementation-step07): Save data

# +
output = {
          '-sig_pi_h': pd.Series(sd_satis),
          'cvar_pi_h': pd.Series(q_pi_h),
         }

df = pd.DataFrame(output)
df.to_csv('~/databases/temporary-databases/db_evaluation_satis_normal.csv')
