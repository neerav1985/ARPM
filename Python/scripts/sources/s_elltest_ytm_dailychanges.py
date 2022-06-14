#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_elltest_ytm_dailychanges [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_elltest_ytm_dailychanges&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=ExerdYinv).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.statistics.invariance_test_ellipsoid import invariance_test_ellipsoid
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_ytm_dailychanges-parameters)

t_ = 1000  # length of time series of yields
tau = 10  # selected time to maturity (years)
l_ = 10  # number of lags
conf_lev = 0.95  # confidence level

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_ytm_dailychanges-implementation-step00): Load data

tau = np.array([tau])
path = '~/databases/global-databases/fixed-income/db_yields'
y = pd.read_csv(path + '/data.csv', header=0, index_col=0)
y = y[tau.astype(float).astype(str)].tail(t_).values.reshape(-1)

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_ytm_dailychanges-implementation-step01): Compute the daily yield increment

epsi = np.diff(y)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_elltest_ytm_dailychanges-implementation-step02): Ellipsoid test

# +
plt.style.use('arpm')

# perform and show ellipsoid test for invariance
rho, conf_int = \
    invariance_test_ellipsoid(epsi, l_, conf_lev=conf_lev, fit=0, r=2,
                              title='Invariance test on daily yield changes')
fig = plt.gcf()
add_logo(fig, set_fig_size=False, size_frac_x=1/8)
