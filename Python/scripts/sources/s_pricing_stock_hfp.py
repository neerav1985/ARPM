#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_pricing_stock_hfp [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_pricing_stock_hfp&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-stock-pl-unitary-hor).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from arpym.tools.histogram_sp import histogram_sp
from arpym.tools.logo import add_logo
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_stock_hfp-implementation-step00): Upload data

# +
data = pd.read_csv('~/databases/temporary-databases/stocks_proj_hfp.csv')

x_t_hor = data['x_t_hor'] # scenario-probability distribution of log-value
p = data['p'] 
x = data['x'] 
x_t_now = x[0] # log-value at the current time
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_stock_hfp-implementation-step01): Compute scenario-probability distribution of the equity P&L

v_t_now = np.exp(x_t_now) # current value of AMZN
pl_t_hor = v_t_now*(np.exp(x_t_hor-x_t_now) - 1) # distribution of the P&L

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_pricing_stock_hfp-implementation-step02): Histogram of the AMZN's equity P&L distribution

h, b = histogram_sp(pl_t_hor, p=p, k_=10 * np.log(np.shape(x_t_hor)))

# ## Plots

# +
# settings
plt.style.use('arpm')
mydpi = 72.0
colhist = [.75, .75, .75]
coledges = [.3, .3, .3]
fig, ax = plt.subplots()
ax.set_facecolor('white')
plt.bar(b, h, width=b[1]-b[0], facecolor=colhist, edgecolor=coledges)
plt.xlabel('P&L')
plt.xticks()
plt.yticks()

add_logo(fig, location=1)
plt.tight_layout()
