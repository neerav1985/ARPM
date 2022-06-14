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

# # s_fund_theorem_mre [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_fund_theorem_mre&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-sdf-mre).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arpym.statistics.simulate_normal import simulate_normal
from arpym.statistics.cdf_sp import cdf_sp
from arpym.statistics.pdf_sp import pdf_sp
from arpym.pricing.numeraire_mre import numeraire_mre
from arpym.tools.logo import add_logo
# -

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_fund_theorem_mre-implementation-step00): Upload data

# +
path = '~/databases/temporary-databases/'

db_vpay = pd.read_csv(path+'db_valuation_vpay.csv', index_col=0)
v_pay = db_vpay.values
db_prob = pd.read_csv(path+'db_valuation_prob.csv', index_col=0)
p = db_prob.values.T[0]
db_v = pd.read_csv(path+'db_valuation_v.csv', index_col=0)
v = db_v.values.T[0]
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_fund_theorem_mre-implementation-step01): Minimum relative entropy numeraire probabilities

p_mre, sdf_mre = numeraire_mre(v_pay, v, p=p, k=1)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_fund_theorem_mre-implementation-step02): Fundamental theorem of asset pricing

rhs = v / v[1]
lhs = p_mre * (v_pay[:, 1]**(-1))@v_pay

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_fund_theorem_mre-implementation-step03): Radon-Nikodym derivative

d_mre = p_mre / p

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_fund_theorem_mre-implementation-step04): Pdfs

h = 0.02
# grid for computing pdfs
x = np.linspace(-1, 4, 100)
# compute pdfs
f_sdf_mre = pdf_sp(h, np.array([x]).T, np.array([sdf_mre]).T, p)
f_d_mre = pdf_sp(h, np.array([x]).T, np.array([d_mre]).T, p)
f_infl = pdf_sp(h, np.array([x]).T, np.array([v_pay[:, 1]/v[1]]).T, p)

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_fund_theorem_mre-implementation-step04): Cdf of numeraire under probability measures p and p_mre

y = np.linspace(0, 12, 100)
ind = np.argsort(v_pay[:, 1])
cdf = cdf_sp(y, v_pay[:, 1], p)
cdf_mre = cdf_sp(y, v_pay[:, 1], p_mre)

# ## Plots

# +
# Fund. theorem of asset pricing empirical verification
plt.style.use('arpm')

fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)
plt.plot([np.min(rhs), np.max(lhs)], [np.min(rhs), np.max(lhs)], 'r')
plt.scatter(rhs, lhs, marker='o')
plt.axis([np.min(rhs), np.max(rhs), np.min(rhs), np.max(rhs)])
plt.xlabel('r. h. side', size=17)
plt.ylabel('l. h. side', size=17)
plt.legend(['identity line'])
plt.title('Fund. theorem of asset pricing')
add_logo(fig, location=4, alpha=0.8, set_fig_size=False)

# Pdfs of mre SDF and Radon-Nykodym derivative
f_sdf_name = r'$\mathit{Sdf}^{\mathit{MRE}}$'
f_d_name = r'$D^{\mathit{MRE}}$'
f_infl_name = r'$[\mathcal{V}^{\mathit{pay}}]_{2, \cdot}/v_{2}$'

fig, axes = plt.subplots(1, 2, figsize=(1280.0/72.0, 720.0/72.0))
axes[0].plot(x, f_sdf_mre, 'b', label=f_sdf_name)
axes[0].plot(x, f_d_mre, 'g', label=f_d_name)
axes[0].plot(x, f_infl, 'r', label=f_infl_name)
yl = axes[0].get_ylim()
axes[0].plot([v[0], v[0]], [0, yl[1]], 'b--',
             label=r'$E\{$' + f_sdf_name + '$\}$')
axes[0].plot([1, 1], [0, yl[1]], 'g--',
             label=r'$E\{$' + f_d_name + '$\}$')
axes[0].plot([p @ v_pay[:, 1] / v[1],
              p @ v_pay[:, 1] / v[1]], [0, yl[1]], 'r--',
             label=r'$E\{$' + f_infl_name + '$\}$')
axes[0].set_xlim([x[0], x[-1]])
axes[0].set_ylim(yl)
axes[0].legend()

axes[1].plot(y, cdf, 'b', label='$F$')
axes[1].plot(y, cdf_mre, 'g', label='$F^{MRE}$')
axes[1].set_ylim([0, 1])
axes[1].set_xlabel(r'$[\mathcal{V}^{\mathit{pay}}]_{2, \cdot}$')
axes[1].legend()

add_logo(fig, location=4, size_frac_x=1/8, set_fig_size=False)
plt.tight_layout()
# -


