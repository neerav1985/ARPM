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

# # s_fit_discrete_markov_chain [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_fit_discrete_markov_chain&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-fit-discrete-markov-chain).

# ## Prepare the environment

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as tstu

from arpym.estimation.fit_trans_matrix_credit import fit_trans_matrix_credit
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_fit_discrete_markov_chain-parameters)

tau_hl = 5    # half-life parameter in years
r = 3    # initial rating

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_fit_discrete_markov_chain-implementation-step00): Upload data

path = '~/databases/temporary-databases/'    # upload data
db_credit = pd.read_csv(path+'db_credit_rd.csv',
                        index_col=0, parse_dates=True)
filt=['(' not in col for col in db_credit.columns]
ratings = [i for indx,i in enumerate(db_credit.columns) if filt[indx] == True]
c_ = len(ratings)-1
n_obligors = db_credit.values[:, :c_+1]
dates = np.array(db_credit.index).astype('datetime64[D]')
t_ = dates.shape[0]
n_cum_trans = db_credit.values[:, c_+1:].reshape(t_, c_+1, c_+1)
stocks_path = '~/databases/global-databases/equities/db_stocks_SP500/'
db_stocks = pd.read_csv(stocks_path + 'db_stocks_sp.csv', skiprows=[0],
                        index_col=0)
v = db_stocks.loc[:, ['GE', 'JPM']].values

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_fit_discrete_markov_chain-implementation-step01): Compute final transition matrix

p = fit_trans_matrix_credit(dates, n_obligors, n_cum_trans, tau_hl)    # transition probability matrix

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_fit_discrete_markov_chain-implementation-step02): Compute cdf

f = np.cumsum(p[3, :])    # conditional cdf

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_fit_discrete_markov_chain-implementation-step03): Save database

out = pd.DataFrame(p)
out.to_csv(path+'db_trans_matrix.csv')
del out

# ## Plots

# +
plt.style.use('arpm')
mydpi = 72.0
fig = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)
bars = ('AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D')

ax1 = plt.axes([0.6, 0.53, 0.35, 0.35])
ax1.step(np.arange(c_+2), np.r_[0, f], color='black')
ax1.tick_params(axis='x', colors='None')
ax1.set_xlim([-0.5, 8])
plt.ylabel('cdf', fontsize=17)
plt.xlabel(r'$\tilde{c}$', horizontalalignment='right', x=1)
plt.title("Markov chain for credit ratings", fontsize=20, fontweight='bold')

ax2 = plt.axes([0.6, 0.4, 0.35, 0.1])
height = p[:, 3]
AAA = [0/255, 166/255, 0/255]
AA = [75/255, 209/255, 29/255]
A = [131/255, 213/255, 32/255]
BBB = [188/255, 217/255, 34/255]
BB = [221/255, 195/255, 36/255]
B = [225/255, 144/255, 38/255]
CCC = [229/255, 92/255, 40/255]
D =  [233/255, 42/255, 47/255]
plt.bar(bars, height, color=[AAA, AA, A, BBB, BB, B, CCC, D])
ax2.set_xlim([-0.5, 8])
ax2.set_ylim([0, 1])
plt.ylabel('pdf', fontsize=17)
plt.xlabel(r'$\tilde{c}$', horizontalalignment='right', x=1)

plt.show()
add_logo(fig)
# -
