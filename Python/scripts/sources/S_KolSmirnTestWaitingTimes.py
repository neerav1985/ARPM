#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script performs the Kolmogorov-Smirnov test for invariance on the
# time intervals between subsequent events in high frequency trading.
# -

# ## For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=exer-expiid-copy-1).

# +
# ## Prepare the environment

# +
import os
import os.path as path
import sys

sys.path.append(path.abspath('../../functions-legacy'))


from numpy import where, diff, array

from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from ARPM_utils import save_plot, date_mtop, struct_to_dict, time_mtop
from TestKolSmirn import TestKolSmirn
from InvarianceTestKolSmirn import InvarianceTestKolSmirn
from TradeQuoteProcessing import TradeQuoteProcessing
# -

# ## Upload the database

# +
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'db_US_10yr_Future_quotes_and_trades'),squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'db_US_10yr_Future_quotes_and_trades'),squeeze_me=True)

quotes = struct_to_dict(db['quotes'], as_namedtuple=False)
trades = struct_to_dict(db['trades'], as_namedtuple=False)
# -

# ## Process the time series, refining the raw data coming from the database

# +
dates_quotes = quotes['time_names']  #
t = quotes['time']  # time vector of quotes
p_bid = quotes['bid']  # bid prices
p_ask = quotes['ask']  # ask prices
q_bid = quotes['bsiz']  # bid volumes
q_ask = quotes['asiz']  # ask volumes

dates_trades = trades['time_names']  #
t_k = trades['time']  # time vector of trades
p_last = trades['price']  # last transaction prices
delta_q = trades['siz']  # flow of traded contracts' volumes
delta_sgn = trades['aggress']  # trade sign flow
match = trades[
    'mtch']  # match events: - the "1" value indicates the "start of a match event" while zeros indicates the "continuation of a match event"
#              - the db is ordered such that the start of a match event is in the last column corresponding to that event

t, _, _, _, _, _, t_k, *_ = TradeQuoteProcessing(t, dates_quotes, q_ask, p_ask, q_bid, p_bid, t_k, dates_trades,
                                                         p_last, delta_q, delta_sgn, match)
t = t.flatten()
t_k = t_k.flatten()
# ## Compute the gaps between subsequent events

k_0 = where(t_k >= t[0])[0][0]  # index of the first trade within the time window
k_1 = where(t_k <= t[len(t)-1])[0][-1]  # index of the last trade within the time window
t_ms = array([time_mtop(i) for i in t_k[k_0:k_1+1]])
t_k = array([3600*i.hour+60*i.minute+i.second+i.microsecond/1e6 for i in t_ms])
delta_t_k = diff(t_k).reshape(1,-1) # gaps
# -

# ## Perform the Kolmogorov-Smirnov test

s_1, s_2, int, F_1, F_2, up, low = TestKolSmirn(delta_t_k)

# ## Plot the results of the IID test

# +
# position settings
pos = {}
pos[1] = [0.1300, 0.74, 0.3347, 0.1717]
pos[2] = [0.5703, 0.74, 0.3347, 0.1717]
pos[3] = [0.1300, 0.11, 0.7750, 0.5]
pos[4] = [0.03, 1.71]

# create figure
f = figure()
InvarianceTestKolSmirn(delta_t_k, s_1, s_2, int, F_1, F_2, up, low, pos, 'Kolmogorov-Smirnov invariance test',
                       [-0.3, 0]);

# save_plot(ax=plt.gca(), extension='png', scriptname=os.path.basename('.')[:-3], count=plt.get_fignums()[-1])
