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

# # s_generalized_flam_toy [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_generalized_flam_toy&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy).

# +
import numpy as np

from arpym.tools.max_info_ratio_2 import max_info_ratio_2
from arpym.tools.transpose_square_root import transpose_square_root
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-parameters)

mu_pi = np.array([0, 0])  # mean of P&L's
mu_s = np.array([0, 0])  # mean of signals
sig_pi1_pi2 = 0.45  # correlation between P&L'ss
sig_s1_s2 = 0.3  # correlation between signals
sig_p1_s1 = 0.8  # correlation between first P&L and first signal
sig_p1_s2 = 0.6  # correlation between first P&L and second signal
sig_p2_s1 = 0.1  # correlation between second P&L and first signal
sig_p2_s2 = 0.2  # correlation between second P&L and second signal
r = 0  # risk-free rate
v = np.array([1, 2])  # current portfolio values
h = np.array([1, -3])  # current holdings
s = np.array([0.3, 0.1])  # current observed signal
sig = 1  # free variance parameter

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-implementation-step01): Information coefficent

# +
# covariance matrix of P&L's and signals
sig2 = np.array([[1, sig_pi1_pi2, sig_p1_s1, sig_p1_s2],
                 [sig_pi1_pi2, 1, sig_p2_s1, sig_p2_s2],
                 [sig_p1_s1, sig_p2_s1, 1, sig_s1_s2],
                 [sig_p1_s2, sig_p2_s2, sig_s1_s2, 1]])
n_ =  len(mu_pi)
k_ = len(mu_s)

# Riccati root of P&L's covariance
sig_pi = transpose_square_root(sig2[:n_, :n_], method='Riccati')
# Riccati root of signal covariance
sig_s = transpose_square_root(sig2[n_:, n_:], method='Riccati')

# linkage matrix
p_pi_s = np.linalg.inv(sig_pi) @ sig2[:n_, n_:]@  np.linalg.inv(sig_s)

# information coefficient of joint signals 
ic2 = np.trace(p_pi_s @ p_pi_s.T)

# information coefficient of single signals 
p_pi_s1 = np.linalg.inv(sig_pi) @ np.atleast_2d(sig2[:n_, 2]).T 
ic2_1 = np.trace(p_pi_s1 @ p_pi_s1.T)

p_pi_s2 = np.linalg.inv(sig_pi) @ np.atleast_2d(sig2[:n_, 3]).T
ic2_2 = np.trace(p_pi_s2 @ p_pi_s2.T)

# verify no relationships between information coefficients
err_ic2 = ic2_1+ic2_2 - ic2
# visualize outputs
print('linkage matrix p_pi_s =', p_pi_s)
print('info coeff. of joint signals ic2 =', ic2)
print('info coeff. of signal 1 ic2_1 =', ic2_1)
print('info coeff. of signal 2 ic2_2 =', ic2_2)
print('ic2_1 + ic2_2 - ic2 =', err_ic2)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-implementation-step02): Conditional information ratio

# +
# conditional moments
mu_pi_cond = mu_pi - r*v + sig2[:n_, n_:]@\
             np.linalg.solve(sig2[n_:, n_:],(s - mu_s).T)

sig2_pi_cond = sig2[:n_, :n_] - sig2[:n_, n_:]@\
               np.linalg.solve(sig2[n_:, n_:],sig2[n_:, :n_])

# conditional information ratio
ir_h_s = h.T@mu_pi_cond/np.sqrt(h.T@sig2_pi_cond @h)

# visualize output
print('conditional info. ratio ir_h_s =', ir_h_s)
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-implementation-step03): Maximum conditional information ratio and transfer coefficient

# +
# argmax conditional information ratio
h_sig = sig * np.linalg.solve(sig2_pi_cond, mu_pi_cond)/\
        np.sqrt(mu_pi_cond.T @ np.linalg.solve(sig2_pi_cond, mu_pi_cond))

# maximum conditional information ratio
max_ir_s = h_sig.T@mu_pi_cond/np.sqrt(h_sig.T@sig2_pi_cond @h_sig)

# transfer coefficient
tc = h.T@sig2_pi_cond @h_sig/np.sqrt(h.T@sig2_pi_cond @h)/\
     np.sqrt(h_sig.T@sig2_pi_cond @h_sig)

# verify flam and its relationship with transfer coeff.
max_ir_s_flam = np.sqrt(mu_pi_cond.T @ np.linalg.solve(sig2_pi_cond, mu_pi_cond))
tc_flam = ir_h_s/max_ir_s_flam

# visualize outputs
print('max cond info. ratio max_ir_s =', max_ir_s)
print('max cond info. ratio via FLAM max_ir_s_flam =', max_ir_s_flam)
print('transfer coefficient tc =', tc)
print('transfer coefficient via FLAM tc_flam =', tc_flam)
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-implementation-step04):  Maximum (l2-mean unconditional) information ratio

# +
# maximum information ratio of joint signals
max_ir2 = max_info_ratio_2(mu_pi-r*v, mu_s, sig2)

# maximum information ratios of single signals
max_ir2_1 = max_info_ratio_2(mu_pi-r*v, mu_s[0].reshape(-1),
                             sig2[[0,1,2], :][:, [0,1,2]])
max_ir2_2 = max_info_ratio_2(mu_pi-r*v, mu_s[1].reshape(-1),
                             sig2[[0,1,3], :][:, [0,1,3]])

# verify no relationships between max. info. ratios
err_max_ir_2 = max_ir2_1 + max_ir2_2 -  max_ir2

# visualize outputs
print('max uncond info.ratio of joint signals max_ir2 =', max_ir2)
print('max uncond info.ratio of signal 1 max_ir2_1 =', max_ir2_1)
print('max uncond info.ratio of signal 2 max_ir2_2 =', max_ir2_2)
print('max_ir2_1 + max_ir2_2 - max_ir2 =', err_max_ir_2)
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-implementation-step05):  Information coefficient relative to independent signal group

# +
# covariance matrix of independent signals
sig2_s_i = np.array([[sig2[2,2], 0],
                     [0, sig2[3,3]]])

sig_s_i = transpose_square_root(sig2_s_i, method='Riccati')

# information coefficient to independent signals
p_pi_s_i =  np.linalg.inv(sig_pi) @ sig2[:n_, n_:]@  np.linalg.inv(sig_s_i)
ic2_i = np.trace(p_pi_s_i @ p_pi_s_i.T)

err_ic2_i = ic2_1 + ic2_2 - ic2_i

# visualize output
# sum of marginal info. coeff. is joint info. coeff.
print('ic2_i =',ic2_i, 'and ic2_1 + ic2_2 - ic2_i =', err_ic2_i)
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-implementation-step06): Maximum information ratio relative to weak and independent signals

# +
# covariance matrix of P&L's and weak independent signals
sig2_wi = np.array([[1, sig_pi1_pi2, 0.05, 0.05],
                    [sig_pi1_pi2, 1, 0.05, 0.05],
                    [0.05, 0.05, 1, 0],
                    [0.05, 0.05, 0, 1]])

# conditional covariance
sig2_pi_cond_wi = sig2[:n_, :n_] - sig2_wi[:n_, n_:]@\
                  np.linalg.solve(sig2_wi[n_:, n_:],sig2_wi[n_:, :n_])

# maximum information ratio of joint weak independent signals
max_ir2_wi= max_info_ratio_2(mu_pi-r*v, mu_s, sig2_wi)

# maximum information ratios of single (weak indenpdent) signals 
max_ir2_wi_1 = max_info_ratio_2(mu_pi-r*v, mu_s[0].reshape(-1),
                                sig2_wi[[0,1,2], :][:, [0,1,2]])
max_ir2_wi_2 = max_info_ratio_2(mu_pi-r*v, mu_s[1].reshape(-1),
                                sig2_wi[[0,1,3], :][:, [0,1,3]])

# check flam
err_max_ir2_wi = max_ir2_wi_2 + max_ir2_wi_2 -  max_ir2_wi

# visualize outputs
print('max uncond info.ratio of joint weak ind. signals max_ir2_wi =', max_ir2_wi)
print('max uncond info.ratio of (weak ind.) signal 1 max_ir2_wi_1 =', max_ir2_wi_1)
print('max uncond info.ratio of (weak ind.) signal 2 max_ir2_wi_2 =', max_ir2_wi_2 )
print('max_ir2_wi_1+ max_ir2_wi_2 - max_ir2_wi =', max_ir2_wi_1+max_ir2_wi_2- max_ir2_wi)
# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_generalized_flam_toy-implementation-step07): Information coefficient relative to weak and independent signals

# +
# information coefficients
sig_s_wi = transpose_square_root(sig2_wi[n_:, n_:], method='Riccati')
p_pi_s_wi = np.linalg.inv(sig_pi) @ sig2_wi[:n_, n_:]@  np.linalg.inv(sig_s_wi)
ic2_wi = np.trace(p_pi_s_wi @ p_pi_s_wi.T)

p_pi_s1_wi = np.linalg.inv(sig_pi) @ np.atleast_2d(sig2_wi[:n_, 2]).T 
ic2_1_wi = np.trace(p_pi_s1_wi @ p_pi_s1_wi.T)

p_pi_s2_wi = np.linalg.inv(sig_pi) @ np.atleast_2d(sig2_wi[:n_, 3]).T
ic2_2_wi = np.trace(p_pi_s2_wi @ p_pi_s2_wi.T)

# check flam
err_max_ir2_ic2_wi =  max_ir2_wi - ic2_wi

# visualize outputs
print('linkage matrix of weak ind. signals p_pi_s_wi =', p_pi_s_wi)
print('ic2_wi =', ic2_wi, ' and  ic2_1_wi + ic2_1_wi - ic2_wi =', ic2_1_wi + ic2_2_wi - ic2_wi)
print('ic2_1_wi =', ic2_1_wi, ' and ic2_2_wi =', ic2_2_wi, ' and ic2_1_wi - ic2_1_wi =', ic2_1_wi - ic2_2_wi)
print('max_ir2_wi - ic2_wi = ', err_max_ir2_ic2_wi)
