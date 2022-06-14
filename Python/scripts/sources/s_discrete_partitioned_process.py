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

# # s_discrete_partitioned_process [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_discrete_partitioned_process&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_discrete_partitioned_process).

import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.integrate import dblquad, nquad

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_discrete_partitioned_process-parameters)

# +
mu = 0.05  # drift of underlying BM
sig2 = 0.2  # variance of underlying BM
v_0 = 20  # current value
ev = np.array([[0, 0.5], [0, 0.5]])
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_discrete_partitioned_process-implementation-step01): Brownian motion parameters

# +
# expectation vector
mu_x = np.array([np.log(v_0)+mu, np.log(v_0)+2*mu])
# covariance matrix
sig2_x = np.array([[sig2, sig2], [sig2, 2*sig2]])
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_discrete_partitioned_process-implementation-step02): Partition

# +
j_ = 3  # number of element of the partition
x_j = np.append(np.append(-np.inf,
                          np.linspace(np.log(v_0)+2*mu-np.sqrt(2*sig2),
                                      np.log(v_0)+2*mu+np.sqrt(2*sig2), j_-1)),
                np.inf)
print("Δ(R)={X⁽¹⁾ = (%s, %.2f],"
      "\n      X⁽²⁾ = (%.2f, %.2f],"
      "\n      X⁽³⁾ = (%.2f, %s)}"
      % (x_j[0], x_j[1], x_j[1], x_j[2], x_j[2], x_j[3]))
# -

# ##  Step 3: Level 1 and level 2 partitions in sample space

# +
sig = np.sqrt(sig2)
x_1_inverse = (x_j-np.log(v_0)-mu)/sig
# level 1 partition in the sample space
delta_1_x = norm.cdf(x_1_inverse)
print("Δ⁽¹⁾(Ω)={Δω⁽¹⁾ = [%.3f, %.3f] × [0, 1],"
      "\n         Δω⁽²⁾ = (%.3f, %.3f] × [0, 1],"
      "\n         Δω⁽³⁾ = (%.3f, %.3f] × [0, 1]}"
      % (delta_1_x[0], delta_1_x[1], delta_1_x[1],
         delta_1_x[2], delta_1_x[2], delta_1_x[3]))
# level 2 partition in the sample space
x_w = (x_j-np.log(v_0)-2*mu)/sig
print("Δ⁽²⁾(Ω)={Δω^{(1,1)} = [%.3f, %.3f] × [0, Φ(%.3f Φ⁻¹(ω₁))],"
      "\n   Δω^{(2,1)} = (%.3f, %.3f] × [0, Φ(%.3f-Φ⁻¹(ω₁))],"
      "\n   Δω^{(3,1)} = (%.3f, %.3f] × [0, Φ(%.3f - Φ⁻¹(ω₁))],"
      "\n   Δω^{(1,2)} = [%.3f, %.3f] × (Φ(%.3f Φ⁻¹(ω₁)), Φ(%.3f-Φ⁻¹(ω₁))],"
      "\n   Δω^{(2,2)} = (%.3f, %.3f] × (Φ(%.3f-Φ⁻¹(ω₁)), Φ(%.3f-Φ⁻¹(ω₁))],"
      "\n   Δω^{(3,2)} = (%.3f, %.3f] × (Φ(%.3f-Φ⁻¹(ω₁)), Φ(%.3f-Φ⁻¹(ω₁))],"
      "\n   Δω^{(1,3)} = [%.3f, %.3f] × (Φ(%.3f-Φ⁻¹(ω₁)), 1],"
      "\n   Δω^{(2,3)} = (%.3f, %.3f] × (Φ(%.3f-Φ⁻¹(ω₁)), 1],"
      "\n   Δω^{(3,3)} = (%.3f, %.3f] × (Φ(%.3f-Φ⁻¹(ω₁)), 1]}"
      % (delta_1_x[0], delta_1_x[1], x_w[1],
         delta_1_x[1], delta_1_x[2], x_w[1],
         delta_1_x[2], delta_1_x[3], x_w[1],
         delta_1_x[0], delta_1_x[1], x_w[1], x_w[2],
         delta_1_x[1], delta_1_x[2], x_w[1], x_w[2],
         delta_1_x[2], delta_1_x[3], x_w[1], x_w[2],
         delta_1_x[0], delta_1_x[1], x_w[2],
         delta_1_x[1], delta_1_x[2], x_w[2],
         delta_1_x[2], delta_1_x[3], x_w[2]))
# -

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_discrete_partitioned_process-implementation-step04): Prob. of first-step and second-step events and cond. probabilities

# +
# probabilities of first-step events
p_1 = np.zeros(j_)
for j1 in range(j_):
    p_1[j1] = nquad(lambda x_1, x_2:
                    multivariate_normal.pdf(np.array([x_1, x_2]), mu_x, sig2_x),
                    [[x_j[j1], x_j[j1+1]], [x_j[0], x_j[-1]]])[0]
    print("p^{(%d)} = %.3f" % (j1+1, p_1[j1]))

# probabilities of second-step events
p_2 = np.zeros((j_, j_))
# conditional probabilities
p_cond = np.zeros((j_, j_))
for j1 in range(j_):
    for j2 in range(j_):
        # prob. of the second-step events
        p_2[j1, j2] = nquad(lambda x_1, x_2:
                            multivariate_normal.pdf(np.array([x_1, x_2]),
                                                    mu_x, sig2_x),
                            [[x_j[j1], x_j[j1+1]], [x_j[j2], x_j[j2+1]]])[0]
        # conditional probabilities
        p_cond[j1, j2] = p_2[j1, j2]/p_1[j1]
        print("p^{(%d,%d)} = %.5f" % (j1+1, j2+1, p_2[j1, j2]))
        print("p^{(%d|%d)} = %.3f" % (j2+1, j1+1, p_cond[j1, j2]))
# -

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_discrete_partitioned_process-implementation-step05): Iterated expectations at t = 1, 2; law of iterated expectations

# +
#  iterated expectation at t = 1
cond_exp_1_z = np.zeros(j_)
for j1 in range(j_):
    cond_exp_1_z[j1] = nquad(lambda w_1, w_2: w_1+w_2,
                             [[delta_1_x[j1], delta_1_x[j1+1]],
                              [0, 1]])[0]/p_1[j1]
    print("E₁{Z}(ω₁,ω₂) = %.2f if X₁(ω₁,ω₂) ∈ X^(%d)"
          % (cond_exp_1_z[j1], j1+1))

#  iterated expectation at t = 2
cond_exp_2_z = np.zeros((j_, j_))
for j1 in range(j_):
    for j2 in range(j_):
        cond_exp_2_z[j1, j2] = dblquad(lambda w_1, w_2: w_1+w_2,
                                       delta_1_x[j1], delta_1_x[j1+1],
                                       lambda w:
                                           norm.cdf(x_w[j2]-norm.ppf(w)),
                                       lambda w:
                                           norm.cdf(x_w[j2+1]-norm.ppf(w)))[0]\
                               / p_2[j1, j2]
        print('E₂{Z}(ω₁,ω₂) = %.2f if X₁(ω₁,ω₂) ∈ X^(%d), X₂(ω₁,ω₂) ∈ X^(%d)'
              % (cond_exp_2_z[j1, j2], j1+1, j2+1))

# law of iterated expectations
cond_exp_12_z = np.zeros(j_)
for j1 in range(j_):
    cond_exp_12_z[j1] = cond_exp_2_z[j1, :]@p_2[j1, :]/p_1[j1]
error = cond_exp_1_z - cond_exp_12_z
print(error)
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_discrete_partitioned_process-implementation-step06): Iterated probabilities of an event at t = 1, 2

# +
# iterated probabilities of an event at t = 1
p_e_1 = np.zeros(j_)

for j1 in range(j_):
    p_e_1[j1] = nquad(lambda w_1, w_2: float(w_1>= ev[0, 0] and w_1<=ev[0, 1]),
                     [[delta_1_x[j1], delta_1_x[j1+1]],
                      [ev[1, 0], ev[1, 1]]])[0]/p_1[j1]
    print("P₁{E}(ω₁,ω₂) = %.3f if X₁(ω₁,ω₂) ∈ X^(%d)" %(p_e_1[j1], j1+1))

# iterated probabilities of an event at t = 2
p_e_2 = np.zeros((j_, j_))
for j1 in range(j_):
    for j2 in range(j_):
        p_e_2[j1, j2]=dblquad(lambda w_1, w_2: float((w_1>=ev[0, 0] and w_1<=ev[0, 1]) and
                                                     (w_2>=ev[1, 0] and w_2<=ev[1, 1])),
                              delta_1_x[j1], delta_1_x[j1+1],
                              lambda w: norm.cdf(x_w[j2]-norm.ppf(w)),
                              lambda w: norm.cdf(x_w[j2+1]-norm.ppf(w)))[0]/p_2[j1, j2]
        print('P₂{E}(ω₁,ω₂) = %.3f if X₁(ω₁,ω₂) ∈ X^(%d), X₂(ω₁,ω₂) ∈ X^(%d)' 
              %(p_e_2[j1, j2], j1+1, j2+1))
# -

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_discrete_partitioned_process-implementation-step07): Adapted expectations at t = 1, 2

# +
# stochastic trading strategy
y_d1 = np.array([-1000, 0, 1000])
y_d2 = np.array([[-1000, 0, 1000],
                 [-1000, 0, 1000],
                 [-1000, 0, 1000]])

# cond. expectation at t=1
cond_exp_1_yd1 = np.zeros(j_)
for j1 in range(j_):
    cond_exp_1_yd1[j1] = nquad(lambda w_1, w_2: y_d1[j1],
                               [[delta_1_x[j1], delta_1_x[j1+1]],
                                [0, 1]])[0]/p_1[j1]
    print("E₁{Y₁^Δ}(ω₁,ω₂)= %.2f if X₁(ω₁,ω₂) ∈ X^(%d)"
          % (cond_exp_1_yd1[j1], j1+1))

cond_exp_1_yd2 = np.zeros((j_))
for j1 in range(j_):
    for j2 in range(j_):
        cond_exp_1_yd2[j1] = cond_exp_1_yd2[j1] + \
                             dblquad(lambda w_1, w_2: y_d2[j1, j2],
                                     delta_1_x[j1], delta_1_x[j1+1],
                                     lambda w: norm.cdf(x_w[j2]-norm.ppf(w)),
                                     lambda w: norm.cdf(x_w[j2+1]-norm.ppf(w)))[0]
    cond_exp_1_yd2[j1] = cond_exp_1_yd2[j1]/p_1[j1]
    print('E₁{Y₂^{Δ}}(ω₁,ω₂) = %.2f if X₁(ω₁,ω₂) ∈ X^(%d)'
          % (cond_exp_1_yd2[j1], j1+1))
# -

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_discrete_partitioned_process-implementation-step08): Radon-Nikodym process

# +
# expectation vector under q
muq_x = np.array([np.log(v_0)-sig2/2, np.log(v_0)-sig2])

# q probabilities of first-step event
q_1 = np.zeros(j_)
# radon-nikodym process at t = 1
d_1 = np.zeros(j_)
for j1 in range(j_):
    q_1[j1] = nquad(lambda x_1, x_2:
                    multivariate_normal.pdf(np.array([x_1, x_2]), muq_x, sig2_x),
                    [[x_j[j1], x_j[j1+1]], [x_j[0], x_j[-1]]])[0]
    d_1[j1] = q_1[j1]/p_1[j1] 
    print("D₁(ω₁,ω₂) = %.3f if X₁(ω₁,ω₂) ∈ X^(%d)" % (d_1[j1], j1+1))

# q probabilities of second-step event
q_2 = np.zeros((j_, j_))
# radon-nikodym process at t = 2
d_2 = np.zeros((j_, j_))
for j1 in range(j_):
    for j2 in range(j_):
        q_2[j1, j2] = nquad(lambda x_1, x_2:
                            multivariate_normal.pdf(np.array([x_1, x_2]),
                                                    muq_x, sig2_x),
                            [[x_j[j1], x_j[j1+1]],
                             [x_j[j2], x_j[j2+1]]])[0]
        d_2[j1, j2] = q_2[j1, j2]/p_2[j1, j2]
        print("D₂(ω₁,ω₂) = %.3f if X₁(ω₁,ω₂) ∈ X^(%d),  X₂(ω₁,ω₂) ∈ X^(%d)"
              % (d_2[j1, j2], j1+1, j2+1))
# -

# ## [Step 9](https://www.arpm.co/lab/redirect.php?permalink=s_discrete_partitioned_process-implementation-step09): Conditional expectations of DZ under P

# +
# Radon-Nikodym derivative as a random variable
def d(w_1, w_2):
    return np.exp((2*mu**2-sig2**2/2-(2*mu+sig2) *
                  (2*mu+sig*norm.ppf(w_1)+sig*norm.ppf(w_2)))/(2*sig2))

# iterated expectation at t=1
cond_exp_p1_dz = np.zeros(j_)
for j1 in range(j_):
    cond_exp_p1_dz[j1] = nquad(lambda w_1, w_2: d(w_1, w_2)*(w_1+w_2),
                               [[delta_1_x[j1], delta_1_x[j1+1]],
                                [0, 1]])[0]/p_1[j1]
    print("E₁^{P}{DZ}(ω₁,ω₂) = %.2f if X₁(ω₁,ω₂) ∈ X^(%d)"
          % (cond_exp_p1_dz[j1], j1+1))

# iterated expectation at t=2
cond_exp_p2_dz = np.zeros((j_, j_))
for j1 in range(j_):
    for j2 in range(j_):
        cond_exp_p2_dz[j1, j2] = dblquad(lambda w_1, w_2: d(w_1, w_2)*(w_1+w_2),
                                         delta_1_x[j1], delta_1_x[j1+1],
                                         lambda w: norm.cdf(x_w[j2] -
                                                            norm.ppf(w)),
                                         lambda w: norm.cdf(x_w[j2+1] -
                                                            norm.ppf(w)))[0] /\
                                p_2[j1, j2]
        print('E₂^{P}{DZ}(ω₁,ω₂) = %.2f if X₁(ω₁,ω₂) ∈ X^(%d), X₂(ω₁,ω₂) ∈ X^(%d)'
              % (cond_exp_p2_dz[j1, j2], j1+1, j2+1))
# -

# ## [Step 10](https://www.arpm.co/lab/redirect.php?permalink=s_discrete_partitioned_process-implementation-step10): Conditional expectations of Z under Q

# +
# q iterated expectation at t=1
cond_exp_q1_z = cond_exp_p1_dz/d_1
for j1 in range(j_):
    print("E₁^{Q}{Z}(ω₁,ω₂) = %.2f if X₁(ω₁,ω₂) ∈ X^(%d)"
          % (cond_exp_q1_z[j1], j1+1))

# q iterated expectation at t=2
cond_exp_q2_z = cond_exp_p2_dz/d_2
for j1 in range(j_):
    for j2 in range(j_):
        print('E₂^{Q}{Z}(ω₁,ω₂) = %.2f if X₁(ω₁,ω₂) ∈ X^(%d), X₂(ω₁,ω₂)∈ X^(%d)'
              % (cond_exp_q2_z[j1, j2], j1+1, j2+1))
# -

# ## [Step 11](https://www.arpm.co/lab/redirect.php?permalink=s_discrete_partitioned_process-implementation-step11): Conditional expectations of events under Q

# +
def d_times_1(w_1, w_2):
    return d(w_1, w_2) * float(w_1>= ev[0,0] and w_1<=ev[0,1])


def d_times_2(w_1, w_2):
    return d(w_1, w_2) * float((w_1>= ev[0,0] and w_1<=ev[0,1]) and
                               (w_2>= ev[1,0] and w_2<=ev[1,1]))


# iterated probabilities of an event at t=1
q_e_1 = np.zeros(j_)
for j in range(j_):
    q_e_1[j] = nquad(d_times_1, [[delta_1_x[j], delta_1_x[j+1]], [ev[1,0], ev[1,1]]])[0]/(d_1[j]*p_1[j])
    print("Q₁{E}(ω₁,ω₂) = %.2f if X₁(ω₁,ω₂) ∈ X^(%d)" %(q_e_1[j], j+1))

# iterated probabilities of an event at t=2
q_e_2 = np.zeros((j_, j_))
for j1 in range(j_):
    for j2 in range(j_):
        q_e_2[j1, j2]=dblquad(d_times_2, delta_1_x[j1], delta_1_x[j1+1],
                              lambda w: norm.cdf(x_w[j2]-norm.ppf(w)),
                              lambda w: norm.cdf(x_w[j2+1]-norm.ppf(w)))[0]/(d_2[j1, j2]*p_2[j1, j2])
        print('Q₂{E}(ω₁,ω₂) = %.3f if X₁(ω₁,ω₂) ∈ X^{(%d)} and X₂(ω₁,ω₂) ∈ X^(%d)'
              % (q_e_2[j1, j2], j1+1, j2+1))
