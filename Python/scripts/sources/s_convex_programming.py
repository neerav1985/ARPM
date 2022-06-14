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

# # s_convex_programming [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_convex_programming&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_convex_programming).

# +
import numpy as np
from cvxpy import *
from cvxopt import solvers, matrix
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from arpym.estimation.enet import enet

solvers.options['show_progress'] = False
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_convex_programming-implementation-step01): Convex programming

# +
a_eq = np.array([[1., 2., 3.]])
b_eq = np.array([[1.]])

# optimization variable
x = Variable((3,1))

# build optimization problem
prob = Problem(Minimize(x[0]**2+x[1]**4+exp(-x[2])), [abs(x[0])**3+abs(x[1])**5+x[2]**6-5 <= 0, a_eq*x == b_eq])

# solve optimization problem
result = prob.solve()

# result
print('x* =', x.value)
print('f(x*) =', prob.value)
# -

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_convex_programming-implementation-step02): Conic programming

c = matrix([1., -2., 1.])
G = matrix([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
h = matrix([0., 0., 0.])
sol = solvers.lp(c, G, h)
print('x* =', sol['x'])
print('f(x*) =', sol['primal objective'])


# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_convex_programming-implementation-step03): Semidefinite programming

c = matrix([1., -2., 1.])
A = matrix(a_eq)
b = matrix(b_eq)
u_1 = np.array(([ 1., -1.],
                [-1.,  0.]))
u_2 = np.array(([ 1.,  0.],
                [ 0.,  1.]))
u_3 = np.array(([ 0., -1.],
                [-1.,  1.]))
u = np.stack([u_1.reshape(-1),
             u_2.reshape(-1),
             u_3.reshape(-1)]).T
u = [matrix(u)]
v = [matrix([[1., 2.], [2., 3.]])]
sol = solvers.sdp(c, Gs=u, hs=v, A=A, b=b)
print('x* =', sol['x'])
print('f(x*) =', sol['primal objective'])

# ## [Step 4](https://www.arpm.co/lab/redirect.php?permalink=s_convex_programming-implementation-step04): Second order cone programming

c = matrix([1., -2., 1.])
A = matrix(a_eq)
b = matrix(b_eq)
G = matrix([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
h = matrix([0., 0., 0.])
sol = solvers.socp(c, Gl = G, hl = h, A=A, b=b)
print('x* =', sol['x'])
print('f(x*) =', sol['primal objective'])

# ## [Step 5](https://www.arpm.co/lab/redirect.php?permalink=s_convex_programming-implementation-step05): Quadratic programming


# +
a_ineq = np.array([[2., -1., 4.]])
b_ineq = np.array([[3.]])

P = matrix(np.eye(3))
q = matrix(np.zeros((3, 1)))

G = matrix(a_ineq)
h = matrix(b_ineq)

A = matrix(a_eq)
b = matrix(b_eq)

sol = solvers.qp(P, q, G, h, A=A, b=b)
print('x* =', sol['x'])
print('f(x*) =', sol['primal objective'])
# -

# ## [Step 6](https://www.arpm.co/lab/redirect.php?permalink=s_convex_programming-implementation-step06): Ridge

u = np.array([[-3., 2., -0.8],
              [1., -0.7, 2.1],
              [2., -5., 0.1],
              [2.9, 3.1, -0.6]])
v = np.array([[2., -1.],
              [1., -0.7],
              [-4.1, 5.3],
              [1.4, 3.1]])
alpha = 3
x = Ridge(alpha=alpha, fit_intercept=False).fit(u, v).coef_.T
print(x)

# ## [Step 7](https://www.arpm.co/lab/redirect.php?permalink=s_convex_programming-implementation-step07): Lasso

lam = 2
x = Lasso(alpha=lam, fit_intercept=False).fit(u, v).coef_.T
print(x)

# ## [Step 8](https://www.arpm.co/lab/redirect.php?permalink=s_convex_programming-implementation-step08): Elastic net

x = ElasticNet(alpha=alpha+lam, l1_ratio=alpha/(alpha+lam), fit_intercept=False).fit(u, v).coef_.T
print(x)

# ## [Step 9](https://www.arpm.co/lab/redirect.php?permalink=s_convex_programming-implementation-step09): Constrained elastic net

# +
a_eq = np.array([[1., -2., 0.], [-1.2, 2.1, 0.3]])
b_eq = np.array([[-1., 4.], [1.4, 3.1]])
a_ineq = np.array([[0.9, -1.2, 3.1], [-0.8, -1.3, 2.1]])
b_ineq = np.array([[0., 0.], [-1., -2.]])

x = enet(v, u, alpha, lam, a_eq, b_eq, a_ineq, b_ineq, eps=10**-4)
print(x)
# -

# ## [Step 9](https://www.arpm.co/lab/redirect.php?permalink=s_convex_programming-implementation-step09): Linear programming

c = matrix([1., -2., 1.])
G = matrix([[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
h = matrix([0., 0., 0.])
sol = solvers.lp(c, G, h)
print('x* =', sol['x'])
print('f(x*) =', sol['primal objective'])
