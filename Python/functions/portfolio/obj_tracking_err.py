# -*- coding: utf-8 -*-
import numpy as np

from cvxopt import matrix
from cvxopt import solvers


def obj_tracking_err(sig2_x_xb, s=None,
                    a_eq=np.array([np.nan]),
                    b_eq=np.array([np.nan]),
                    a_ineq=np.array([np.nan]),
                    b_ineq=np.array([np.nan])):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=TrackingErrObj).

    Parameters
    ----------
        sig2_x_xb : array, shape (n_ + 1, n_ + 1)
        s : array, shape (k_, ) or int
        a_eq : array, shape(p_, n_)
        b_eq : array, shape(p_, 1)
        a_ineq : array, shape(q_, n_)
        b_ineq : array, shape(q_, 1)
        
    Returns
    -------
    w_star : array, shape (n_, )
    te_wstar : scalar

    """
    # read the number of components in x
    n_ = np.shape(sig2_x_xb)[0] - 1
    
    
    # shift indices of instruments by -1 
    if s is None:
        s = np.arange(n_)
    elif np.isscalar(s):
        s=np.array([s-1]) 
    else:
        s=s-1

    # handle constraints
    if a_eq.ndim==1:
        a_eq = np.atleast_2d(a_eq).T
    if b_eq.ndim==1:
        b_eq = np.atleast_2d(b_eq).T
    if a_ineq.ndim==1:
        a_ineq = np.atleast_2d(a_ineq).T
    if b_ineq.ndim==1:
        b_ineq = np.atleast_2d(b_ineq).T

    # by default, consider long-only and budget constraints
    if np.isnan(a_eq).any():
        a_eq = np.ones((1, n_))
    if np.isnan(b_eq).any():
        b_eq = np.ones((1, 1))
    if np.isnan(a_ineq).any():
        a_ineq = -np.eye(n_)
    if np.isnan(b_ineq).any():
        b_ineq = np.zeros((n_, 1))

    ## Step 0: QP optimization setup
    
    #quadratic objective parameters
    q2 = sig2_x_xb[:-1, :-1]
    c = -(sig2_x_xb[:-1, -1].reshape(-1, 1))
    v = sig2_x_xb[-1, -1]

    not_s = np.array([n for n in range(n_) if n not in s])
    for n in not_s:
        delta_n = np.zeros((n_, 1))
        delta_n[n, 0] = 1
        a_eq = np.vstack([a_eq, delta_n.T])
        b_eq = np.vstack([b_eq, 0])

    ## Step 1: Solve the QP problem
    
    # prepare data types for CVXPOT
    P = matrix(q2, tc='d')
    q = matrix(c, tc='d')
    G = matrix(a_ineq, tc='d')
    h = matrix(b_ineq, tc='d')
    A = matrix(a_eq, tc='d')
    b = matrix(b_eq, tc='d')

    # run optimization function
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A=A, b=b)
    
    # prepare output
    w_star = np.array(sol['x'])
    te_wstar = np.sqrt(w_star.T@q2@w_star + 2*w_star.T@c + v)

    return np.squeeze(w_star), np.squeeze(te_wstar)
