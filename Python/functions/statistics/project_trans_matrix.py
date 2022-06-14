# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import expm, logm
from cvxopt import matrix
from cvxopt.solvers import qp, options

from arpym.views.min_rel_entropy_sp import min_rel_entropy_sp

def project_trans_matrix(p, delta_t, credit=False):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=project_trans_matrix).

    Parameters
    ----------
        p : array, shape (c_, c_)
        delta_t : scalar
        credit : boolean (optional)

    Returns
    -------
        p_delta_t : array, shape (c_, c_)

    """

    c_ = len(p)

    # Step 1: Compute log-matrix

    l = logm(p)    # log-matrix of p1

    # Step 2: Compute generator
    
    p_a = matrix(np.eye(c_*c_))
    q = matrix(-l.reshape((c_*c_, 1)))
    G = matrix(0.0, (c_*c_, c_*c_))
    G[::c_*c_+1] = np.append([0], np.tile(np.append(-np.ones(c_), [0]), c_-1))
    h = matrix(0.0, (c_*c_, 1))
    a = matrix(np.repeat(np.diagflat(np.ones(c_)), c_, axis=1))
    b = matrix(0.0, (c_, 1))
    options['show_progress'] = False
    g = qp(p_a, q, G, h, a, b)['x']    # generator matrix

    # Step 3: Compute projected transition matrix
    
    g = np.array(g).reshape((c_, c_)) 
    p_delta_t = expm(delta_t*g)    # projected transition matrix

    if credit is True:
        p = p_delta_t
        
        # probability constraint
        a_eq = np.ones((1, c_))   # 1×c_ dimensional vector of ones
        b_eq = np.array([1])

        # initialize monotonicity constraint
        a_ineq = {}
        a_ineq[0] = np.diagflat(np.ones((1, c_-1)), 1) -\
                   np.diagflat(np.ones((1, c_)), 0)     # (c_-1)×c_ upper triangular matrix
        a_ineq[0] = a_ineq[0][:-1]
        b_ineq = np.zeros((c_-1))    # 1×(c_-1) dimensional vector of ones
    
        p_delta_t = np.zeros((c_, c_))
        for c in range(c_-1):
            p_delta_t[c, :] = min_rel_entropy_sp(p[c, :],    # minimize relative entropy
                                 a_ineq[c], b_ineq, a_eq, b_eq, False)
            a_temp = a_ineq.get(c).copy()    # update monotonicity constraint
            a_temp[c, :] = -a_temp[c, :]
            a_ineq[c+1] = a_temp.copy()

        p_delta_t[-1, :] = np.zeros((1, p.shape[1]))    # default constraint
        p_delta_t[-1, -1] = 1
        
    return p_delta_t
