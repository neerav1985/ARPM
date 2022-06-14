# -*- coding: utf-8 -*-

import numpy as np
from cvxopt import matrix, solvers
from scipy.linalg import null_space
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from arpym.tools.transpose_square_root import transpose_square_root

solvers.options['show_progress'] = False


def vectorization(v, u,
                  a_eq=np.array([np.nan]), b_eq=np.array([np.nan]),
                  a_ineq=np.array([np.nan]), b_ineq=np.array([np.nan]),
                  delta=np.array([np.nan]), d=np.array([np.nan]),
                  y=np.array([np.nan])):
    if v.ndim==1:
        v = np.atleast_2d(v).T
    if u.ndim==1:
        u = np.atleast_2d(u).T
    if a_eq.ndim==1:
        a_eq = np.atleast_2d(a_eq).T
    if b_eq.ndim==1:
        b_eq = np.atleast_2d(b_eq).T
    if a_ineq.ndim==1:
        a_ineq = np.atleast_2d(a_ineq).T
    if b_ineq.ndim==1:
        b_ineq = np.atleast_2d(b_ineq).T
    if delta.ndim==1:
        delta = np.atleast_2d(delta).T
    if d.ndim==1:
        d = np.atleast_2d(d).T
    if y.ndim==1:
        y = np.atleast_2d(y).T

    if np.isnan(a_eq).any():
        a_eq = np.nan*np.eye(np.shape(u)[1])
    if np.isnan(b_eq).any():
        b_eq = np.nan*np.ones((np.shape(u)[1], v.shape[1]))
    if np.isnan(a_ineq).any():
        a_ineq = np.nan*np.eye(np.shape(u)[1])
    if np.isnan(b_ineq).any():
        b_ineq = np.nan*np.ones((np.shape(u)[1], v.shape[1]))
    if np.isnan(delta).any():
        delta = np.eye(np.shape(u)[1])
    if np.isnan(d).any():
        d = np.eye(np.shape(u)[1])
    if np.isnan(y).any():
        y = np.zeros((u.shape[1], v.shape[1]))

    l_ = np.shape(v)[1]

    u = np.kron(np.eye(l_), u)
    v = v.reshape(-1, 1, order='F')

    a_eq = np.kron(np.eye(l_), a_eq)
    b_eq = b_eq.reshape(-1, 1, order='F')

    a_ineq = np.kron(np.eye(l_), a_ineq)
    b_ineq = b_ineq.reshape(-1, 1, order='F')

    delta = np.kron(np.eye(l_), delta)
    d = np.kron(np.eye(l_), d)      

    y = y.reshape(-1, 1, order='F')

    return v, u, a_eq, b_eq, a_ineq, b_ineq, delta, d, y


def enet2lasso(v, u, alpha=0, delta=np.array([np.nan])):
    m_ = v.shape[0]
    if not np.isnan(delta).any():
        u_ = transpose_square_root(u.T@u + 2*alpha*m_*delta.T@delta, method='Cholesky').T
    else:
        u_ = transpose_square_root(u.T@u + 2*alpha*m_, method='Cholesky').T
    v_ = np.linalg.solve(u_.T, u.T)@v

    return v_, u_


def constrained_lasso(v, u, lam=0, a_eq=np.array([np.nan]), b_eq=np.array([np.nan]), a_ineq=np.array([np.nan]), b_ineq=np.array([np.nan])):
    m_, n_ = np.shape(u)
    q2 = np.block([[u.T@u, -u.T@u],
                   [-u.T@u, u.T@u]])
    c_lam = lam*m_ - np.block([[u.T@v], [-u.T@v]])
    if (not np.isnan(a_eq).any() and not np.isnan(b_eq).any()):
        a_eq_lasso = np.block([a_eq, -a_eq])
        b_eq_lasso = b_eq
    if (not np.isnan(a_ineq).any() and not np.isnan(b_ineq).any()):
        a_ineq_lasso = np.block([[a_ineq, -a_ineq],
                                [-np.eye(n_), np.zeros((n_, n_))],
                                [np.zeros((n_, n_)), -np.eye(n_)]])
        b_ineq_lasso = np.block([[b_ineq], [np.zeros((2*n_, 1))]])

    P = matrix(q2)
    q = matrix(c_lam)

    if (not np.isnan(a_eq).any() and not np.isnan(b_eq).any() and not np.isnan(a_ineq).any() and not np.isnan(b_ineq).any()):
        A = matrix(a_eq_lasso)
        b = matrix(b_eq_lasso)
        G = matrix(a_ineq_lasso)
        h = matrix(b_ineq_lasso)

        x_vec_opt = np.array(solvers.qp(P, q, G, h, A, b)['x'])
    elif (np.isnan(a_eq).any() and np.isnan(b_eq).any() and not np.isnan(a_ineq).any() and not np.isnan(b_ineq).any()):
        G = matrix(a_ineq_lasso)
        h = matrix(b_ineq_lasso)

        x_vec_opt = np.array(solvers.qp(P, q, G=G, h=h)['x'])
    elif (not np.isnan(a_eq).any() and not np.isnan(b_eq).any() and np.isnan(a_ineq).any() and np.isnan(b_ineq).any()):
        A = matrix(a_eq_lasso)
        b = matrix(b_eq_lasso)
        G = matrix(np.block([[-np.eye(n_), np.zeros((n_, n_))],
                              [np.zeros((n_, n_)), -np.eye(n_)]]))
        h = matrix(np.zeros((2*n_, 1)))

        x_vec_opt = np.array(solvers.qp(P, q, G, h, A, b)['x'])
    elif (np.isnan(a_eq).any() and np.isnan(b_eq).any() and np.isnan(a_ineq).any() and np.isnan(b_ineq).any()):
        G = matrix(np.block([[-np.eye(n_), np.zeros((n_, n_))],
                              [np.zeros((n_, n_)), -np.eye(n_)]]))
        h = matrix(np.zeros((2*n_, 1)))
        x_vec_opt = np.array(solvers.qp(P, q, G=G, h=h)['x'])
        
    x_vec_opt_plus = x_vec_opt[:n_]
    x_vec_opt_minus = x_vec_opt[n_:]

    x_lasso = (x_vec_opt_plus - x_vec_opt_minus).squeeze()

    return x_lasso


def enet(v, u, alpha=0, lam=0,
         a_eq=np.array([np.nan]), b_eq=np.array([np.nan]),
         a_ineq=np.array([np.nan]), b_ineq=np.array([np.nan]),
         delta=np.array([np.nan]), d=np.array([np.nan]),
         y=np.array([np.nan]),
         s_k = [],
         eps=10**(-9)):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=enet).

    Parameters
    ----------
         v : array, shape(m_, l_)
         u : array, shape(m_, n_)
         alpha : scalar
         lam : scalar
         a_eq : array, shape(p_, n_)
         b_eq : array, shape(p_, l_)
         a_ineq : array, shape(q_, n_)
         b_ineq : array, shape(q_, l_)
         delta : array, shape(s_, n_)
         d : array, shape(r_, n_)
         y : array, shape(n_, l_)
         s_k : list
         eps : scalar

    Returns
    -------
        x_enet : array, shape(n_, l_)

    """
    # In case of standard regression, lasso, ridge or elastic nets, use built-in routines
    regression = (alpha==0 and lam==0 and np.isnan(a_eq).any() and np.isnan(b_eq).any()
                  and np.isnan(a_ineq).any() and np.isnan(b_ineq).any()
                  and np.isnan(delta).any() and np.isnan(d).any() and not s_k)
    ridge = (alpha!=0 and lam==0 and np.isnan(a_eq).any() and np.isnan(b_eq).any()
             and np.isnan(a_ineq).any() and np.isnan(b_ineq).any()
             and np.isnan(delta).any() and np.isnan(d).any() and not s_k)
    lasso = (alpha==0 and lam!=0 and np.isnan(a_eq).any() and np.isnan(b_eq).any()
             and np.isnan(a_ineq).any() and np.isnan(b_ineq).any()
             and np.isnan(delta).any() and np.isnan(d).any() and not s_k)
    enet = (alpha!=0 and lam!=0 and np.isnan(a_eq).any() and np.isnan(b_eq).any()
            and np.isnan(a_ineq).any() and np.isnan(b_ineq).any()
            and np.isnan(delta).any() and np.isnan(d).any() and not s_k)
    if np.isnan(y).any():
        y = np.zeros((np.shape(u)[1], np.shape(v)[1]))
    v = v - u@y
    if regression:  # regression
        print('Regression')
        x_enet = LinearRegression(fit_intercept=False).fit(u, v).coef_
        return x_enet.T + y.squeeze()

    if ridge:  # ridge
        print('Ridge')
        x_enet = Ridge(alpha=alpha, fit_intercept=False).fit(u, v).coef_
        return x_enet.T + y.squeeze()

    elif lasso:  # lasso
        print('Lasso')
        x_enet = Lasso(alpha=lam, fit_intercept=False).fit(u, v).coef_
        return x_enet.T + y.squeeze()

    elif enet:  # elastic net
        print('Elastic net')
        x_enet = ElasticNet(alpha=lam+alpha, l1_ratio=lam/(lam+alpha), fit_intercept=False).fit(u, v).coef_
        return x_enet.T + y.squeeze()

    else:  # constrained generalized elastic net
        # dimensions
        m_, l_ = v.shape
        n_ = u.shape[1]
        p_ = a_eq.shape[0]
        q_ = a_eq.shape[0]
        if np.isnan(delta).any():
            delta = np.eye(n_)
        s_ = delta.shape[0]
        if np.isnan(d).any():
            d = np.eye(n_)
        r_ = d.shape[0]

        if s_k:  # add selection constraints
            all_ind = [[r, l] for r in range(r_) for l in range(l_)]
            not_s_k = [item for item in all_ind if item not in s_k]
            s_ = np.zeros((len(not_s_k), 1, r_))
            s = np.zeros((len(not_s_k), 1, n_))
            t = np.zeros((len(not_s_k), l_, 1))
            c = 0
            for ind in not_s_k:
                r = ind[0]
                l = ind[1]
                s_[c, 0, r] = 1
                s[c] = s_[c]@d
                t[c, l, 0] = 1
                c = c + 1

        # Step 1: Vectorization
        v, u, a_eq, b_eq, a_ineq, b_ineq, delta, d, y = vectorization(v, u, a_eq, b_eq, a_ineq, b_ineq, delta, d, y)

        if s_k:  # add constraints (cont.d)
            for c in range(len(not_s_k)):
                a_eq = np.vstack([a_eq, np.kron(t[c].T, s[c])])
                b_eq = np.vstack([b_eq, 0])
        # Step 2: Elastic net to lasso
        v, u = enet2lasso(v, u, alpha, delta)

        # Step 3: Generalized constrained elastic net
        if r_>n_:
            print('d must be an array with shape(r_, n_) with r_ lower than or equal n_')
            return None
        elif r_==n_:  # invertible d
            # change of variables
            d_inv = np.linalg.solve(d, np.eye(d.shape[0]))
            u = u@d_inv
            a_eq = a_eq@d_inv
            b_eq = b_eq - (a_eq@d)@y
            a_ineq = a_ineq@d_inv
            b_ineq = b_ineq - (a_ineq@d)@y

            # constrained lasso
            x_lasso = constrained_lasso(v, u, lam, a_eq, b_eq, a_ineq, b_ineq)
            x_enet = d_inv@x_lasso + y.squeeze()
        else:  # orthogonal completion
            d_ort = null_space(d).T
            d_tilde = np.block([[d], [eps*d_ort]])
            d_tilde_inv = np.linalg.solve(d_tilde, np.eye(d_tilde.shape[0]))

            # change of variables
            u = u@d_tilde_inv
            a_eq = a_eq@d_tilde_inv
            b_eq = b_eq - (a_eq@d_tilde)@y
            a_ineq = a_ineq@d_tilde_inv
            b_ineq = b_ineq - (a_ineq@d_tilde)@y

            # constrained lasso
            x_lasso = constrained_lasso(v, u, lam, a_eq, b_eq, a_ineq, b_ineq)
            x_enet = d_tilde_inv@x_lasso + y.squeeze()

    x_enet = x_enet.reshape(n_, -l_, order='F').squeeze()

    return x_enet
