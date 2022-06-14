# -*- coding: utf-8 -*-

import numpy as np
from arpym.estimation.enet import enet


def enet_selection(v, u, alpha=0,
         a_eq=np.array([np.nan]), b_eq=np.array([np.nan]),
         a_ineq=np.array([np.nan]), b_ineq=np.array([np.nan]),
         d=np.array([np.nan]),
         a=1.3,
         eps=10**-9,
         thr=10**-15,
         maxiter=500):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=enet_selection).

    Parameters
    ----------
         v : array, shape(m_, l_)
         u : array, shape(m_, n_)
         alpha : scalar
         a_eq : array, shape(p_, n_)
         b_eq : array, shape(p_, l_)
         a_ineq : array, shape(q_, n_)
         b_ineq : array, shape(q_, l_)
         d : array, shape(r_, n_)
         a : scalar
         eps : scalar
         thr : scalar
         maxiter : int

    Returns
    -------
        x_k_enet : array, shape(up to r_*l_, n_, l_)
        f_k_enet : array, shape(up to r_*l_, )
        s_k_enet : list
        k_lam : array, shape(up to r_*l_, )
        lam_vec : array, shape(up to r_*l_, )

    """

    if v.ndim==1:
        v = np.atleast_2d(v).T
    if u.ndim==1:
        u = np.atleast_2d(u).T
    m_, l_ = v.shape
    n_ = u.shape[1]
    if np.isnan(d).any():
        d = np.eye(n_)
    r_ = d.shape[0]

    # Step 0: Initialize
    eta = 1/(a*n_*l_)
    lam = 0.
    k = r_*l_
    k_l = r_*l_
    all_ind = [[r, l] for r in range(r_) for l in range(l_)]
    s_k = all_ind

    x_k_enet = []
    f_k_enet = []
    s_k_enet = []
    k_lam = [k]
    lam_vec = [lam]

    iterr=0
    while k>=1 and iterr<maxiter:
        # Step 1: Optimize over k-element selection
        x_k_enet_ = enet(v, u, alpha=0, lam=0,
                         a_eq=a_eq, b_eq=b_eq,
                         a_ineq=a_ineq, b_ineq=b_ineq,
                         d=d, s_k=s_k, eps=eps)
        f_k_enet_ = 1/(2*m_) * np.linalg.norm(u@x_k_enet_-v)**2
        x_k_enet.append(x_k_enet_)
        f_k_enet.append(f_k_enet_)
        s_k_enet.append(s_k)
        if k==1:
            break

        while k_l==k and iterr<maxiter:
            # Step 2: Increase penalty
            lam = lam + eta

            # Step 3: Solve constrained elastic net
            x_enet = enet(v, u, alpha=alpha, lam=lam,
                          a_eq=a_eq, b_eq=b_eq,
                          a_ineq=a_ineq, b_ineq=b_ineq,
                          d=d, s_k=s_k, eps=eps)
            if np.ndim(x_enet)==1:
                x_enet = x_enet.reshape((-1, 1))
            prod = d@x_enet

            # Step 4: Count elements
            k_l = np.count_nonzero(np.abs(prod)>thr)

            iterr = iterr+1
            if iterr==maxiter:
                print('Maximum number of iterations exceeded')
                break

        # Step 5: Update counter and selection
        k = k_l
        k_lam.append(k)
        lam_vec.append(lam)
        r, l = np.where(np.abs(prod)>thr)
        s_k = [list(x) for x in zip(r, l)]

    return np.array(x_k_enet), np.array(f_k_enet), s_k_enet, np.array(k_lam), np.array(lam_vec)
