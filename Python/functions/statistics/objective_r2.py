# -*- coding: utf-8 -*-

import numpy as np

from arpym.statistics.multi_r2 import multi_r2


def objective_r2(s_k, s2_xz, n_, sigma2=None):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=R-SquareObj).

    Parameters
    ----------
        s_k : array, shape(k_,)
        s2_xz : array, shape(n_+m_, n_+m_)
        n_ : int
        sigma2 : array, shape(n_, n_), optional

    Returns
    -------
        r2 : float

    """
    k_ = s_k.shape[0]
    m_ = s2_xz.shape[0] - n_

    s_kc = s_k.copy()-1
    s2_x = s2_xz[:n_, :n_]
    s_xz = s2_xz[:n_, n_:][:, s_kc]
    beta = np.zeros((n_, m_))
    beta[:, s_kc] = s_xz
    s2_z = s2_xz[n_:, n_:][s_kc,:][:,s_kc]
    s2_z_inverse=np.linalg.inv(s2_z)
    s2_u = s2_x - s_xz @ s2_z_inverse @ (s_xz).T 
       
    r2 = multi_r2(s2_u, s2_x, sigma2)
    
    return s_k, -r2
