# -*- coding: utf-8 -*-
import numpy as np


def black_litterman(mu_, s2_, tau, v_mu, mu_view, sig2_view):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-blformula-fun).

    Parameters
    ----------
        mu_ : array, shape (n_,)
        s2_ : array, shape (n_, n_)
        tau : scalar
        v_mu : array, shape (k_, n_)
        mu_view : array, shape (k_,)
        sig2_view : array, shape (k_, k_)

    Returns
    -------
        blmu : array, shape (n_,)
        bls2 : array, shape (n_, n_)

    """
    
    sig2pos = np.linalg.inv(np.linalg.inv((1/tau)*s2_) + v_mu.T@np.linalg.inv(sig2_view)@v_mu)
    pos = (1/tau)*v_mu@s2_@v_mu.T + sig2_view

    blmu = mu_ + (1/tau)*s2_@v_mu.T@np.linalg.solve(pos, mu_view - v_mu@mu_)
    
    bls2 = sig2pos+s2_

    return blmu, bls2
