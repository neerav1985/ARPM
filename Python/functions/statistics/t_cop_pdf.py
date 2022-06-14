# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
from numpy.linalg import solve, det
from scipy.special import gamma
from scipy.stats import t

def t_cop_pdf(u, nu, mu, sigma2):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=tCopFunction).

    Parameters
    ----------
        u :  array, shape (n_,)
        nu : scalar
        mu : array, shape (n_,)
        sigma2 :  array, shape (n_, n_)

    Returns
    -------
        f_u : scalar

    """

    # Step 1: Compute the inverse marginal cdf's

    sigvec = np.sqrt(np.diag(sigma2))
    x = mu.flatten() + sigvec * t.ppf(u.flatten(), nu)

    # Step 2: Compute the joint pdf

    n_ = len(u)
    z2 = (x - mu).T@solve(sigma2,(x - mu))
    const  = (nu*np.pi)**(-n_ / 2)*gamma((nu + n_) / 2) / gamma(nu / 2)*((det(sigma2))**(-.5))
    f_x = const*(1 + z2 / nu) ** (-(nu + n_) / 2)
    
    # Step 3: Compute the marginal pdf's

    f_xn = stats.t.pdf((x.flatten()-mu.flatten())/ sigvec, nu)/sigvec

    # Step 4: Compute the pdf of t copula

    f_u = np.squeeze(f_x / np.prod(f_xn))

    return f_u