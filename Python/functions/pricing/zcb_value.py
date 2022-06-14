# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate

from arpym.pricing.nelson_siegel_yield import nelson_siegel_yield
from arpym.pricing.shadowrates_ytm import shadowrates_ytm

def zcb_value(t_hor, x_thor, tau, t_end, rd='y', facev=1, eta=0.013):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=zcb_value).

    Parameters
    ----------
        t_hor : date
        x_thor : array, shape(j_, d_) if d_>1 or (t_,) for d_=1
        tau : array, shape(d_,)
        t_end : array of dates, shape(k_,)
        rd : string, optional
        facev : array, optional, shape(k_,)
        eta : scalar, optional

    Returns
    -------
        v : array, shape(j_, k_) if k_>1 or (t_,) for k_=1

    """

    j_ = x_thor.shape[0]
    k_ = t_end.shape[0]

    if (rd != 'sr') and (rd != 'ns'):
        rd = 'y'

    if isinstance(facev, int):
        facev = facev*np.ones(k_)

    # Step 1: Compute the time to maturiy at the horizon for each zcb
    tau_star = np.array([np.busday_count(t_hor, t_end[i])/252
                        for i in range(k_)])

    # Step 2: Compute the yield for each time to maturity
    if rd == 'y':

        # Step 2a (risk drivers are yields)
        interp = interpolate.interp1d(tau.flatten(), x_thor, axis=1,
                                         fill_value='extrapolate')
        x_star = interp(tau_star)

    elif rd == 'sr':

        # Step 2b (risk drivers are shadow rates)
        interp = interpolate.interp1d(tau.flatten(), x_thor, axis=1,
                                         fill_value='extrapolate')
        # Transform shadow rates to yields
        x_star = shadowrates_ytm(interp(tau_star), eta)

    elif rd == 'ns':

        # Step 2c (risk drivers are NS parameters)
        x_star = np.zeros((j_, k_))
        idx_nonzero = (tau_star > 0)
        for j in range(j_):
            x_star[j, idx_nonzero] = nelson_siegel_yield(tau_star[idx_nonzero]
                                                         , x_thor[j])

    # Step 3: Compute the value of each zero coupon-bond
    v = facev*np.exp(- tau_star * x_star)

    return np.squeeze(v)
