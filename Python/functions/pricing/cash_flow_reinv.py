# -*- coding: utf-8 -*-
import numpy as np


def cash_flow_reinv(c, r, t_m, inv):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=cash_flow_reinv).

    Parameters
    ----------
        c : array, shape (k_,)
        r : array, shape (k_,)
        t_m : array, shape (m_+1,)
        inv : array, shape (j_, m_) if m_>1 or (j_,) for m_=1

    Returns
    -------
        cf_tnow_thor : array, shape(j_, m_)

    """
    if len(inv.shape) == 1:
        j_, m_ = inv.shape[0], 1
    else:
        j_, m_ = inv.shape
    k_ = r.shape[0]

    cf = np.zeros((j_, m_, k_))

    # Step 0: Find monit. time indexes corresponding to the coupon pay. dates

    ml = np.array([np.where(t_m == rc) for rc in r]).reshape(-1)
    ml = np.array([mll for mll in ml if mll.size != 0]).reshape(-1)
    l_ = len(ml)

    for l in np.arange(l_):

        # Step 1: Compute reinvestment factors from each payment day

        m_l = ml[l]
        if m_l != m_:
            inv_tmk = inv[:, m_l:]
        else:
            inv_tmk = np.ones((j_, 1))

        # Step 2: Compute scenarios for the cumulative cash-flow path

        cf[:, m_l:, l] = c[l] * np.cumprod(inv_tmk, axis=1)

    # compute cumulative reinvested cash-flow stream
    cf_tnow_thor = np.sum(cf, 2)

    return np.squeeze(cf_tnow_thor)
