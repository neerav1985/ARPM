# -*- coding: utf-8 -*-
import numpy as np
from arpym.pricing.zcb_value import zcb_value


def bond_value(t_hor, x_thor, tau, c, r, rd, facev=1, eta=0.013):
   
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=bond_value).

    Parameters
    ----------
        t_hor : date
        x_thor : array, shape(j_, d_)
        tau : array, shape(k_,)
        c : array, shape(k_,)
        r : array, shape(k_,)
        rd : string, optional
        facev : scalar, optional
        eta : scalar, optional

    Returns
    -------
        v : array, shape(j_,)

    """
    
    if (rd != 'sr') and (rd != 'ns'):
        rd = 'y'
    
    # Step 0: Consider only coupons after the horizon time

    c = c[r >= t_hor]
    r = r[r >= t_hor]

    # Step 1: compute scenarios for coupon bond value

    # compute zero-coupon bond value
    v_zcb = zcb_value(t_hor, x_thor, tau, r, rd)

    # include notional
    c[-1] = c[-1] + 1

    # compute coupon bond value
    if np.ndim(v_zcb) == 1:
        v_zcb = v_zcb.reshape(1,v_zcb.shape[0]) 
    v = facev*(v_zcb @  c)
    

    return v.reshape(-1)
