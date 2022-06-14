# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate

from arpym.pricing.bsm_function import bsm_function
from arpym.pricing.shadowrates_ytm import shadowrates_ytm

def call_option_value(t_hor, x_s_thor, x_y_thor, tau_y, x_sig_thor, m_sig, tau_sig, k_strk,
                      t_end, sr=1, logsig=1, eta=0.013):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=call_option_value).

    Parameters
    ----------
        t_hor : date
        x_s_thor : scalar
        x_y_thor : scalar
        tau_y : shape(k_,)
        x_sig_thor : array, shape(k_*l_,) or shape(k_, l_)
        m_sig : shape(l_,)
        tau_sig : shape(k_,)
        k_strk : scalar
        t_end : date
        sr : boolean {0,1}
        logsig : boolean {0,1}
        eta : scalar


    Return
    ------
        v : scalar


    """

    x_sig_thor = x_sig_thor.reshape(-1)

    if (sr != 0) and (sr != 1):
        sr = 1

    if (logsig != 0) and (logsig != 1):
        logsig = 1

    # Step 1: Time to expiry of the call option at the horizon

    tau_star = np.busday_count(t_hor, t_end)/252

    # Step 2: Compute m moneyness

    m_star = np.log(np.exp(x_s_thor)/k_strk)/np.sqrt(tau_star)

    # Step 3: Yield/shadow yield for the time to expiry

    if x_y_thor.shape[0] == 1:
        x_y_star = x_y_thor
    else:
        interp = interpolate.interp1d(tau_y.flatten(), x_y, axis=1,
                                         fill_value='extrapolate')
        x_y_star = interp(tau_star)
    if sr == 1:
        x_y_star = shadowrates_ytm(x_y_star, eta)

    # Step 4: (Log-)implied volatility at the horizon moneyness and time to expire

    points = list(zip(*[grid.flatten() for grid in
                        np.meshgrid(*[tau_sig, m_sig])]))
    m_e = min(max(m_star, min(m_sig)), max(m_sig))  # extrapolation
    t_e = min(max(tau_star, tau_sig[0]), tau_sig[-1])  # extrapolation
    x_sig_star = interpolate.LinearNDInterpolator(points
                                                  , x_sig_thor)(*np.r_[t_e, m_e])
    if logsig == 1:
        x_sig_star = np.exp(x_sig_star)

    # Step 5: Call option value

    v_call= bsm_function(np.exp(x_s_thor), x_y_star, x_sig_star, m_star, tau_star)

    return v_call
