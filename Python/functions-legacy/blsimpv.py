from scipy.optimize import root, brentq

from blsprice import blsprice


def blsimpv(p, s, k, rf, t, div=0, cp=1):
    """
    Computes implied Black vol from given price, forward, strike and time.
    For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=bsm_implvol).
    """
    f = lambda x: blsprice(s, k, rf, t, x, div, cp) - p
    result = brentq(f, 1e-9, 1e+9)
    return result
