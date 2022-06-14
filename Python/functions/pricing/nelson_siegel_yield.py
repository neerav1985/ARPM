import numpy as np


def nelson_siegel_yield(tau, theta):
    """For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=nelson_siegel_yield).
     
    Parameters
    ----------
        tau : array, shape (n_,)
        theta : array, shape (4,)

    Returns
    -------
        y_ns : array, shape (n_,)

    """
 
    y_ns = theta[0] - theta[1] * \
        ((1 - np.exp(-theta[3] * tau)) /
         (theta[3] * tau)) + theta[2] * \
        ((1 - np.exp(-theta[3] * tau)) /
         (theta[3] * tau) - np.exp(-theta[3] * tau))

    return np.squeeze(y_ns)
