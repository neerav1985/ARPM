# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_t_copula_norm_marginals [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_t_copula_norm_marginals&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=eb-2-ex-tcop-giv-marg).

# +
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from arpym.statistics.simulate_t import simulate_t
from arpym.statistics.t_cop_pdf import t_cop_pdf
from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_t_copula_norm_marginals-parameters)

# +
j_ = 1000  # number of scenarios
mu = np.array([0, 0])  # location
rho = 0.2  # correlation coefficient
svec = np.array([1, 1])  # standard deviations
nu = 10  # degrees of freedom

# grid in the unit square
grid = np.arange(0.01, 1, 0.01)
n_grid = len(grid)
# -

# ## Generate copula scenarios

# +
c2 = np.array([[1, rho], [rho, 1]])  # correlation matrix
sigma2 = np.diag(svec)@c2@np.diag(svec)  # covariance

z = simulate_t(mu, sigma2, nu, j_)  # t scenarios
u1 = stats.t.cdf(z[:, 0], nu, mu[0], svec[0])
u2 = stats.t.cdf(z[:, 1], nu, mu[1], svec[1])
u = np.r_[u1, u2]  # grade scenarios
# -

# ## Generate joint scenarios

x1 = stats.norm.ppf(u1, mu[0], svec[0])
x2 = stats.norm.ppf(u2, mu[1], svec[1])
x = np.r_[x1, x2]  # joint scenarios


# ## Compute pdf of joint distribution

f_u = np.zeros((n_grid, n_grid))
f_x = np.zeros((n_grid, n_grid))
for n in range(n_grid):
    for m in range(n_grid):
        u = np.r_[grid[n], grid[m]].reshape(-1, 1)  # evaluation points
        f_u[n, m] = t_cop_pdf(u, nu, mu, sigma2)  # pdf of copula
        f_x[n, m] = f_u[n, m]*np.prod(stats.norm.pdf(stats.norm.ppf(u, mu, svec), mu, svec))  # pdf of joint distribution

# ## Plots

# +
xx_1 = stats.norm.ppf(grid, mu[0], svec[0])
xx_2 = stats.norm.ppf(grid, mu[1], svec[1])
[x_1, x_2] = np.meshgrid(xx_1, xx_2)

plt.style.use('arpm')
x_color = [4/255, 63/255, 114/255]

# set figure specification
f = plt.figure(1, figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

ax1 = plt.axes([0.3, 0.53, 0.35, 0.35], projection='3d')
ax1.plot_surface(x_1, x_2, f_x.T, facecolor='k', edgecolor=x_color)
ax1.view_init(30, -120)
plt.xlabel('$X_1$', labelpad=7)
plt.ylabel('$X_2$', labelpad=7)
ax1.set_zlabel('Joint pdf')

ax3 = plt.axes([0.408, 0.12, 0.1623, 0.35])
ax3.scatter(x1, x2, s=10, color=x_color, marker='*')
ax3.tick_params(axis='x', colors='None')
ax3.tick_params(axis='y', colors='None')
plt.xlabel('$X_1$', labelpad=-5)
plt.ylabel('$X_2$', labelpad=-5)

ax4 = plt.axes([0.408, 0.001, 0.1623, 0.07])
plt.hist(np.sort(x1), bins=int(10*np.log(j_)), color=x_color, density=True, bottom=0)
plt.gca().invert_yaxis()

ax5 = plt.axes([0.32, 0.12, 0.05, 0.35])
plt.hist(np.sort(x2), bins=int(10*np.log(j_)), density=True,
         color=x_color, bottom=0, orientation='horizontal')
plt.gca().invert_xaxis()

add_logo(f, axis=ax1, location=4, set_fig_size=False)
