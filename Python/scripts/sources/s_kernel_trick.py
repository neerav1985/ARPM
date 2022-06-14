# # s_kernel_trick [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_kernel_trick&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_kernel_trick).

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_kernel_trick-parameters)

phi2 = lambda z, v, gamma : np.exp(-gamma*np.linalg.norm(z-v))
gamma = 1

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_kernel_trick-implementation-step00): Load input and target scenarios

data = pd.read_csv('~/databases/temporary-databases/db_ml_variables.csv')
j_ = int(data['j_in_sample'][0])
z = data['z'].values.reshape(j_, 2)
j_ = 150  # reduce dimensionality of the problem to speed up the computations
z = z[:j_, :]
x = data['x'].values[:j_]

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_kernel_trick-implementation-step01): Compute Gram matrix

phi2_gram = np.zeros((j_, j_))
for i in range(j_):
    for j in range(j_):
        phi2_gram[i, j] = phi2(z[i], z[j], gamma)
inv_phi2_gram = np.linalg.inv(phi2_gram)

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_kernel_trick-implementation-step02): Kernel trick

premult = x.T@inv_phi2_gram
chi = lambda zz : np.array([premult@np.array([phi2(z[j], zz[i], gamma) for j in range(z.shape[0])]) for i in range(zz.shape[0])])

# ## Plots

# +
plt.style.use('arpm')

idxx0 = np.where(np.abs(z[:, 0]) <= 2)[0]
idxx1 = np.where(np.abs(z[:, 1]) <= 2)[0]
idxx = np.intersect1d(idxx0, idxx1)
lightblue = [0.2, 0.6, 1]
lightgreen = [0.6, 0.8, 0]

# Auxiliary functions

def muf(z1, z2):
    return z1 - np.tanh(10*z1*z2)

def sigf(z1, z2):
    return np.sqrt(np.minimum(z1**2, 1/(10*np.pi)))

plt.figure()
mydpi = 72.0
fig = plt.figure(figsize=(1280.0/mydpi, 720.0/mydpi), dpi=mydpi)

# Parameters
n_classes = 2
plot_colors = "rb"
plot_step = 0.08

z_1_min = z[:, 0].min()
z_1_max = z[:, 0].max()
z_2_min = z[:, 1].min()
z_2_max = z[:, 1].max()
zz1, zz2 = np.meshgrid(np.arange(z_1_min, z_1_max, plot_step),
                       np.arange(z_2_min, z_2_max, plot_step))


# Conditional expectation surface
ax2 = plt.subplot2grid((1, 2), (0, 0), projection='3d')
zz1, zz2 = np.meshgrid(np.arange(-2, 2, plot_step), np.arange(-2, 2, plot_step))
ax2.plot_surface(zz1, zz2, muf(zz1, zz2), color=lightblue, alpha=0.7,
                 label='$\mu(z_1, z_2)$')

ax2.scatter3D(z[idxx, 0], z[idxx, 1],
              x[idxx], s=10, color=lightblue, alpha=1,
              label='$(Z_1, Z_2, X)$')
ax2.set_xlabel('$Z_1$')
ax2.set_ylabel('$Z_2$')
ax2.set_zlabel('$X$')
ax2.set_title('Conditional expectation surface', fontweight='bold')
ax2.set_xlim([-2, 2])
ax2.set_ylim([-2, 2])
ax2.set_zlim([-3, 3])
# ax.legend()

# Fitted surface
ax3 = plt.subplot2grid((1, 2), (0, 1), projection='3d')
x_plot = chi(np.c_[zz1.ravel(), zz2.ravel()])
x_plot = x_plot.reshape(zz1.shape)
ax3.plot_surface(zz1, zz2, x_plot, alpha=0.5, color=lightgreen)
ax3.scatter3D(z[idxx, 0], z[idxx, 1],
              chi(z[idxx, :]), s=10,
              alpha=1, color=lightgreen)
ax3.set_xlabel('$Z_1$')
ax3.set_ylabel('$Z_2$')
ax3.set_zlabel('$\overline{X}$')
plt.title('Fitted surface ', fontweight='bold')
ax3.set_xlim([-2, 2])
ax3.set_ylim([-2, 2])
#ax3.set_zlim([-3, 3])

add_logo(fig, size_frac_x=1/8)
plt.tight_layout()
