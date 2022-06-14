#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # s_linclass_svm [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_linclass_svm&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_linclass_svm).

# +
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.svm import LinearSVC
from scipy.special import expit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_linclass_svm-parameters)

lam = 1  # regularization parameter

# ## [Step 0](https://www.arpm.co/lab/redirect.php?permalink=s_linclass_svm-implementation-step00): Load data

data = pd.read_csv('~/databases/temporary-databases/db_ml_variables.csv')
j_ = int(data['j_in_sample'][0])
x = data['x'].values[:j_]  # scenarios of outputs and inputs 
z = data['z'].values.reshape(j_, 2)


# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_linclass_svm-implementation-step01): Define target variables

phi_z = np.c_[z[:, 0], z[:, 0]*z[:, 1]]  # feature space basis
x = np.heaviside(x, 1)  # scenarios of binary output

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_linclass_svm-implementation-step02): Predict simulations via SVM and compute expected hinge loss and misclassification error

svm_clf = LinearSVC(C=lam)  
alpha_beta = svm_clf.fit(phi_z, x)  # fit support vector machines
x_bar = alpha_beta.predict(phi_z)  # predictions
hinge_loss = np.sum((1/j_)*np.maximum(np.zeros(j_), 1-(2*x-1)*alpha_beta.decision_function(phi_z))) + lam*np.linalg.norm(alpha_beta.coef_)**2  # expected regularized hinge loss
error = np.linalg.norm(x-x_bar,ord=0)/j_  # misclassification error

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_linclass_svm-implementation-step03): Compute scores, AUC, confusion matrix

s_0 = alpha_beta.decision_function(phi_z[x == 0])  # conditional scores
s_1 = alpha_beta.decision_function(phi_z[x == 1])
fpr, tpr, _ = roc_curve(x, expit(alpha_beta.decision_function(phi_z)))  # false and true positive rates
auc = auc(fpr, tpr)  # AUC
p_xxbar = confusion_matrix(x, alpha_beta.predict(phi_z))/np.sum(confusion_matrix(x, alpha_beta.predict(phi_z)), axis=1)  # confusion matrix


# ## Plots

# +
plt.style.use('arpm')
fig = plt.figure(figsize=(1280.0/72.0, 720.0/72.0), dpi=72.0)

# parameters
orange = [255/255, 153/255, 0/255]
green = [60/255, 149/255, 145/255]
grtrue = [0, 0.4, 0]
grhatch = [0, 0.1, 0]
grfalse = [0.2, 0.6, 0]
red = [227/255, 66/255, 52/255]
blue = [13/255, 94/255, 148/255]
n_classes = 2
plot_colors = ["red", "blue"]
plot_step = 0.02
phi_zz1, phi_zz2 = np.meshgrid(np.arange(-2, 2, plot_step),
                               np.arange(-2, 2, plot_step))
idxx0 = np.where(np.abs(phi_z[:, 0]) <= 2)[0]
idxx1 = np.where(np.abs(phi_z[:, 1]) <= 2)[0]
idxx = np.intersect1d(idxx0, idxx1)

# ROC curve
ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
ax1.plot(fpr, tpr, color='b')
ax1.set_xlabel('fpr')
ax1.set_ylabel('tpr')
ax1.text(0.2, 0.05, 'AUC = %.2f' % auc)
plt.text(0.2, 0.14, '$L_0$ error = %.2f' % error)
plt.text(0.2, 0.23, 'Hinge loss = %.2f' % hinge_loss)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.axis('square')
ax1.set_title('ROC curve', fontweight='bold')

add_logo(fig, axis=ax1, location=5, size_frac_x=1/8)

# 3D plot
ax2 = plt.subplot2grid((4, 4), (2, 0), colspan=2, rowspan=2, projection='3d')
x_plot = alpha_beta.predict(np.c_[phi_zz1.ravel(), phi_zz2.ravel()])
x_plot = x_plot.reshape(phi_zz1.shape)
ax2.plot_surface(phi_zz1, phi_zz2, x_plot,
                 cmap=plt.cm.RdYlBu, alpha=0.7)
# scatter plot
for i, color in zip(range(n_classes), plot_colors):
    idx = np.intersect1d(np.where(x == i), idxx)
    ax2.scatter3D(phi_z[idx, 0], phi_z[idx, 1], x[idx], c=color,
                  label=['0', '1'][i],
                  cmap=plt.cm.RdYlBu, edgecolor='black', s=15, alpha=0.5)
ax2.view_init(30, -90)
ax2.set_xlabel('$\phi_1(Z)$')
ax2.set_ylabel('$\phi_2(Z)$')
ax2.set_zlabel('$X$')
ax2.set_title('Surface fitted with SVM', fontweight='bold')
ax2.set_xlim([-2, 2])
ax2.set_ylim([-2, 2])
# ax.legend()

# regions plot
ax3 = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=2)
xx_perc = alpha_beta.predict(np.c_[phi_zz1.ravel(), phi_zz2.ravel()])
# Put the result into a color plot
xx_perc = xx_perc.reshape(phi_zz1.shape)
ax3.contourf(phi_zz1, phi_zz2, xx_perc, cmap=plt.cm.RdYlBu, alpha=0.5)

# scatter plot
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(x == i)
    ax3.scatter(phi_z[idx, 0], phi_z[idx, 1], c=color,
                label=['0', '1'][i],
                cmap=plt.cm.RdYlBu, edgecolor='black', s=15, alpha=0.7)
ax3.set_xlabel('$\phi_1(Z)$')
ax3.set_ylabel('$\phi_2(Z)$')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
ax3.set_title('Decision regions', fontweight='bold')

# scores
ax4 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=1)
ax4.hist(s_0, 80, density=True, alpha=0.7, color=red)
ax4.hist(s_1, 80, density=True, alpha=0.7, color=blue)
yymax = ax4.get_ylim()[1]
ax4.plot([0, 0], [0, yymax], 'k--')
ax4.legend(['S | 0', 'S | 1'])
ax4.set_title('Scores distribution', fontweight='bold')

# rates
ax5 = plt.subplot2grid((4, 4), (1, 2), colspan=2, rowspan=1)
ax5.fill([0.15, 0.15, 0.35, 0.35],
         [0, p_xxbar[0, 1], p_xxbar[0, 1], 0], facecolor=grfalse,
         edgecolor=grhatch, hatch='//', alpha=0.7)
ax5.fill([0.15, 0.15, 0.35, 0.35],
         [p_xxbar[0, 1], 1, 1, p_xxbar[0, 1]],
         facecolor=grfalse, alpha=0.7)
ax5.fill([0.45, 0.45, 0.65, 0.65],
         [0, p_xxbar[1, 1], p_xxbar[1, 1], 0], facecolor=grtrue,
         alpha=0.7)
ax5.fill([0.45, 0.45, 0.65, 0.65],
         [p_xxbar[1, 1], 1, 1, p_xxbar[1, 1]], facecolor=grtrue,
         edgecolor=grhatch, hatch='\\\\', alpha=0.7)
ax5.set_ylim([0, 1])
ax5.legend(['fpr', 'tnr', 'tpr', 'fnr'], bbox_to_anchor=(0.001, -0.07, 1., .1),
           facecolor='white',
           loc=1, ncol=5, mode="expand")
ax5.set_title('Confusion matrix', fontweight='bold')
ax5.set_xticks([])
ax5.grid(False)

plt.tight_layout()
