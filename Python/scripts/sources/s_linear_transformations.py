#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# # s_linear_transformations [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_linear_transformations&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_linear_transformations).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_linear_transformations-parameters)

# +
# linear transformations
a = np.array([[2, 0.5],
              [1, 1]])
b = np.array([[1, 0.5],
              [2, 1]])

# vectors
v = np.array([3.0, 1.5])
u = np.array([2.5, -2.0])

# scalar
c = 0.25
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_linear_transformations-implementation-step01): Apply transformations

Av = a@v
Au = a@u
Bv = b@v
Bu = b@u

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_linear_transformations-implementation-step02): Test linearity conditions

# +
# test transformation A
A_u_plus_v = a@(u+v)
A_cv = a@(c*v)
print('A preserves the sum:', np.all(A_u_plus_v==(Au + Av)))
print('A preserves scaling:', np.all(A_cv==c*(Av)))

# test transformation B
B_u_plus_v = b@(u+v)
B_cv = b@(c*v)
print('B preserves the sum:', np.all(B_u_plus_v==(Bu + Bv)))
print('B preserves scaling:', np.all(B_cv==c*(Bv)))
# -

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_linear_transformations-implementation-step03): Compose linear transformations

A_circ_B = a@b

# ## Plots

# +
plt.style.use('arpm')

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0)

# transformation A
ax1.set_title('Invertible transformation', fontsize=20, fontweight='bold')

# v
ax1.arrow(0, 0, v[0], v[1], color='C0',
         length_includes_head=True,
         head_width=0.2)
ax1.text(v[0], v[1], r'$\mathbf{v}$', fontsize=14)
# u
ax1.arrow(0, 0, u[0], u[1], color='C0',
         length_includes_head=True,
         head_width=0.2)
ax1.text(u[0], u[1], r'$\mathbf{u}$', fontsize=14)
# parallelotope
ax1.plot([v[0], (v+u)[0]], [v[1], (v+u)[1]],
         color='C0', linestyle='--')
ax1.plot([u[0], (v+u)[0]], [u[1], (v+u)[1]],
         color='C0', linestyle='--')

# Av
ax1.arrow(0, 0, Av[0], Av[1], color='darkorange',
         length_includes_head=True,
         head_width=0.2)
ax1.text(Av[0], Av[1],
         r'$\mathcal{A}\mathbf{v}$', fontsize=14)
# Au
ax1.arrow(0, 0, Au[0], Au[1], color='darkorange',
         length_includes_head=True,
         head_width=0.2)
ax1.text(Au[0], Au[1], r'$\mathcal{A}\mathbf{u}$',
         fontsize=14)
# parallelotope
ax1.plot([Av[0], A_u_plus_v[0]], [Av[1], A_u_plus_v[1]],
         color='darkorange', linestyle='--')
ax1.plot([Au[0], A_u_plus_v[0]], [Au[1], A_u_plus_v[1]],
         color='darkorange', linestyle='--')

limval = max(np.abs(B_u_plus_v))
ax1.set_xlim([-limval*1.2, limval*1.2])
ax1.set_ylim([-limval*1.2, limval*1.2])
ax1.axhline(color='black')
ax1.axvline(color='black')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# transformation B
ax2.set_title('Non-invertible transformation', fontsize=20, fontweight='bold')
# v
ax2.arrow(0, 0, v[0], v[1], color='C0',
         length_includes_head=True,
         head_width=0.2)
ax2.text(v[0], v[1], r'$\mathbf{v}$', fontsize=14)
# u
ax2.arrow(0, 0, u[0], u[1], color='C0',
         length_includes_head=True,
         head_width=0.2)
ax2.text(u[0], u[1], r'$\mathbf{u}$', fontsize=14)
# parallelotope
ax2.plot([v[0], (v+u)[0]], [v[1], (v+u)[1]],
         color='C0', linestyle='--')
ax2.plot([u[0], (v+u)[0]], [u[1], (v+u)[1]],
         color='C0', linestyle='--')

# Bv
ax2.arrow(0, 0, Bv[0], Bv[1], color='darkorange',
         length_includes_head=True,
         head_width=0.2)
ax2.text(Bv[0], Bv[1],
         r'$\mathcal{B}\mathbf{v}$', fontsize=14)
# Bu
ax2.arrow(0, 0, Bu[0], Bu[1], color='darkorange',
         length_includes_head=True,
         head_width=0.2)
ax2.text(Bu[0], Bu[1], r'$\mathcal{B}\mathbf{u}$',
         fontsize=14)
# parallelotope
ax2.plot([Bv[0], B_u_plus_v[0]], [Bv[1], B_u_plus_v[1]],
         color='darkorange', linestyle='--')
ax2.plot([Bu[0], B_u_plus_v[0]], [Bu[1], B_u_plus_v[1]],
         color='darkorange', linestyle='--')

ax2.set_xlim([-limval*1.2, limval*1.2])
ax2.set_ylim([-limval*1.2, limval*1.2])
ax2.axhline(color='black')
ax2.axvline(color='black')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

add_logo(fig, set_fig_size=False)
plt.tight_layout()
