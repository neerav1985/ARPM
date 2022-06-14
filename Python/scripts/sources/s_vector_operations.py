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

# # s_vector_operations [<img src="https://www.arpm.co/lab/icons/icon_permalink.png" width=30 height=30 style="display: inline;">](https://www.arpm.co/lab/redirect.php?code=s_vector_operations&codeLang=Python)
# For details, see [here](https://www.arpm.co/lab/redirect.php?permalink=s_vector_operations).

# +
import numpy as np
import matplotlib.pyplot as plt

from arpym.tools.logo import add_logo
# -

# ## [Input parameters](https://www.arpm.co/lab/redirect.php?permalink=s_vector_operations-parameters)

# +
# vectors
v = np.array([3.0, 1.5])
u = np.array([0.5, 2.0])

# scalars
c_1 = 2  # stretch
c_2 = 0.6  # contract
c_3 = -1  # reflect
# -

# ## [Step 1](https://www.arpm.co/lab/redirect.php?permalink=s_vector_operations-implementation-step01): Scalar multiplication

v_stretch = c_1*v
v_contract = c_2*v
v_reflect = c_3*v

# ## [Step 2](https://www.arpm.co/lab/redirect.php?permalink=s_vector_operations-implementation-step02): Addition

w = u + v

# ## [Step 3](https://www.arpm.co/lab/redirect.php?permalink=s_vector_operations-implementation-step03): Subtraction

u_tilde = w - v

# ## Plots

# +
plt.style.use('arpm')

fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(1280.0/72.0, 720.0/72.0), dpi = 72.0)

# vector addition
ax1.set_title('Vector addition', fontsize=20, fontweight='bold')
# v
ax1.arrow(0, 0, v[0], v[1], color='C0',
         length_includes_head=True,
         head_width=0.1)
ax1.text(v[0], v[1], r'$\mathbf{v}$', fontsize=14)

# u
ax1.arrow(0, 0, u[0], u[1], color='C0',
         length_includes_head=True,
         head_width=0.1)
ax1.text(u[0], u[1], r'$\mathbf{u}$', fontsize=14)

# w=u+v
ax1.arrow(0, 0, w[0], w[1], color='darkorange',
         length_includes_head=True,
         head_width=0.1)
ax1.text(w[0], w[1],
         r'$\mathbf{w}=\mathbf{u}+\mathbf{v}$', fontsize=14)

ax1.plot([v[0], w[0]], [v[1], w[1]],
         color='lightgrey', ls='--', lw=2,
         alpha=0.6)
ax1.plot([u[0], w[0]], [u[1], w[1]],
         color='lightgrey', ls='--', lw=2,
         alpha=0.6)

limval = max(np.abs(w))
ax1.set_xlim([-limval*1.4, limval*1.4])
ax1.set_ylim([-limval*1.4, limval*1.4])
ax1.axhline(color='black')
ax1.axvline(color='black')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# vector subtraction
ax2.set_title('Vector subtraction', fontsize=20, fontweight='bold')
# v
ax2.arrow(0, 0, v[0], v[1], color='C0',
         length_includes_head=True,
         head_width=0.1)
ax2.text(v[0], v[1], r'$\mathbf{v}$', fontsize=14)

# w
ax2.arrow(0, 0, w[0], w[1], color='C0',
         length_includes_head=True,
         head_width=0.1)
ax2.text(w[0], w[1], r'$\mathbf{w}$', fontsize=14)

# u=w-v
ax2.arrow(0, 0, u_tilde[0], u_tilde[1],
          color='darkorange',
          length_includes_head=True,
          head_width=0.1)
ax2.text(u_tilde[0], u_tilde[1],
         r'$\mathbf{u}=\mathbf{w}-\mathbf{v}$', fontsize=14)

ax2.plot([0, -v[0]], [0, -v[1]],
         color='lightgrey', ls='--', lw=2,
         alpha=0.6)
ax2.text(-v[0], -v[1], r'$-\mathbf{v}$', fontsize=14,
         horizontalalignment='right')
ax2.plot([-v[0], u_tilde[0]], [-v[1], u_tilde[1]],
         color='lightgrey', ls='--', lw=2,
         alpha=0.6)
ax2.plot([w[0], u_tilde[0]], [w[1], u_tilde[1]],
         color='lightgrey', ls='--', lw=2,
         alpha=0.6)

ax2.set_xlim([-limval*1.4, limval*1.4])
ax2.set_ylim([-limval*1.4, limval*1.4])
ax2.axhline(color='black')
ax2.axvline(color='black')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

add_logo(fig1, set_fig_size=False)
plt.tight_layout()

fig2, [ax3, ax4, ax5] = plt.subplots(3, 1, figsize=(720.0/72.0, 960.0/72.0), dpi = 72.0)

# vector scalar multiplication - stretch
ax3.set_title('Stretch', fontsize=20, fontweight='bold')
# v
ax3.arrow(0, 0, v[0], v[1], color='C0',
         length_includes_head=True,
         head_width=0.2, head_length=0.1)
ax3.text(v[0], v[1], r'$\mathbf{v}$',
         horizontalalignment='right', fontsize=14)

# c_1*v
ax3.arrow(v[0], v[1], v_stretch[0]-v[0], v_stretch[1]-v[1],
          color='darkorange',
          length_includes_head=True,
          head_width=0.2, head_length=0.1)
ax3.text(v_stretch[0], v_stretch[1], str(c_1)+r'$\mathbf{v}$',
         horizontalalignment='right', fontsize=14)

ax3.set_xlim([-np.abs(c_1*v[0])*1.4, np.abs(c_1*v[0])*1.4])
ax3.set_ylim([-np.abs(c_1*v[1])*1.4, np.abs(c_1*v[1])*1.4])
ax3.set_aspect('equal')
ax3.axhline(color='black')
ax3.axvline(color='black')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# vector scalar multiplication - contract
ax4.set_title('Contract', fontsize=20, fontweight='bold')
# v
ax4.arrow(v_contract[0], v_contract[1], v[0]-v_contract[0], v[1]-v_contract[1],
          color='C0', length_includes_head=True,
          head_width=0.2, head_length=0.1)
ax4.text(v[0], v[1], r'$\mathbf{v}$', fontsize=14)

# c_2*v
ax4.arrow(0, 0, v_contract[0], v_contract[1],
          color='darkorange',
          length_includes_head=True,
          head_width=0.2, head_length=0.1)
ax4.text(v_contract[0], v_contract[1], str(c_2)+r'$\mathbf{v}$',
         horizontalalignment='right', fontsize=14)

ax4.set_xlim([-np.abs(c_1*v[0])*1.4, np.abs(c_1*v[0])*1.4])
ax4.set_ylim([-np.abs(c_1*v[1])*1.4, np.abs(c_1*v[1])*1.4])
ax4.set_aspect('equal')
ax4.axhline(color='black')
ax4.axvline(color='black')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# vector scalar multiplication - reflect
ax5.set_title('Reflect', fontsize=20, fontweight='bold')
# v
ax5.arrow(0, 0, v[0], v[1],
          color='C0', length_includes_head=True,
          head_width=0.2, head_length=0.1)
ax5.text(v[0], v[1], r'$\mathbf{v}$', fontsize=14)

# c_3*v
ax5.arrow(0, 0, v_reflect[0], v_reflect[1],
          color='darkorange',
          length_includes_head=True,
          head_width=0.2, head_length=0.1)
if c_3 == -1:
    c3_str = r'$-$'
else:
    c3_str = str(c_3)
    
ax5.text(v_reflect[0], v_reflect[1], c3_str+r'$\mathbf{v}$',
         horizontalalignment='right', fontsize=14)

ax5.set_xlim([-np.abs(c_1*v[0])*1.4, np.abs(c_1*v[0])*1.4])
ax5.set_ylim([-np.abs(c_1*v[1])*1.4, np.abs(c_1*v[1])*1.4])
ax5.set_aspect('equal')
ax5.axhline(color='black')
ax5.axvline(color='black')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

add_logo(fig2, set_fig_size=False)
plt.tight_layout()
