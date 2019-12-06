#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Linear regression illustration.

Author: Leonard Seydoux
Email: leonard.seydoux@univ-grenoble-alpes.fr
Date: Nov. 2019
"""

import numpy as np

from matplotlib import pyplot as plt


# Show data
n = 200
np.random.seed(1)
x = np.random.randn(n)
a = .7
b = -.5
c = -1
d = -.5
y = a * x ** 3 + b * x ** 2 + c * x + d + .5 * np.random.randn(n)

# Put into good shape
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Show data
fig, ax = plt.subplots(2, figsize=(4, 5))
ax[0].plot(x[:n // 2], y[:n // 2], '.', label='Training data', mec='k', c='C1')
ax[0].plot(x[n // 2:], y[n // 2:], '.', label='Testing data', mec='k', c='C4')
ax[0].set_xlim([-3, 3])
ax[0].set_ylim([-3, 3])
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlabel(r'Input data ($x$)')
ax[0].set_ylabel(r'Label ($y$)')
ax[0].legend()

# Generate data
n = 200
x = np.linspace(-3, 5, n)
y = np.exp(-x)
z = np.exp(-2 * x)
ax[1].plot(x, y - 1, label='Training loss', mec='k', c='C1')
ax[1].plot(x - 2, z, label='Testing loss', mec='k', c='C4')
ax[1].plot(0, 0, 'o', mfc=(0, 0, 0, 0), mec='k', label='Overfitting point')
ax[1].set_xlim([-2.5, 3])
ax[1].set_ylim([-3, 15])
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xlabel(r'Optimization stage')
ax[1].set_ylabel(r'Loss')
ax[1].legend()
fig.savefig('fig_9.png', transparent=True)
