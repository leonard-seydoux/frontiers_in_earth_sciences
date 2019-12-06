#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Linear regression illustration.

Author: Leonard Seydoux
Email: leonard.seydoux@univ-grenoble-alpes.fr
Date: Nov. 2019
"""

import numpy as np

from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Generate data
n = 15
np.random.seed(1)
x = np.random.randn(n)
a = .7
b = -.5
c = -1
d = -.5
y = a * x ** 3 + b * x ** 2 + c * x + d + 1. * np.random.randn(n)

# Put into good shape
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

model = KNeighborsRegressor(n_neighbors=1)
model.fit(x, y)

# Show data
xt = np.linspace(-3, 3).reshape(-1, 1)
yt = a * xt ** 3 + + b * xt ** 2 + c * xt + d
fig, ax = plt.subplots(1, figsize=(4, 3))
ax.plot(x, y, '.', label='Data point', mec='k')
ax.plot(xt, yt, '--', label='Ground truth')
ax.plot(xt, model.predict(xt), '-', label='Overfitting model')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'Input data ($x$)')
ax.set_ylabel(r'Label ($y$)')
ax.legend()
fig.savefig('fig_7.png', transparent=True)
