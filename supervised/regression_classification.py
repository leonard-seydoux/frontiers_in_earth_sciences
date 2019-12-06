#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Linear regression illustration.

Author: Leonard Seydoux
Email: leonard.seydoux@univ-grenoble-alpes.fr
Date: Nov. 2019
"""

import numpy as np

from matplotlib import pyplot as plt


with plt.xkcd():
    # Generate data
    n = 50
    np.random.seed(1)
    x = np.random.randn(n)
    a = 1
    b = -.5
    y = a * x + b + .3 * np.random.randn(n)

    # Put into good shape
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Show data
    xt = np.linspace(-10, 10)
    yt = a * xt + b
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(x, y, '.')
    ax[0].plot(xt, yt, 'k--')
    ax[0].set_xlim([-3, 3])
    ax[0].set_ylim([-3, 3])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlabel(r'Input data')
    ax[0].set_ylabel(r'Continuous label')
    ax[0].set_title('Regression')

    x = np.random.randn(n) + 2
    y = np.random.randn(n) + 2
    ax[1].plot(x, y, '.', label='Label A')
    x = np.random.randn(n) - 2
    y = np.random.randn(n) - 2
    ax[1].plot(x, y, '.', label='Label B')

    ax[1].plot(xt, -yt, 'k--', label='Boundary')
    ax[1].set_xlim([-8, 8])
    ax[1].set_ylim([-8, 8])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlabel(r'Feature #1')
    ax[1].set_ylabel(r'Feature #2')
    ax[1].set_title('Classification')
    ax[1].legend()

    fig.patch.set_alpha(0)
    fig.savefig('fig_1.png', dpi=300)
