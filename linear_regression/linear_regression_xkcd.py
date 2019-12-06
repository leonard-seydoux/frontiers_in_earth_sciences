#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Linear regression illustration.

Author: Leonard Seydoux
Email: leonard.seydoux@univ-grenoble-alpes.fr
Date: Nov. 2019
"""

import numpy as np

from matplotlib import pyplot as plt
from sklearn import linear_model


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
    xt = np.linspace(-3, 3)
    yt = a * xt + b
    fig, ax = plt.subplots(1, figsize=(4, 3))
    ax.plot(x, y, '.', label='Data point', mec='k')
    ax.plot(xt, yt, '--', label='Ground truth')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r'Input data ($x$)')
    ax.set_ylabel(r'Label ($y$)')
    ax.legend()
    fig.savefig('xkcd_1.png')

    # Show data
    fig, ax = plt.subplots(1, figsize=(4, 3))
    ax.plot(x, y, '.', label='Data point', mec='k')
    ax.plot(xt, yt, label='Ground truth')
    first = True
    for c, d in 2 * (np.random.randn(2, 10).T - .5):
        yt = c * xt + d
        if first is True:
            ax.plot(xt, yt, label='Brute-force tests', c='C2', alpha=.4)
            first = False
        else:
            ax.plot(xt, yt, c='C2', alpha=.4)
    ax.axhline(0, c='k', ls='--', lw=.5)
    ax.axvline(0, c='k', ls='--', lw=.5)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r'Input data ($x$)')
    ax.set_ylabel(r'Label ($y$)')
    ax.legend(loc=2)
    fig.savefig('xkcd_2.png')

    # Show data
    xt = np.linspace(-3, 3)
    yt = a * xt + b
    yw = (a * 1.5) * xt + (1.1 * b)
    fig, ax = plt.subplots(1, figsize=(4, 3))
    ax.plot(x, y, '.', label='Data point', mec='k')
    ax.plot(xt, yt, '--', label='Ground truth')
    ax.plot(xt, yw, '-', label='Tested model')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r'Input data ($x$)')
    ax.set_ylabel(r'Label ($y$)')
    ax.legend(loc=2)
    fig.savefig('xkcd_3.png')

    # Perform regression
    fig, ax = plt.subplots(1, figsize=(4, 3))
    model = linear_model.LinearRegression()
    model.fit(x, y)
    xp = np.linspace(-3, 3).reshape(-1, 1)
    yp = model.predict(xp)
    a = model.coef_[0][0]
    b = model.intercept_[0]
    e = model.score(x, y)
    ax.plot(x, y, '.', label='Data point', mec='k')
    ax.plot(xt, yt, '--', label='Ground truth')
    ax.plot(xp, yp, label=r'Learned model ($\epsilon$ = {:.2f})'.format(1 - e))
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r'Input data ($x$)')
    ax.set_ylabel(r'Label ($y$)')
    ax.legend(loc=2)
    fig.savefig('xkcd_4.png')

    # Perform regression
    fig, ax = plt.subplots(1, figsize=(4, 3))
    model = linear_model.LinearRegression()
    model.fit(x, y)
    xp = np.linspace(-3, 3).reshape(-1, 1)
    yp = model.predict(xp)
    a = model.coef_[0][0]
    b = model.intercept_[0]
    e = model.score(x, y)
    ax.plot(x[:n // 2], y[:n // 2], '.', label='Training data', mec='k')
    ax.plot(x[n // 2:], y[n // 2:], '.', label='Testing data', mec='k', c='C3')
    # ax.plot(xt, yt, '--', label='Ground truth')
    # ax.plot(xp, yp, label=r'Learned model ($\epsilon$ = {:.2f})'.format(1 - e))
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r'Input data ($x$)')
    ax.set_ylabel(r'Label ($y$)')
    ax.legend(loc=2)
    fig.savefig('xkcd_5.png')
