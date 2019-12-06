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
x = np.linspace(-3, 3)
y = x ** 2

# Show data
fig, ax = plt.subplots(1, 3, figsize=(7, 3))
ax[0].plot(x, y)
fig.savefig('fig_1.png', transparent=True)
