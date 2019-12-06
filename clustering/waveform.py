#!/usr/bin/env python -W ignore::UserWarning
# -*- coding: utf-8 -*-
"""Figure 1: seismic data."""

import warnings
import numpy as np
import covnet as cn
import matplotlib.pyplot as plt

PATH_DATA = \
    '/Users/seydoux/Documents/Work/data/DK/continuous31/processed/DK.NUUG..HHZ'


# Seismogram
# ----------
w = list()

for i in [2, 8]:

    # Read
    stream = cn.data.read(PATH_DATA)
    stream.filter(type='bandpass', freqmin=.1, freqmax=12.)
    stream.cut('2017-06-17 23:40:00', '2017-06-17 23:42')

    # Show temporal
    fig, ax = plt.subplots(1, figsize=(3, 1))
    d = stream[0].data - stream[0].data.mean()
    d /= np.abs(d).max()
    w.append(d[i::10])
    ax.plot(stream.times[i::10], d[i::10], lw=.2, c='k', marker='.', ms=2)
    ax.set_axis_off()
    ax.set_xlim(stream.times[[0, -1]])
    ax.set_ylim([-1, 1])

    # Save
    # ----

    ax.figure.patch.set_alpha(0)
    warnings.filterwarnings("ignore")
    ax.figure.savefig('waveform_{}.png'.format(i), transparent=True)

print(np.corrcoef(w[0], w[1]))
