#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate entropy and save."""

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, mixture
from sklearn import datasets as dat
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

# Generate datasets
np.random.seed(0)
n_samples = 500
noisy_circles = dat.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = dat.make_moons(n_samples=n_samples, noise=.05)
blobs = dat.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = dat.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = dat.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5],
                        random_state=random_state)

# Show
plt.figure(figsize=(9, 3))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.1,
                    hspace=.1)

plot_num = 1

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3}

dat = [
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied, {'eps': .18, 'n_neighbors': 2}),
    (aniso, {'eps': .15, 'n_neighbors': 2})]

for i_dataset, (dataset, algo_params) in enumerate(dat):

    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=params['eps'])
    affinity_propagation = cluster.AffinityPropagation(
        damping=params['damping'], preference=params['preference'])
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (
        ('k means', two_means),
        ('affinity\npropagation', affinity_propagation),
        ('mean\nshift', ms),
        ('spectral', spectral),
        ('ward\nhierachical', ward),
        ('agglomerative', average_linkage),
        ('DB scan', dbscan),
        ('Gaussian\nmixture', gmm)
    )

    for name, algorithm in clustering_algorithms:
        print(name)
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(dat), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name)

        colors = np.array(list(islice(cycle([
            '#199fff', '#f4a742', '#7b47c1', '#ffd018', '#9ac44c']),
            int(max(y_pred) + 1))))
        colors = np.append(colors, ["#bbbbbb"])

        plt.scatter(X[:, 0], X[:, 1], s=7, color=colors[y_pred],
                    linewidth=.2, edgecolor='k')
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 3.)
        plt.xticks(())
        plt.yticks(())
        plt.gca().set_axis_off()
        plot_num += 1

plt.gcf().patch.set_alpha(0)
plt.savefig('illustration.png', dpi=300, bbox_inches='tight')
