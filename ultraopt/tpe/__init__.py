#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-15
# @Contact    : qichun.tang@bupt.edu.cn

from collections import namedtuple

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity


class SampleDisign():
    def __init__(self, ratio=0.0, n_samples=0, is_random=False, bw_factor=1.2):
        self.bw_factor = bw_factor
        self.is_random = is_random
        self.n_samples = n_samples
        self.ratio = ratio


def top15_gamma(x: int) -> int:
    return max(1, round(0.15 * x))


def optuna_gamma(x: int) -> int:
    return min(int(np.ceil(0.1 * x)), 25)


def hyperopt_gamma(x: int) -> int:
    return min(int(np.ceil(0.25 * np.sqrt(x))), 25)


def estimate_bw(data, bw_method="scott", cv_times=100):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.estimate_bandwidth.html
    ndata = data.shape[0]
    if bw_method == 'scott':
        bandwidth = ndata ** (-1 / 5) * np.std(data, ddof=1)
        bandwidth = np.clip(bandwidth, 0.01, None)
    elif bw_method == 'silverman':
        bandwidth = (ndata * 3 / 4) ** (-1 / 5) * np.std(data, ddof=1)
        bandwidth = np.clip(bandwidth, 0.01, None)
    elif bw_method == 'cv':
        if ndata <= 3:
            return estimate_bw(data)
        bandwidths = np.std(data, ddof=1) ** np.linspace(-1, 1, cv_times)
        bandwidths = np.clip(bandwidths, 0.01, None)
        grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths},
                            cv=KFold(n_splits=3, shuffle=True, random_state=0))
        grid.fit(data)
        bandwidth = grid.best_params_['bandwidth']
    elif np.isscalar(bw_method):
        bandwidth = bw_method
    else:
        raise ValueError("Unrecognized input for bw_method.")
    return bandwidth
