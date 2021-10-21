#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-20
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_random_state


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


class BaseKernelDensity(BaseEstimator):
    def fit(self, X):
        raise NotImplementedError

    def score_samples(self, X):
        '''log_density'''
        raise NotImplementedError

    def score(self, X, y=None):
        return np.sum(self.score_samples(X))

    def sample(self, n_samples=1, random_state=None):
        return NotImplementedError


class UnivariateCategoricalKernelDensity(BaseKernelDensity):
    def __init__(self, bandwidth=None):
        self.freq = []
        self.sum_freq = 0

    def fit(self, X: np.ndarray, y=None, sample_weight=None):
        assert X.shape[1] == 1
        max_ = int(np.max(X))+1
        if not self.freq:
            self.freq = [1] * max_
        cat_idxs, cnts = np.unique(X.flatten(), return_counts=True)
        for cat_idx, cnt in zip(cat_idxs, cnts):
            self.freq[int(cat_idx)] += cnt
        self.sum_freq = sum(self.freq)
        self.pvals = np.array(self.freq) / self.sum_freq

    def score_samples(self, X: np.ndarray):
        assert X.shape[1] == 1
        freq = np.array(self.freq)
        return np.log(freq[X.flatten] / self.sum_freq)

    def sample(self, n_samples=1, random_state=None):
        rng = check_random_state(random_state)
        return rng.multinomial(self.pvals.size, self.pvals, size=[n_samples, 1])
