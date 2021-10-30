#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-20
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
from ultraopt.tpe import estimate_bw


class NormalizedKernelDensity(KernelDensity):

    def fit(self, X, y=None, sample_weight=None):
        self.normalizer_ = MinMaxScaler()
        X = self.normalizer_.fit_transform(X)
        self.bandwidth = estimate_bw(X)
        return super(NormalizedKernelDensity, self).fit(X, y, sample_weight)

    def score_samples(self, X, y=None):
        '''score = return np.sum(self.score_samples(X))'''
        X = self.normalizer_.transform(X)
        return super(NormalizedKernelDensity, self).score_samples(X)

    def sample(self, n_samples=1, random_state=None):
        samples = super(NormalizedKernelDensity, self).sample(n_samples, random_state)
        return self.normalizer_.inverse_transform(samples)


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
        max_ = int(np.max(X)) + 1
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
