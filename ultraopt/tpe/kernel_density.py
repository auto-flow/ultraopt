#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-20
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state


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
