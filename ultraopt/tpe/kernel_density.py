#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-20
# @Contact    : qichun.tang@bupt.edu.cn
from collections import Counter

import numpy as np
import scipy
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_random_state
from ultraopt.tpe import estimate_bw
from sklearn.base import  TransformerMixin

class BaseKernelDensity(BaseEstimator):
    def fit(self, X, y=None, sample_weight=None):
        raise NotImplementedError

    def score_samples(self, X):
        '''log_density'''
        raise NotImplementedError

    def score(self, X, y=None):
        return np.sum(self.score_samples(X))

    def sample(self, n_samples=1, random_state=None):
        return NotImplementedError

from sklearn.preprocessing import StandardScaler


class DiyTransformer(StandardScaler):
    def fit(self,X,**kwargs):
        self.mean_=np.mean(X,axis=0)
        self.scale_=np.array([estimate_bw(X[:,i]) for i in range(X.shape[1])])
        return self



class NormalizedKernelDensity(KernelDensity):

    def fit(self, X, y=None, sample_weight=None):
        from sklearn.preprocessing import StandardScaler,MinMaxScaler, RobustScaler, QuantileTransformer
        # self.normalizer_ = DiyTransformer()
        # X = self.normalizer_.fit_transform(X)
        self.bandwidth = estimate_bw(X)
        return super(NormalizedKernelDensity, self).fit(X, y, sample_weight)

    def score_samples(self, X):
        '''score = return np.sum(self.score_samples(X))'''
        # X = self.normalizer_.transform(X)
        return super(NormalizedKernelDensity, self).score_samples(X)

    def sample(self, n_samples=1, random_state=None):
        samples = super(NormalizedKernelDensity, self).sample(n_samples, random_state)
        return samples
        # return self.normalizer_.inverse_transform(samples)


EPS = 1e-12
SIGMA0_MAGNITUDE = 0.2


def _normal_cdf(x: float, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    mu, sigma = map(np.asarray, (mu, sigma))
    denominator = x - mu
    numerator = np.maximum(np.sqrt(2) * sigma, EPS)
    z = denominator / numerator
    return 0.5 * (1 + scipy.special.erf(z))


class MultivariateKernelDensity(BaseKernelDensity):

    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth

    def set_bounds(self, bounds):
        self.bounds = bounds

    def build_multivar_norm(self):
        self.bandwidth = np.array(self.bandwidth)
        self.cov = np.diag(self.bandwidth)
        self.dists = []
        for i in range(self.N):
            self.dists.append(multivariate_normal(mean=self.data[i, :], cov=self.cov))

    def fit(self, X, y=None, sample_weight=None):
        self.data = X
        N, D = X.shape
        assert len(self.bounds) == D
        self.bandwidth = []
        self.mus = []
        self.sigmas = []
        self.p_accepts = []
        self.log_p_accept = 0
        for i in range(D):
            univar_data = X[:, i]
            # fixme: bandwidth=sigma是否符合预期
            sigma = estimate_bw(univar_data)
            self.bandwidth.append(sigma)
            self.mus.append(univar_data)
            self.sigmas.append(np.ones([N]) * sigma)
            if self.bounds[i] is None:
                p_accept = 1
            else:
                low, high = self.bounds[i]
                p_accept = (_normal_cdf(low, self.mus[i], self.sigmas[i]) -
                            _normal_cdf(high, self.mus[i], self.sigmas[i])).mean()
            self.log_p_accept += np.log(p_accept)
            self.p_accepts.append(p_accept)
        self.N, self.D = N, D
        self.build_multivar_norm()
        return self

    def score_samples(self, X):
        self.build_multivar_norm()
        cur_N, D = X.shape
        assert self.D == D
        log_pdf = np.zeros([cur_N]) - self.log_p_accept
        for i in range(self.N):
            log_pdf += self.dists[i].logpdf(X)
        log_pdf -= np.log(self.N)
        return log_pdf

    def sample(self, n_samples=1, random_state=None):
        self.build_multivar_norm()
        rng = check_random_state(random_state)
        return np.vstack([
            self.dists[ix].rvs(size=sub_n_samples, random_state=rng)
            for ix, sub_n_samples in Counter(rng.randint(0, self.N, size=[n_samples])).items()
        ])


class UnivariateCategoricalKernelDensity(BaseKernelDensity):
    def __init__(self):
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
