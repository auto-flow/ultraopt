#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from time import time

import numpy as np
from frozendict import frozendict
from scipy.stats import norm
# todo:  PI LCB
from sklearn.utils import check_random_state
class LogEI():
    def __init__(self, xi=0.01):
        self.xi = xi

    def __call__(self, mu, std, X, y_opt):
        var = std ** 2
        values = np.zeros_like(mu)
        mask = std > 0
        f_min = y_opt - self.xi
        improve = f_min - mu[mask]
        # in SMAC, v := scaled
        # smac/optimizer/acquisition.py:388
        scaled = improve / std[mask]
        values[mask] = (np.exp(f_min) * norm.cdf(scaled)) - \
                       (np.exp(0.5 * var[mask] + mu[mask]) * norm.cdf(scaled - std[mask]))
        return values


class EI():
    def __init__(self, xi=0.01):
        # in SMAC, xi=0.0,
        # smac/optimizer/acquisition.py:341
        # par: float=0.0
        # in scikit-optimize, xi=0.01
        # this blog recommend xi=0.01
        # http://krasserm.github.io/2018/03/21/bayesian-optimization/
        self.xi = xi

    def __call__(self, mu, std, X, y_opt):
        values = np.zeros_like(mu)
        mask = std > 0
        improve = y_opt - self.xi - mu[mask]
        scaled = improve / std[mask]
        cdf = norm.cdf(scaled)
        pdf = norm.pdf(scaled)
        exploit = improve * cdf
        explore = std[mask] * pdf
        values[mask] = exploit + explore
        # You can find the derivation of the EI formula in this blog
        # http://ash-aldujaili.github.io/blog/2018/02/01/ei/
        return values