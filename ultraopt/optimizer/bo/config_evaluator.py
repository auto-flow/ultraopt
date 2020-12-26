#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
from copy import deepcopy
from time import time

import numpy as np
from frozendict import frozendict
from scipy.stats import norm
# todo:  PI LCB
from sklearn.utils import check_random_state

from ultraopt.utils.logging_ import get_logger
from ultraopt.utils.misc import pbudget


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


class ConfigEvaluator:
    def __init__(
            self, budget2epm, budget,
            acq_func="EI", acq_func_params=frozendict(), random_state=0
    ):
        self.acq_func_params = dict(acq_func_params)
        # todo: 引入包的形式
        if acq_func == "EI":
            acq_func_cls = EI
        elif acq_func == "LogEI":
            acq_func_cls = LogEI
        else:
            raise NotImplementedError
        self.acq_func = acq_func_cls(**self.acq_func_params)
        self.budget2weight = None
        self.budget = budget
        self.budget2epm = budget2epm
        self.logger = get_logger(self)
        self.rng = check_random_state(random_state)

    def calc_weight(self, X_test, y_test):
        # 计算outputs
        budgets = []
        epms = []
        for budget, epm in self.budget2epm.items():
            if epm is None:
                continue
            budgets.append(budget)
            epms.append(epm)
        max_budget_idx = int(np.argmax(budgets))
        # epms[max_budget_idx] = max_budget_epm
        outputs = [epm.predict(X_test)[None, :] for epm in epms]
        outputs = np.vstack(outputs)
        if budgets[max_budget_idx] != self.budget:
            self.logger.warning(f"In here max_budget = {budgets[max_budget_idx]}, != {self.budget}, it's invalid ! ")
            self.budget2weight = None
            return

        def linear_regression(w):
            w = np.array(w)[:, None]
            output = outputs * w
            sum = output.sum(axis=0)
            return sum

        def rank_loss(output, label):
            return np.count_nonzero(np.logical_xor((output < output.T), (label < label.T)))

        def objective(w):
            output = linear_regression(w)[None, :]
            return rank_loss(output, y_test[None, :])

        # todo: 向量化上述操作
        L = len(budgets)
        # lb = [0] * L
        # ub = [1] * L
        # rranges=[slice(0,1,20) for i in range(L)]
        N = 10000  # 1w 样本随机搜索
        weights = self.rng.rand(N, L)
        weights[weights - (1e-3) <= 0] = 0
        weights[weights + (1e-3) >= 1] = 1
        losses = [objective(weights[i, :]) for i in range(N)]
        result = weights[np.argmin(losses), :]
        # result = differential_evolution(objective, bounds=Bounds(lb, ub), maxiter=1000)
        # result = brute(objective, rranges)
        self.logger.debug(str(result))
        w = result
        if np.sum(w) == 0:
            w = np.ones([L]) * (1 / L)
        else:
            w /= np.sum(w)
        weight = w.tolist()
        return dict(zip(budgets, weight))

    def update_weight(self, budget2weight):
        # self.budget2weight = dict(zip(budgets, weight))
        self.budget2weight = budget2weight
        msg = ""
        for budget, weight in self.budget2weight.items():
            msg += f"W[{pbudget(budget)}] = {weight:.2f}  "
        # msg += f"ranking loss = {objective(w)}"
        self.logger.debug(msg)
        # 不做集成学习的形式

    def __call__(self, X, y_opt):
        start_time = time()
        if self.budget2weight is None:
            epm = self.budget2epm[self.budget]
            mean, std = epm.predict(X, return_std=True)
        else:
            # 集成学习
            # todo: 如果太慢，可以考虑先用1个筛100个点出来，然后再细筛
            mean = np.zeros([X.shape[0]], dtype="float64")
            var = deepcopy(mean)
            for budget, weight in self.budget2weight.items():
                mean_, std_ = self.budget2epm[budget].predict(X, return_std=True)
                var_ = np.square(std_)
                mean += mean_ * weight
                var += var_ * (weight ** 2)
            std = np.sqrt(var)
        cost_time = time() - start_time
        self.logger.debug(f"evaluate cost {cost_time:.3f}s")
        return self.acq_func(mean, std, X, y_opt)
