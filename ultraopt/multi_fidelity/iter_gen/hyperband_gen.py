#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-20
# @Contact    : tqichun@gmail.com
import numpy as np

from ultraopt.multi_fidelity.iter import RankReductionIteration
from ultraopt.utils.misc import get_max_SH_iter
from .base_gen import BaseIterGenerator


class HyperBandIterGenerator(BaseIterGenerator):
    def __init__(
            self,
            min_budget,
            max_budget,
            eta,
            SH_only=False,
            iter_klass=None
    ):
        super(HyperBandIterGenerator, self).__init__(iter_klass)
        self.SH_only = SH_only
        self.eta = eta
        self.max_budget = max_budget
        self.min_budget = min_budget
        self.max_SH_iter = get_max_SH_iter(self.min_budget, self.max_budget, self.eta)
        self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter - 1, 0, self.max_SH_iter))
        if self.SH_only:
            s = self.max_SH_iter - 1
            # todo 收纳代码
            n0 = int(np.floor((self.max_SH_iter) / (s + 1)) * self.eta ** s)
            ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]
            self.configs_loop = [ns]
        else:
            self.configs_loop = []
            for s in reversed(range(self.max_SH_iter)):
                n0 = int(np.floor((self.max_SH_iter) / (s + 1)) * self.eta ** s)
                ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]
                self.configs_loop.append(ns)

    def get_next_iteration(self, iteration, **kwargs):
        if self.SH_only:
            s = self.max_SH_iter - 1
        else:
            s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter) / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]
        return RankReductionIteration(
            HPB_iter=iteration,
            num_configs=ns,
            budgets=self.budgets[(-s - 1):],
            config_sampler=self.config_sampler,
            **kwargs
        )

    def get_budgets(self):
        return self.budgets

    def num_all_configs(self, n_iterations) -> int:
        def get_sum(configs_loop):
            return sum([sum(num_configs) for num_configs in configs_loop])

        L = len(self.configs_loop)
        N = n_iterations // L
        M = n_iterations % L
        res = 0
        res += N * get_sum(self.configs_loop)
        res += get_sum(self.configs_loop[:M])
        return res
