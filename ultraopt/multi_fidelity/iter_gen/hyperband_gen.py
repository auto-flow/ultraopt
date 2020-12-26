#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-20
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np

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
        budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter - 1, 0, self.max_SH_iter))
        self.budgets = budgets.tolist()
        if self.SH_only:
            s = self.max_SH_iter - 1
            self.iter_cycle_ = 1
            ns = self.get_ns(s)
            self.num_configs_list_ = [ns]
            self.budgets_list_ = [self.budgets]
        else:
            self.iter_cycle_ = self.max_SH_iter
            self.num_configs_list_ = []
            self.budgets_list_ = []
            for s in reversed(range(self.max_SH_iter)):
                ns = self.get_ns(s)
                self.num_configs_list_.append(ns)
                self.budgets_list_.append(self.budgets[(-s - 1):])

    def get_ns(self, s):
        n0 = int(np.floor((self.max_SH_iter) / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]
        return ns


class SuccessiveHalvingIterGenerator(HyperBandIterGenerator):
    def __init__(
            self,
            min_budget,
            max_budget,
            eta,
            iter_klass=None
    ):
        super(SuccessiveHalvingIterGenerator, self).__init__(min_budget, max_budget, eta, True, iter_klass)
