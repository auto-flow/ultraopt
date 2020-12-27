#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-18
# @Contact    : qichun.tang@bupt.edu.cn
from typing import Type

import pandas as pd

from ultraopt.multi_fidelity.iter import BaseIteration, RankReductionIteration
from ultraopt.utils.misc import pbudget


class BaseIterGenerator():
    def __init__(
            self,
            iter_klass: type = None
    ):
        if iter_klass is None:
            iter_klass = RankReductionIteration
        assert issubclass(iter_klass, BaseIteration)
        self.iter_klass: Type[BaseIteration] = iter_klass
        self.optimizer = None

    def initialize(self, optimizer):
        self.optimizer = optimizer

    @property
    def num_configs_list(self):
        return self.num_configs_list_

    @property
    def budgets_list(self):
        return self.budgets_list_

    @property
    def iter_cycle(self):
        return self.iter_cycle_

    def get_budgets(self):
        return self.budgets_list[0]

    def num_all_configs(self, n_iterations) -> int:
        def get_sum(num_configs_list):
            return sum([sum(num_configs) for num_configs in num_configs_list])

        L = self.iter_cycle
        N = n_iterations // L
        M = n_iterations % L
        res = 0
        res += N * get_sum(self.num_configs_list)
        res += get_sum(self.num_configs_list[:M])
        return res

    def get_next_iteration(self, iteration, **kwargs):
        iter_ix = iteration % self.iter_cycle
        return self.iter_klass(
            HPB_iter=iteration,
            num_configs=self.num_configs_list[iter_ix],
            budgets=self.budgets_list[iter_ix],
            optimizer=self.optimizer,
            **kwargs
        )

    def get_table(self, init_iter=0):
        column_index_list = []
        data = [[], []]
        row_index = ["num_config", "budget"]
        for iter, (num_configs, budgets) in enumerate(zip(
                self.num_configs_list, self.budgets_list)):
            for stage, (num_config, budget) in enumerate(zip(
                    num_configs, budgets)):
                data[0].append(str(num_config))
                data[1].append(pbudget(budget))
                column_index_list.append([f"iter {iter + init_iter}", f"stage {stage}"])
        df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(column_index_list), index=row_index)
        return df

    def __str__(self):
        return self.__class__.__name__ + " :\n" + str(self.get_table())

    __repr__ = __str__
