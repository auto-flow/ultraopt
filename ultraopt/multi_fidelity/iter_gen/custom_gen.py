#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-18
# @Contact    : tqichun@gmail.com
from typing import List

from .base_gen import BaseIterGenerator


class CustomIterGenerator(BaseIterGenerator):
    def __init__(
            self,
            num_configs: List[int],
            budgets: List[float],
            iter_klass: type = None,
    ):
        super(CustomIterGenerator, self).__init__(iter_klass)
        self.budgets = budgets
        self.num_configs = num_configs
        assert len(self.budgets) == len(self.num_configs), ValueError(
            "length of budgets and state configs should be equal.")

    def get_next_iteration(self, iteration, **kwargs):
        return self.iter_klass(
            HPB_iter=iteration,
            num_configs=self.num_configs,
            budgets=self.budgets,
            config_sampler=self.config_sampler,
            **kwargs
        )

    def get_budgets(self):
        return self.budgets

    @property
    def num_all_configs(self) -> int:
        return sum(self.num_configs)
