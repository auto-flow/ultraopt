#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-18
# @Contact    : qichun.tang@bupt.edu.cn
from typing import List, Union

from .base_gen import BaseIterGenerator


class CustomIterGenerator(BaseIterGenerator):
    def __init__(
            self,
            num_configs_list: Union[List[int], List[List[int]]],
            budgets_list: Union[List[float], List[List[float]]],
            iter_klass: type = None,
    ):
        super(CustomIterGenerator, self).__init__(iter_klass)
        if isinstance(num_configs_list[0], (list, tuple)):
            self.budgets_list_ = budgets_list
            self.num_configs_list_ = num_configs_list
            self.iter_cycle_ = len(budgets_list)
            assert len(budgets_list) == len(num_configs_list), ValueError
            for budgets, num_configs in zip(budgets_list, num_configs_list):
                assert len(budgets) == len(num_configs), ValueError(
                    "length of budgets and state configs should be equal.")

        else:
            self.budgets_list_ = [budgets_list]
            self.num_configs_list_ = [num_configs_list]
            self.iter_cycle_ = 1
