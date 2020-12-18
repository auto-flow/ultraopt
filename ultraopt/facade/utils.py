#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-18
# @Contact    : tqichun@gmail.com
import numpy as np

from ultraopt.optimizer.base_opt import BaseOptimizer
from ultraopt.utils.config_space import get_dict_from_config


def warm_start_optimizer(optimizer: BaseOptimizer, budget2obvs):
    if budget2obvs is None:
        return
    for budget, obvs in budget2obvs.items():
        L = len(obvs["losses"])
        for i in range(L):
            optimizer.tell(
                get_dict_from_config(obvs["configs"][i]),
                obvs["losses"][i],
                budget,
                update_model=(i == L - 1))


def get_wanted(opt_: BaseOptimizer):
    budget2obvs = opt_.budget2obvs
    max_budget = opt_.get_available_max_budget()
    obvs = budget2obvs[max_budget]
    losses = obvs["losses"]
    configs = obvs["configs"]
    idx = np.argmin(losses)
    best_loss = losses[idx]
    best_config = get_dict_from_config(configs[idx])
    return max_budget, best_loss, best_config