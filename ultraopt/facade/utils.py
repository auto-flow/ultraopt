#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-18
# @Contact    : qichun.tang@bupt.edu.cn

from typing import Union

import numpy as np
from joblib import load

from ultraopt.optimizer.base_opt import BaseOptimizer
from ultraopt.utils.config_space import get_dict_from_config


def warm_start_optimizer(optimizer: BaseOptimizer, previous_result,
                         warm_start_strategy="resume"):
    from ultraopt.facade.result import FMinResult
    if previous_result is None:
        return optimizer
    if isinstance(previous_result, str):
        previous_result = load(previous_result)  # type: Union[FMinResult, BaseOptimizer]
    if warm_start_strategy == "resume":
        for budget, obvs in previous_result.budget2obvs.items():
            L = len(obvs["losses"])
            for i in range(L):
                optimizer.tell(
                    get_dict_from_config(obvs["configs"][i]),
                    obvs["losses"][i],
                    budget,
                    update_model=(i == L - 1))
    elif warm_start_strategy == "continue":
        if isinstance(previous_result, FMinResult):
            prev_opt = previous_result.optimizer
        else:
            prev_opt = previous_result
        prev_opt.resume_time()
        return prev_opt
    else:
        raise ValueError(f"Invalid warm_start_strategy {warm_start_strategy} not in ['resume', 'continue']")
    return optimizer


def get_wanted(opt_: BaseOptimizer):
    budget2obvs = opt_.budget2obvs
    max_budget = opt_.get_available_max_budget()
    obvs = budget2obvs[max_budget]
    losses = obvs["losses"]
    configs = obvs["configs"]
    if len(losses):
        idx = np.argmin(losses)
        best_loss = losses[idx]
        best_config = get_dict_from_config(configs[idx])
        return max_budget, best_loss, best_config
    return max_budget, np.inf, None
