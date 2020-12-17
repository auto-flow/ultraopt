#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-17
# @Contact    : tqichun@gmail.com
import importlib
import inspect
from typing import Callable, Union, Optional, List, Type

from ConfigSpace import ConfigurationSpace, Configuration

from ultraopt.optimizer.base_opt import BaseOptimizer
from ultraopt.hdl import HDL2CS


# 设计目标：单机并行、多保真优化
def fmin(
        eval_func: Callable,
        config_space: Union[ConfigurationSpace, dict],
        config_generator: Union[BaseOptimizer, str, Type] = "TPE",
        initial_points: Union[None, List[Configuration], List[dict]] = None,
        random_state=42,
        n_iterations=100,
        n_jobs=1,
        budgets: Optional[List[float]] = None,
        stage_configs: Optional[List[float]] = None,
        multi_fidelity: Optional[str] = None,
):
    # ------------   config_space   ---------------#
    if isinstance(config_space, dict):
        checked_cs = HDL2CS()(config_space)
    elif isinstance(config_space, ConfigurationSpace):
        checked_cs = config_space
    else:
        raise NotImplementedError
    # ------------      budgets     ---------------#
    if budgets is None:
        checked_budgets = [1]
    else:
        checked_budgets = list(budgets)
    # ------------ config_generator ---------------#
    if inspect.isclass(config_generator):
        if not issubclass(config_generator, BaseOptimizer):
            raise ValueError(f"config_generator {config_generator} is not subclass of BaseOptimizer")
        checked_cg = config_generator()
    elif isinstance(config_generator, BaseOptimizer):
        checked_cg = config_generator
    elif isinstance(config_generator, str):
        try:
            checked_cg = getattr(importlib.import_module("ultraopt.optimizer"),
                                 f"{config_generator}Optimizer")()
        except Exception:
            raise ValueError(f"Invalid config_generator string-indicator: {config_generator}")
    else:
        raise NotImplementedError
    # non-parallelism debug mode
    if n_jobs == 1 and (not multi_fidelity):
        checked_budgets = [1]
        checked_cg.initialize(checked_cs, checked_budgets, random_state, initial_points)
        pass
    else:
        raise NotImplementedError
