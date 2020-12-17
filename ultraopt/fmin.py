#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-17
# @Contact    : tqichun@gmail.com
from typing import Callable, Union, Optional, List

from ConfigSpace import ConfigurationSpace, Configuration

from ultraopt.config_generators.base_cg import BaseConfigGenerator


# 设计目标：单机并行、多保真优化
def fmin(
        eval_func: Callable,
        config_space: Union[ConfigurationSpace, dict],
        config_generator: Union[BaseConfigGenerator, str] = "TPE",
        initial_points: Union[None, List[Configuration], List[dict]] = None,
        random_state=42,
        n_iterations=100,
        n_jobs=1,
        budgets: Optional[List[float]] = None,
        stage_configs: Optional[List[float]] = None,
        multi_fidelity: Optional[str] = None,
):
    pass
