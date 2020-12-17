#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-17
# @Contact    : tqichun@gmail.com
from typing import Callable, Union, Optional

from ConfigSpace import ConfigurationSpace

from ultraopt.config_generators.base_cg import BaseConfigGenerator


# 设计目标：单机并行、多保真优化
def fmin(
        eval_func: Callable,
        config_space: Union[ConfigurationSpace, dict],
        config_generator: Union[BaseConfigGenerator, str] = "TPE",
        initial_points:Union[None,List]=None,
        n_jobs=1,
        random_state=42,
        n_iterations=100,
)
