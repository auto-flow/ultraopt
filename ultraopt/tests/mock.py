#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-20
# @Contact    : qichun.tang@bupt.edu.cn
from ConfigSpace import ConfigurationSpace, Configuration

__all__ = ["config_space", "evaluate"]

from ultraopt.benchmarks.synthetic_functions import MultiFidelityRosenbrock2D

synthetic_function_cls = MultiFidelityRosenbrock2D

config_space = ConfigurationSpace()
config_space.generate_all_continuous_from_bounds(synthetic_function_cls.get_meta_information()['bounds'])
synthetic_function = synthetic_function_cls()


# 定义目标函数
def evaluate(config: dict, budget=100):
    config = Configuration(config_space, values=config)
    return synthetic_function.objective_function(config, budget=budget)["function_value"] - \
           synthetic_function.get_meta_information()["f_opt"]
