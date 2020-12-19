#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-19
# @Contact    : tqichun@gmail.com
from ConfigSpace import ConfigurationSpace, Configuration
from hpolib.benchmarks.synthetic_functions import Bohachevsky

from ultraopt import fmin

synthetic_function_cls = Bohachevsky

config_space = ConfigurationSpace()
config_space.generate_all_continuous_from_bounds(synthetic_function_cls.get_meta_information()['bounds'])
synthetic_function = synthetic_function_cls()


# 定义目标函数
def evaluation(config: dict):
    config = Configuration(config_space, values=config)
    return synthetic_function.objective_function(config)["function_value"] - \
           synthetic_function.get_meta_information()["f_opt"]


for i in range(20):
    ret = fmin(
        evaluation, config_space, optimizer="TPE", n_iterations=200, n_jobs=10, parallel_strategy="AsyncComm",
        random_state=i
    )
