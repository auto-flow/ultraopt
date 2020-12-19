#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-19
# @Contact    : tqichun@gmail.com
from hpolib.benchmarks.synthetic_functions import Bohachevsky, Camelback, Forrester
from ConfigSpace import ConfigurationSpace,Configuration
from ultraopt.optimizer import TPEOptimizer
from ultraopt import fmin

synthetic_function_cls=Bohachevsky

config_space = ConfigurationSpace()
config_space.generate_all_continuous_from_bounds(synthetic_function_cls.get_meta_information()['bounds'])
synthetic_function = synthetic_function_cls()

opt=TPEOptimizer()
opt.initialize(config_space)


# 定义目标函数
def evaluation(config: dict):
    config = Configuration(config_space, values=config)
    return synthetic_function.objective_function(config)["function_value"] - \
           synthetic_function.get_meta_information()["f_opt"]

for i in range(20):
    ret=fmin(
        evaluation, config_space, optimizer="TPE",n_iterations=200, n_jobs=10, parallel_strategy="MasterWorkers", random_state=i
    )
