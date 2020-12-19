#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-19
# @Contact    : tqichun@gmail.com
from ConfigSpace import ConfigurationSpace, Configuration
from hpolib.benchmarks.synthetic_functions import Bohachevsky

from ultraopt import fmin

repetitions = 20
base_random_state = 50
base_max_iter = 200

synthetic_function_cls = Bohachevsky
meta_info = synthetic_function_cls.get_meta_information()
if "num_function_evals" in meta_info:
    max_iter = meta_info["num_function_evals"]
else:
    max_iter = base_max_iter

# 构造超参空间
config_space = ConfigurationSpace()
config_space.generate_all_continuous_from_bounds(synthetic_function_cls.get_meta_information()['bounds'])
synthetic_function = synthetic_function_cls()


# 定义目标函数
def evaluation(config: dict, budget: float = 100):
    config = Configuration(config_space, values=config)
    return synthetic_function.objective_function(config, budget=budget)["function_value"] - \
           synthetic_function.get_meta_information()["f_opt"]
optimizer="TPE"
n_iterations=30
p_res = fmin(
    evaluation,
    config_space,
    optimizer=optimizer,
    n_jobs=1,
    n_iterations=n_iterations,
)
# todo: 传递 _bw_factor
for i in range(3):
    res = fmin(
        evaluation,
        config_space,
        optimizer=optimizer,
        n_jobs=1,
        n_iterations=n_iterations,
        previous_budget2obvs=p_res["budget2obvs"]
    )
    p_res = res
    print(len(res["budget2obvs"][1]["losses"]))

p_res = fmin(
    evaluation,
    config_space,
    optimizer="TPE",
    n_jobs=1,
    n_iterations=n_iterations*4,
)