#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-16
# @Contact    : tqichun@gmail.com
from ConfigSpace import ConfigurationSpace, Configuration
from hpolib.benchmarks.synthetic_functions import MultiFidelityRosenbrock2D, MultiFidelityRosenbrock5D, \
    MultiFidelityRosenbrock10D, MultiFidelityRosenbrock20D

from ultraopt import fmin
from ultraopt.optimizer import TPEOptimizer, ForestOptimizer
from ultraopt.multi_fidelity import CustomIterGenerator

synthetic_functions = [
    MultiFidelityRosenbrock2D,
    MultiFidelityRosenbrock5D,
    MultiFidelityRosenbrock10D,
    MultiFidelityRosenbrock20D
]

repetitions = 20
base_random_state = 50
base_max_iter = 200

for synthetic_function_cls in synthetic_functions:
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
    def evaluation(config: dict, budget: float=100):
        config = Configuration(config_space, values=config)
        return synthetic_function.objective_function(config, budget=budget)["function_value"] - \
               synthetic_function.get_meta_information()["f_opt"]


    res = fmin(
        evaluation,
        config_space,
        optimizer=TPEOptimizer(min_points_in_model=20),
        # optimizer=ForestOptimizer(min_points_in_model=40),
        n_jobs=1,
        n_iterations=100,
        multi_fidelity_iter_generator=CustomIterGenerator([4, 2, 1], [25, 50, 100])
    )
