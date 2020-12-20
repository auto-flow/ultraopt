#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-20
# @Contact    : tqichun@gmail.com

from ultraopt.multi_fidelity import HyperBandIterGenerator

hp_iter_gen = HyperBandIterGenerator(1 / 16, 1, 4, SH_only=True)
res = hp_iter_gen.num_all_configs(3)
assert res == 21 * 3
res = hp_iter_gen.num_all_configs(5)
assert res == 21 * 5

hp_iter_gen = HyperBandIterGenerator(1 / 16, 1, 4)
res = hp_iter_gen.num_all_configs(3)
assert res == 29
res = hp_iter_gen.num_all_configs(5)
assert res == 55

from ConfigSpace import ConfigurationSpace, Configuration
from hpolib.benchmarks.synthetic_functions import MultiFidelityRosenbrock5D

from ultraopt import fmin

repetitions = 20
base_random_state = 50
base_max_iter = 200

synthetic_function_cls = MultiFidelityRosenbrock5D
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


optimizer = "TPE"
n_iterations = 100
p_res = fmin(
    evaluation,
    config_space,
    optimizer=optimizer,
    n_jobs=2,
    n_iterations=n_iterations,
    multi_fidelity_iter_generator=HyperBandIterGenerator(25, 100, 2, SH_only=False)
)
print(p_res)
p_res = fmin(
    evaluation,
    config_space,
    optimizer=optimizer,
    n_jobs=2,
    n_iterations=n_iterations,
    multi_fidelity_iter_generator=HyperBandIterGenerator(25, 100, 2, SH_only=True)
)
print(p_res)
