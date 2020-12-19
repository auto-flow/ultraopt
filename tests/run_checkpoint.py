#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-19
# @Contact    : tqichun@gmail.com
from ConfigSpace import ConfigurationSpace, Configuration
from hpolib.benchmarks.synthetic_functions import  MultiFidelityRosenbrock2D
from joblib import load

from ultraopt import fmin
from ultraopt.multi_fidelity import CustomIterGenerator

repetitions = 20
base_random_state = 50
base_max_iter = 200

synthetic_function_cls = MultiFidelityRosenbrock2D
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
n_iterations = 11
n_jobs = 4
p_res = fmin(
    evaluation,
    config_space,
    optimizer=optimizer,
    n_jobs=n_jobs,
    n_iterations=n_iterations,
    # parallel_strategy="MapReduce",
    checkpoint_file="checkpoint.pkl",
    checkpoint_freq=9,
    multi_fidelity_iter_generator=CustomIterGenerator([4, 2, 1], [25, 50, 100])
)
res = load("checkpoint.pkl")
assert p_res.budget2info == res.budget2info
print(res)


exit(0)

optimizer = "TPE"
n_iterations = 11
n_jobs = 4
p_res = fmin(
    evaluation,
    config_space,
    optimizer=optimizer,
    n_jobs=n_jobs,
    n_iterations=n_iterations,
    parallel_strategy="MapReduce",
    checkpoint_file="checkpoint.pkl",
    checkpoint_freq=20
)
res = load("checkpoint.pkl")
assert p_res.budget2info == res.budget2info
print(res)
