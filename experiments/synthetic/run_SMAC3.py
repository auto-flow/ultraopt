#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-08
# @Contact    : qichun.tang@bupt.edu.cn
'''
smac  0.12.0
'''

import json
from pathlib import Path

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from ultraopt.benchmarks.synthetic_functions.bohachevsky import Bohachevsky
from ultraopt.benchmarks.synthetic_functions.camelback import Camelback
from ultraopt.benchmarks.synthetic_functions.hartmann3 import Hartmann3
from ultraopt.benchmarks.synthetic_functions.hartmann6 import Hartmann6
from ultraopt.benchmarks.synthetic_functions.levy import Levy3D, Levy6D, Levy9D
from ultraopt.benchmarks.synthetic_functions.rosenbrock import Rosenbrock2D
from ultraopt.benchmarks.synthetic_functions.sin_two import SinTwo
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario


def raw2min(df: pd.DataFrame):
    df_m = pd.DataFrame(np.zeros_like(df.values), columns=df.columns)
    for i in range(df.shape[0]):
        df_m.loc[i, :] = df.loc[:i, :].min()
    return df_m


synthetic_functions = [
    Bohachevsky,
    Camelback,
    Hartmann3,
    Hartmann6,
    Levy3D,
    Levy6D,
    Levy9D,
    Rosenbrock2D,
    SinTwo,
]

final_result = {}
repetitions = 20
base_random_state = 50
base_max_iter = 200


def main():
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
        def evaluation(config: dict):
            config = Configuration(config_space, values=config)
            return synthetic_function.objective_function(config)["function_value"] - \
                   synthetic_function.get_meta_information()["f_opt"]

        res = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)], index=range(max_iter))
        print(meta_info["name"])
        for trial in range(repetitions):
            random_state = base_random_state + 10 * trial
            # Scenario object
            scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                                 "runcount-limit": max_iter,
                                 # max. number of function evaluations; for this example set to a low number
                                 "cs": config_space,  # configuration space
                                 "deterministic": "true"
                                 })
            smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(random_state),
                            tae_runner=evaluation, initial_design_kwargs={"init_budget": 20})
            incumbent = smac.optimize()
            runhistory = smac.runhistory
            configs = runhistory.get_all_configs()
            losses = [runhistory.get_cost(config) for config in configs]
            res[f"trial-{trial}"] = losses
            print(min(losses))
        res = raw2min(res)
        m = res.mean(1)
        s = res.std(1)
        name = synthetic_function.get_meta_information()["name"]
        final_result[name] = {"mean": m.tolist(), "std": s.tolist()}
    Path(f"SMAC3.json").write_text(json.dumps(final_result))

if __name__ == '__main__':
    main()