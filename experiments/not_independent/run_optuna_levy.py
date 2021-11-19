#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
import json
import sys, os
from importlib import import_module
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

benchmark = sys.argv[1]
multivariate = sys.argv[2].lower() == 'true'
assert benchmark in ['levy', 'rosenbrock']

print(f'benchmark = {benchmark}')
print(f'multivariate = {multivariate}')

levy = {}
levy_module = import_module(f'ultraopt.benchmarks.synthetic_functions.{benchmark}')
for i in range(2, 21):
# for i in range(20, 1, -1):
    levy[i] = getattr(levy_module, f'Levy{i}D' if benchmark == 'levy' else f'Rosenbrock{i}D')

from joblib import Parallel, delayed
from optuna import Trial
import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, Configuration


def raw2min(df: pd.DataFrame):
    df_m = pd.DataFrame(np.zeros_like(df.values), columns=df.columns)
    for i in range(df.shape[0]):
        df_m.loc[i, :] = df.loc[:i, :].min()
    return df_m


repetitions = 100
base_random_state = 50
base_max_iter = 200


# 定义目标函数

class Evaluator():
    def __init__(self, config_space: ConfigurationSpace = None, synthetic_function=None):
        self.synthetic_function = synthetic_function
        self.config_space = config_space
        self.losses = []

    def __call__(self, trial: Trial, ):
        config = {}
        for hp in self.config_space.get_hyperparameters():
            hp: UniformFloatHyperparameter
            config[hp.name] = trial.suggest_uniform(hp.name, hp.lower, hp.upper)
        config = Configuration(config_space, values=config)
        loss = synthetic_function.objective_function(config)["function_value"] - \
               synthetic_function.get_meta_information()["f_opt"]
        self.losses.append(loss)
        return loss


def evaluate(trial, config_space, synthetic_function, max_iter):
    from ultraopt.tpe import hyperopt_gamma
    random_state = base_random_state + trial * 10
    tpe = TPESampler(
        n_startup_trials=20, multivariate=multivariate,
        gamma=hyperopt_gamma,
        seed=random_state)
    evaluator = Evaluator(config_space, synthetic_function)
    study = optuna.create_study(sampler=tpe, )
    study.optimize(evaluator, n_trials=max_iter)
    print(study)
    return trial, evaluator.losses


software = 'optuna'
fname = f'{software}_{benchmark}'
if multivariate:
    fname += f"_multival"

json_name=f"{fname}.json"

final_result = {}

# if os.path.exists(json_name):
#     final_result =

# @click.command()
# @click.option('--optimizer', '-o', default='ETPE')
# def main(optimizer):
name2df = {}
for dim, synthetic_function_cls in levy.items():
    meta_info = synthetic_function_cls.get_meta_information()
    if "num_function_evals" in meta_info:
        max_iter = meta_info["num_function_evals"]
    else:
        max_iter = base_max_iter
    # 构造超参空间
    config_space = ConfigurationSpace()
    config_space.generate_all_continuous_from_bounds(synthetic_function_cls.get_meta_information()['bounds'])
    synthetic_function = synthetic_function_cls()
    print(meta_info["name"])
    df = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)], index=range(max_iter))

    for trial, losses in Parallel(backend='multiprocessing', n_jobs=-1)(
            delayed(evaluate)(trial, config_space, synthetic_function, max_iter)
            for trial in range(repetitions)
    ):
        df[f"trial-{trial}"] = np.log(np.array(losses))
    name2df[meta_info["name"]] = df
    res = raw2min(df)
    m = res.mean(1)
    s = res.std(1)
    name = synthetic_function.get_meta_information()["name"]
    final_result[name] = {"mean": m.tolist(), "std": s.tolist(),
                          "q25": res.quantile(0.25, 1).tolist(),
                          "q10": res.quantile(0.1, 1).tolist(),
                          "q75": res.quantile(0.75, 1).tolist(), "q90": res.quantile(0.90, 1).tolist()}

Path(json_name).write_text(json.dumps(final_result))
from joblib import dump

dump(name2df, f"{fname}.pkl")
