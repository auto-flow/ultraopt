#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
import sys
from importlib import import_module

benchmark = sys.argv[1]
assert benchmark in ['levy', 'rosenbrock']

levy = {}
levy_module = import_module(f'ultraopt.benchmarks.synthetic_functions.{benchmark}')
for i in range(2, 11):
    levy[i] = getattr(levy_module, f'Levy{i}D' if benchmark == 'levy' else f'Rosenbrock{i}D')

from ultraopt import fmin
from joblib import Parallel, delayed
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from functools import partial
from ultraopt.utils.config_space import CS2HyperoptSpace
from hyperopt import tpe, fmin, Trials


def raw2min(df: pd.DataFrame):
    df_m = pd.DataFrame(np.zeros_like(df.values), columns=df.columns)
    for i in range(df.shape[0]):
        df_m.loc[i, :] = df.loc[:i, :].min()
    return df_m


final_result = {}
repetitions = 20
base_random_state = 50
base_max_iter = 200


# 定义目标函数
def evaluator(config: dict, config_space=None, synthetic_function=None):
    config = Configuration(config_space, values=config)
    return synthetic_function.objective_function(config)["function_value"] - \
           synthetic_function.get_meta_information()["f_opt"]


def evaluate(trial, config_space, synthetic_function, max_iter):
    space = CS2HyperoptSpace(config_space)
    trials = Trials()
    random_state = base_random_state + trial * 10
    evaluator_part = partial(evaluator, config_space=config_space, synthetic_function=synthetic_function)

    best = fmin(
        evaluator_part, space, algo=partial(tpe.suggest, n_startup_jobs=20), max_evals=max_iter,
        rstate=np.random.RandomState(random_state), trials=trials,
    )
    losses = trials.losses()
    return trial, losses


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
                          "q10": res.quantile(0.1, 1).tolist(),
                          "q25": res.quantile(0.25, 1).tolist(),
                          "q75": res.quantile(0.75, 1).tolist(), "q90": res.quantile(0.90, 1).tolist()}
software = 'hyperopt'
name = f'{software}_{benchmark}'
Path(f"{name}.json").write_text(json.dumps(final_result))
from joblib import dump

dump(name2df, f"{name}.pkl")
