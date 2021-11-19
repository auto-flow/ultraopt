#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
import json
import os
from importlib import import_module
from pathlib import Path

from hpbandster.core.dispatcher import Job
from hpbandster.optimizers.config_generators.bohb import BOHB

benchmark = 'rosenbrock'
assert benchmark in ['levy', 'rosenbrock']

levy = {}
levy_module = import_module(f'ultraopt.benchmarks.synthetic_functions.{benchmark}')
# for i in range(2, 20):
# for i in range(30, 1,-1):
# for i in range(2, 3):
for i in range(2, 21):
    levy[i] = getattr(levy_module, f'Levy{i}D' if benchmark == 'levy' else f'Rosenbrock{i}D')

from joblib import Parallel, delayed

import numpy as np
import pandas as pd


def raw2min(df: pd.DataFrame):
    df_m = pd.DataFrame(np.zeros_like(df.values), columns=df.columns)
    for i in range(df.shape[0]):
        df_m.loc[i, :] = df.loc[:i, :].min()
    return df_m


software = 'bohb'
fname = f'{software}_{benchmark}'

final_result = {}
json_name = f"{fname}.json"
# if os.path.exists(json_name):
#     final_result = json.load(json_name)
repetitions = 100
base_random_state = 50
base_max_iter = 200


# 定义目标函数
def evaluator(config: dict, config_space=None, synthetic_function=None):
    config = Configuration(config_space, values=config)
    return synthetic_function.objective_function(config)["function_value"] - \
           synthetic_function.get_meta_information()["f_opt"]


from ConfigSpace import Configuration, ConfigurationSpace
from tqdm import tqdm


def evaluate(trial, config_space, synthetic_function, max_iter):
    config_space.seed(trial)
    bohb = BOHB(config_space,min_points_in_model=20,random_fraction=0,bandwidth_factor=1, num_samples=48)
    losses = []
    for i in tqdm(range(max_iter)):
        config = bohb.get_config(1)[0]
        loss = evaluator(config, config_space, synthetic_function)
        job = Job(id=i, budget=1, config=config)
        job.result = {'loss': loss}
        bohb.new_result(job)
        losses.append(loss)
    print(min(losses))
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
                          "q25": res.quantile(0.25, 1).tolist(),
                          "q10": res.quantile(0.1, 1).tolist(),
                          "q75": res.quantile(0.75, 1).tolist(), "q90": res.quantile(0.90, 1).tolist()}
Path(json_name).write_text(json.dumps(final_result))
from joblib import dump

dump(name2df, f"{fname}.pkl")
