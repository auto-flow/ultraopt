#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from hpolib.benchmarks.synthetic_functions import Bohachevsky, Camelback, Forrester
from hpolib.benchmarks.synthetic_functions.goldstein_price import GoldsteinPrice
from hpolib.benchmarks.synthetic_functions.hartmann3 import Hartmann3
from hpolib.benchmarks.synthetic_functions.hartmann6 import Hartmann6
from hpolib.benchmarks.synthetic_functions.levy import Levy1D
from hpolib.benchmarks.synthetic_functions.rosenbrock import Rosenbrock2D
from hpolib.benchmarks.synthetic_functions.sin_one import SinOne
from hpolib.benchmarks.synthetic_functions.sin_two import SinTwo

from ultraopt import fmin
from ultraopt.optimizer import ETPEOptimizer

synthetic_functions = [
    Bohachevsky,
    Camelback,
    Forrester,
    GoldsteinPrice,
    Hartmann3,
    Hartmann6,
    Levy1D,
    Rosenbrock2D,
    SinOne,
    SinTwo,
]
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace


def raw2min(df: pd.DataFrame):
    df_m = pd.DataFrame(np.zeros_like(df.values), columns=df.columns)
    for i in range(df.shape[0]):
        df_m.loc[i, :] = df.loc[:i, :].min()
    return df_m


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

        # 对experiment_param的删除等操作放在存储后面
        res = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)], index=range(max_iter))
        print(meta_info["name"])
        for trial in range(repetitions):
            random_state = base_random_state + trial * 10
            # 设置超参空间的随机种子（会影响后面的采样）
            ret = fmin(
                evaluation, config_space, optimizer=ETPEOptimizer(
                    gamma2=0.95
                ),
                random_state=random_state, n_iterations=max_iter)
            print(ret)
            losses = ret["budget2obvs"][1]["losses"]
            # print('iter |  loss    | config origin')
            # print('----------------------------')
            print(ret["best_loss"])
            res[f"trial-{trial}"] = losses
        res = raw2min(res)
        m = res.mean(1)
        s = res.std(1)
        name = synthetic_function.get_meta_information()["name"]
        final_result[name] = {"mean": m.tolist(), "std": s.tolist()}
    Path("ultraopt.json").write_text(json.dumps(final_result))


if __name__ == '__main__':
    main()
