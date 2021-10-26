#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-23
# @Contact    : qichun.tang@bupt.edu.cn
import sys
from collections import Counter
from typing import Union

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, CategoricalHyperparameter, ConfigurationSpace
from hyperopt import hp
from ultraopt.hdl import hdl2cs


def raw2min(df: pd.DataFrame):
    df_m = pd.DataFrame(np.zeros_like(df.values), columns=df.columns)
    for i in range(df.shape[0]):
        df_m.loc[i, :] = df.loc[:i, :].min()
    return df_m


benchmark = sys.argv[1]
print(benchmark)
repetitions = int(sys.argv[2])
max_iter = int(sys.argv[3])
n_startup_trials = int(sys.argv[4])

print(f"repetitions={repetitions}, max_iter={max_iter}, n_startup_trials={n_startup_trials}")


class BaseEvaluator():
    def __init__(self):
        self.losses = []

    def call(self, config: dict):
        raise NotImplementedError

    def __call__(self, config: Union[Configuration, dict]):
        config = config.get_dictionary() if isinstance(config, Configuration) else config
        loss = self.call(config)
        self.losses.append(loss)
        return loss


if benchmark in ["GB1", "PhoQ"]:
    df = pd.read_csv(f"{benchmark}.csv")
    df2 = df.copy()
    df2.index = df2.pop('Variants')
    y = df.pop("Fitness")
    loss = y.max() - y
    df2['loss'] = np.array(loss)
    for i in range(4):
        df[f'X{i}'] = df['Variants'].str[i]
    df.pop('Variants')

    choices = sorted(Counter(df['X0']).keys())
    hdl = {
        f"X{i}": {"_type": "choice", "_value": choices} for i in range(4)
    }


    class Evaluator(BaseEvaluator):
        def call(self, config: dict):
            return df2.loc["".join([config.get(f"X{i}") for i in range(4)]), 'loss']

elif benchmark == "RNA":
    import RNA

    choices = ["A", "U", "G", "C"]
    hdl = {
        f"X{i}": {"_type": "choice", "_value": choices} for i in range(30)
    }


    class Evaluator(BaseEvaluator):
        def call(self, config: dict):
            return RNA.fold("".join([config.get(f"X{i}") for i in range(30)]))[1]
elif benchmark == "arylation":
    csv = 'All_CH_arylation_Experiments_Jay_12052019.csv'
    df = pd.read_csv(csv)
    label = df.pop('yield').tolist()
    df = df.astype(str)
    max_ = max(label)
    hdl = {}
    columns = sorted(df.columns)
    df = df[columns]
    for column in columns:
        hdl[column] = {"_type": "choice", "_value": sorted(list(set(df[column])))}
    mapper = {}
    for i, row in df.iterrows():
        key = tuple(row.to_list())
        value = max_ - label[i]
        mapper[key] = value


    class Evaluator(BaseEvaluator):
        def call(self, config: dict):
            return mapper[tuple(config.values())]
else:
    raise Exception

evaluator = Evaluator()

cs = hdl2cs(hdl)


# 一个将configspace转hyperopt空间的函数
def CS2HyperoptSpace(cs: ConfigurationSpace):
    result = {}
    for hyperparameter in cs.get_hyperparameters():
        name = hyperparameter.name
        if isinstance(hyperparameter, CategoricalHyperparameter):
            result[name] = hp.choice(name, hyperparameter.choices)
        elif isinstance(hyperparameter, CategoricalHyperparameter):
            lower = hyperparameter.lower
            upper = hyperparameter.upper
            result[name] = hp.uniform(name, lower, upper)
        else:
            raise ValueError
        # todo: 考虑更多情况
    return result


space = CS2HyperoptSpace(cs)
