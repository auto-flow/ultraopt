#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-16
# @Contact    : tqichun@gmail.com
from collections import Counter

import numpy as np
import pandas as pd
from ConfigSpace import Configuration

from ultraopt.optimizer import ETPEOptimizer
from ultraopt.hdl import HDL2CS, layering_config
from ultraopt.structure import Job
from functools import partial
from hyperopt import hp, tpe, fmin, Trials
import hyperopt.pyll.stochastic


experiment = "GB1"
# experiment = "PhoQ"
# experiment = "RNA"


df = pd.read_csv("GB1.csv")
df2 = df.copy()
df2.index = df2.pop('Variants')
y = df.pop("Fitness")
for i in range(4):
    df[f'X{i}'] = df['Variants'].str[i]
df.pop('Variants')

n_top = 3
choices = sorted(Counter(df['X0']).keys())
space = hp.choice(
    "parent",
    [
        {f"O{j}":{f"O{j}X{i}": hp.choice(f"O{j}X{i}", choices) for i in range(4)}}
        for j in range(n_top)
    ]
)
print(hyperopt.pyll.stochastic.sample(space))

choice2coef = {
    0: 0.8,
    1: 0.7,
    2: 1
}


def evaluation(config: dict):
    choice, sub_config = config.popitem()
    coef = choice2coef[int(choice[-1])]
    # print(choice)
    return -df2.loc["".join(list(sub_config.values())), 'Fitness'] * coef


repetitions = 10
max_iter = 1000


def main():
    res = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)], index=range(max_iter))
    for trial in range(repetitions):
        random_state = 50 + trial * 10
        # 设置超参空间的随机种子（会影响后面的采样）
        print("==========================")
        print(f"= Trial -{trial:01d}-               =")
        print("==========================")
        # print('iter |  loss    | config origin')
        # print('----------------------------')
        trials = Trials()
        best = fmin(
            evaluation, space, algo=partial(tpe.suggest, n_startup_jobs=20), max_evals=max_iter,
            rstate=np.random.RandomState(random_state), trials=trials,
        )
        losses = trials.losses()
        res[f"trial-{trial}"] = losses
    res.to_csv(f"conditional_hyperopt.csv", index=False)
    print(res.min()[:repetitions].mean())


if __name__ == '__main__':
    main()
