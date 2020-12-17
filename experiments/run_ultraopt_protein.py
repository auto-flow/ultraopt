#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from collections import Counter

import numpy as np
import pandas as pd
from ConfigSpace import Configuration

from ultraopt import fmin
from ultraopt.optimizer import TPEOptimizer
from ultraopt.hdl import HDL2CS
from ultraopt.structure import Job

# experiment = "GB1"
experiment = "PhoQ"
# experiment = "RNA"

if experiment == "GB1":
    df = pd.read_csv("GB1.csv")
    df2 = df.copy()
    df2.index = df2.pop('Variants')
    y = df.pop("Fitness")
    for i in range(4):
        df[f'X{i}'] = df['Variants'].str[i]
    df.pop('Variants')

    choices = sorted(Counter(df['X0']).keys())
    hdl = {
        f"X{i}": {"_type": "choice", "_value": choices} for i in range(4)
    }
    config_space = HDL2CS().recursion(hdl)


    def evaluation(config: Configuration):
        return -df2.loc["".join([config.get(f"X{i}") for i in range(4)]), 'Fitness']
elif experiment == "PhoQ":
    df = pd.read_csv("PhoQ.csv")
    df2 = df.copy()
    df2.index = df2.pop('Variants')
    y = df.pop("Fitness")
    for i in range(4):
        df[f'X{i}'] = df['Variants'].str[i]
    df.pop('Variants')

    choices = sorted(Counter(df['X0']).keys())
    hdl = {
        f"X{i}": {"_type": "choice", "_value": choices} for i in range(4)
    }
    config_space = HDL2CS().recursion(hdl)


    def evaluation(config: Configuration):
        return -df2.loc["".join([config.get(f"X{i}") for i in range(4)]), 'Fitness']

repetitions = 10
max_iter = 1000


def main():
    res = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)], index=range(max_iter))
    for trial in range(repetitions):
        random_state = 50 + trial * 10
        # 设置超参空间的随机种子（会影响后面的采样）
        config_space.seed(random_state)
        print("==========================")
        print(f"= Trial -{trial:01d}-               =")
        print("==========================")
        # print('iter |  loss    | config origin')
        # print('----------------------------')
        ret = fmin(
            evaluation, config_space, optimizer=TPEOptimizer(
                gamma1=0.95
            ),
            random_state=random_state, n_iterations=max_iter)
        losses = ret["budget2obvs"][1]["losses"]
        print(ret["best_loss"])
        res[f"trial-{trial}"] = losses
    res.to_csv(f"{experiment}_7.csv", index=False)
    print(res.min()[:repetitions].mean())


if __name__ == '__main__':
    main()
