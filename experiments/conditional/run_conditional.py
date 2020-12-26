#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-16
# @Contact    : qichun.tang@bupt.edu.cn
from collections import Counter

import numpy as np
import pandas as pd
from ConfigSpace import Configuration

from ultraopt.optimizer import ETPEOptimizer
from ultraopt.hdl import HDL2CS, layering_config
from ultraopt.structure import Job

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
hdl = {
    "parent(choice)": {
        f"C{j}": {
            f"X{i}": {"_type": "choice", "_value": choices} for i in range(4)
        }
        for j in range(n_top)
    }
}
config_space = HDL2CS().recursion(hdl)

choice2coef = {
    0: 0.8,
    1: 0.7,
    2: 1
}


def evaluation(config: Configuration):
    dict_ = layering_config(config)
    choice, sub_config = dict_['parent'].popitem()
    coef = choice2coef[int(choice[-1])]
    # print(choice)
    return -df2.loc["".join([sub_config.get(f"X{i}") for i in range(4)]), 'Fitness'] * coef


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
        ambo = ETPEOptimizer(
            config_space, [1], random_state=random_state, min_points_in_model=20,
            # initial_points=config_space.sample_configuration(40)
        )
        loss = np.inf
        for ix in range(max_iter):
            config, config_info = ambo.get_config(1)
            cur_loss = evaluation(config)
            loss = min(loss, cur_loss)
            print(f" {ix:03d}   {loss:.4f}    {config_info.get('origin')}")
            job = Job("")
            job.result = {"loss": cur_loss}
            job.kwargs = {"budget": 1, "config": config, "config_info": config_info}
            ambo.new_result(job)
            res.loc[ix, f"trial-{trial}"] = cur_loss
        print(loss)
    res.to_csv(f"conditional_ultraopt.csv", index=False)
    print(res.min()[:repetitions].mean())


if __name__ == '__main__':
    main()
