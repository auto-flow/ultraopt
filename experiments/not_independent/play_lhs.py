#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-30
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from skopt.sampler import Lhs
from skopt.space import Real, Categorical, Integer
from ultraopt.hdl import hdl2cs

r = Real(-15, 10)

c = Categorical(['0', '1', '2'])

HDL = {
    'a': {'_type': 'uniform', '_value': [-15, 10]},
    'b': {'_type': 'uniform', '_value': [-15, 10]},
}

CS = hdl2cs(HDL)

lhs = Lhs()


# points=lhs.generate([('a','b','c'),('a','b','c'),],20)
def CS_to_skopt(CS: ConfigurationSpace):
    skopt_spaces = []
    for hp in CS.get_hyperparameters():
        space = Real(hp.lower, hp.upper)
        skopt_spaces.append(space)
    return skopt_spaces


skopt_spaces = CS_to_skopt(CS)
points = lhs.generate(skopt_spaces, 20)


def sampled_points_to_configurations(points, skopt_spaces, CS):
    points = np.array(points, dtype='float32')
    M = len(skopt_spaces)
    N = points.shape[0]
    for i in range(M):
        space = skopt_spaces[i]
        if isinstance(space, (Real, Integer)):
            points[:, i] -= space.low
            points[:, i] /= (space.high - space.low)
    configs = []
    for i in range(N):
        config = Configuration(CS, vector=points[i, :])
        configs.append(config)
    return configs


print(sampled_points_to_configurations(points, skopt_spaces, CS))
