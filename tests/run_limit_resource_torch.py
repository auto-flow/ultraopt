#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-03-31
# @Contact    : qichun.tang@bupt.edu.cn
from ultraopt import fmin
from tabular_nn import TabularNNClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
import warnings

warnings.filterwarnings("ignore")

HDL = {
    'dummy_a': {"_type": "uniform", "_value": [0, 1]},
    'dummy_b': {"_type": "uniform", "_value": [0, 1]},
}

X, y = load_digits(10, True)


def evaluator(config):
    return 1 - TabularNNClassifier(max_epoch=10).fit(X, y).score(X, y)


result = fmin(
    evaluator, HDL, n_jobs=3, limit_resource=True, verbose=1,
    n_iterations=10,
    time_limit=100, memory_limit=200)
print(result)
