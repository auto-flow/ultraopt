#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-02
# @Contact    : qichun.tang@bupt.edu.cn
import warnings

from joblib import dump
from joblib import parallel_backend
from joblibspark import register_spark

from ultraopt import fmin
from ultraopt.tests.automl import evaluator, config_space

warnings.filterwarnings("ignore")
n_jobs = 4
register_spark()
with parallel_backend("spark"):
    result = fmin(evaluator, config_space, n_jobs=n_jobs, parallel_strategy="MapReduce", n_iterations=40)
dump(result, "pyspark-result.pkl")
