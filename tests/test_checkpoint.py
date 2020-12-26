#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-20
# @Contact    : qichun.tang@bupt.edu.cn
import unittest

from joblib import load

from ultraopt import fmin
from ultraopt.constants import valid_parallel_strategies
from ultraopt.multi_fidelity import CustomIterGenerator
from ultraopt.tests.mock import evaluate, config_space


class TestCheckpoint(unittest.TestCase):
    def test(self):
        for parallel_strategy in valid_parallel_strategies:
            print(parallel_strategy)
            optimizer = "ETPE"
            n_iterations = 11
            n_jobs = 4
            p_res = fmin(
                evaluate,
                config_space,
                optimizer=optimizer,
                n_jobs=n_jobs,
                n_iterations=n_iterations,
                parallel_strategy=parallel_strategy,
                checkpoint_file="checkpoint.pkl",
                checkpoint_freq=9,
                multi_fidelity_iter_generator=CustomIterGenerator([4, 2, 1], [25, 50, 100])
            )
            res = load("checkpoint.pkl")
            assert p_res.budget2info == res.budget2info
            print(res)
