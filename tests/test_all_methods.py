#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-19
# @Contact    : tqichun@gmail.com
import unittest

from ultraopt import fmin
from ultraopt.constants import valid_optimizers, valid_parallel_strategies
from ultraopt.multi_fidelity import HyperBandIterGenerator, SuccessiveHalvingIterGenerator
from ultraopt.tests.mock import evaluate, config_space

class TestAllMethod(unittest.TestCase):
    def test_all_methods(self):
        for optimizer in valid_optimizers:
            for parallel_strategie in valid_parallel_strategies:
                if parallel_strategie == "AsyncComm":
                    multi_fidelity_iter_generators = [
                        HyperBandIterGenerator(25, 100, 2),
                        SuccessiveHalvingIterGenerator(25, 100, 2)]
                else:
                    multi_fidelity_iter_generators = [None]
                for multi_fidelity_iter_generator in multi_fidelity_iter_generators:
                    print(optimizer, parallel_strategie, multi_fidelity_iter_generator)
                    ret = fmin(
                        evaluate, config_space, optimizer=optimizer, n_iterations=10, n_jobs=3,
                        parallel_strategy=parallel_strategie, multi_fidelity_iter_generator=multi_fidelity_iter_generator
                    )
                    print(ret)
