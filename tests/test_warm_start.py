#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-19
# @Contact    : qichun.tang@bupt.edu.cn
import unittest

from ultraopt import fmin
from ultraopt.constants import valid_warm_start_strategies
from ultraopt.multi_fidelity import HyperBandIterGenerator
from ultraopt.tests.mock import evaluate, config_space


class TestWarmStart(unittest.TestCase):
    def test_warm_start_serial(self):
        # fixme: resume strategy occur `runId not in runId2info`
        optimizer = "ETPE"
        n_iterations = 5
        for warm_start_strategy in valid_warm_start_strategies:
            print(warm_start_strategy)
            p_res = fmin(
                evaluate,
                config_space,
                optimizer=optimizer,
                n_jobs=1,
                n_iterations=n_iterations,
            )
            for i in range(3):
                res = fmin(
                    evaluate,
                    config_space,
                    warm_start_strategy=warm_start_strategy,
                    n_jobs=1,
                    n_iterations=n_iterations,
                    previous_result=p_res
                )
                p_res = res
                assert len(res["budget2obvs"][1]["losses"]) == n_iterations * (i + 2)

    # def test_warm_start_mapreduce(self):
    #     # fixme: resume strategy occur `runId not in runId2info`
    #     optimizer = "ETPE"
    #     n_iterations = 5
    #     n_jobs = 3
    #     parallel_strategy = "MapReduce"
    #     for warm_start_strategy in valid_warm_start_strategies:
    #         print(warm_start_strategy)
    #         p_res = fmin(
    #             evaluate,
    #             config_space,
    #             optimizer=optimizer,
    #             n_jobs=n_jobs,
    #             n_iterations=n_iterations,
    #             parallel_strategy=parallel_strategy
    #         )
    #         for i in range(3):
    #             res = fmin(
    #                 evaluate,
    #                 config_space,
    #                 warm_start_strategy=warm_start_strategy,
    #                 n_jobs=n_jobs,
    #                 n_iterations=n_iterations,
    #                 previous_result=p_res,
    #                 parallel_strategy=parallel_strategy
    #             )
    #             p_res = res
    #             assert len(res["budget2obvs"][1]["losses"]) == n_iterations * (i + 2)

    def test_warm_start_multi_fidelity(self):
        optimizer = "ETPE"
        n_iterations = 15
        n_jobs = 3
        parallel_strategy = "AsyncComm"
        multi_fidelity_iter_generator = HyperBandIterGenerator(25, 100, 2)
        for warm_start_strategy in valid_warm_start_strategies:
            p_res = fmin(
                evaluate,
                config_space,
                optimizer=optimizer,
                n_jobs=n_jobs,
                n_iterations=n_iterations,
                parallel_strategy=parallel_strategy,
                multi_fidelity_iter_generator=multi_fidelity_iter_generator
            )
            print(p_res)
            for i in range(3):
                res = fmin(
                    evaluate,
                    config_space,
                    warm_start_strategy=warm_start_strategy,
                    n_jobs=n_jobs,
                    n_iterations=n_iterations,
                    previous_result=p_res,
                    parallel_strategy=parallel_strategy,
                    multi_fidelity_iter_generator=multi_fidelity_iter_generator
                )
                p_res = res
                print(p_res)
                assert len(res["budget2obvs"][25]["losses"]) == 20 * (i + 2)
