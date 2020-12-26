#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-20
# @Contact    : qichun.tang@bupt.edu.cn
import unittest
from ultraopt.hdl import HDL2CS, layering_config


# 测试嵌套依赖
class TestHDL(unittest.TestCase):
    def test_multi_choice(self):
        cs = HDL2CS()({
            "A(choice)": {
                "A1(choice)": {
                    "B1": {
                        "B1_h": {
                            "_type": "ordinal",
                            "_value": ["1", "2", "3"]
                        },
                    },
                    "B2": {
                        "B2_h": {
                            "_type": "choice",
                            "_value": ["1", "2", "3"]
                        },
                    },
                },
                "A2": {
                    "A2_h1": {
                        "_type": "choice",
                        "_value": ["1", "2", "3"]
                    },
                    "A2_h2": {
                        "_type": "uniform",
                        "_value": [-999, 666]
                    }

                },
                "A3": {
                    "A3_h1": {
                        "_type": "choice",
                        "_value": ["1", "2", "3"]
                    },
                    "A3_h2": {
                        "_type": "uniform",
                        "_value": [-999, 666]
                    }
                },
            }
        })
        print(cs)

    def test_conditional_space(self):
        # 测试conditional
        cs = HDL2CS()({
            "model(choice)": {
                "linearsvc": {
                    "max_iter": {"_type": "int_quniform", "_value": [300, 3000, 100], "_default": 600},
                    "penalty": {"_type": "choice", "_value": ["l1", "l2"], "_default": "l2"},
                    "dual": {"_type": "choice", "_value": [True, False], "_default": False},
                    "loss": {"_type": "choice", "_value": ["hinge", "squared_hinge"], "_default": "squared_hinge"},
                    "C": {"_type": "loguniform", "_value": [0.01, 10000], "_default": 1.0},
                    "multi_class": "ovr",
                    "random_state": 42,
                    "__forbidden": [
                        {"penalty": "l1", "loss": "hinge"},
                        {"penalty": "l2", "dual": False, "loss": "hinge"},
                        {"penalty": "l1", "dual": False},
                        {"penalty": "l1", "dual": True, "loss": "squared_hinge"},
                    ]
                },
                "svc": {
                    "C": {"_type": "loguniform", "_value": [0.01, 10000], "_default": 1.0},
                    "kernel": {"_type": "choice", "_value": ["rbf", "poly", "sigmoid"], "_default": "rbf"},
                    "degree": {"_type": "int_uniform", "_value": [2, 5], "_default": 3},
                    "gamma": {"_type": "loguniform", "_value": [1e-05, 8], "_default": 0.1},
                    "coef0": {"_type": "quniform", "_value": [-1, 1], "_default": 0},
                    "shrinking": {"_type": "choice", "_value": [True, False], "_default": True},
                    "class_weight": None,
                    "probability": True,
                    "decision_function_shape": "ovr",
                    "__activate": {
                        "kernel": {
                            "rbf": ["gamma"],
                            "sigmoid": ["gamma", "coef0"],
                            "poly": ["degree", "gamma", "coef0"]
                        }
                    },
                    "random_state": 42
                },
            }
        })
        print(cs)
        configs=cs.sample_configuration(200)
        res=[layering_config(config) for config in configs]
        print(len(res))
        print("OK")

