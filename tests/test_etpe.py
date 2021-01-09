#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-24
# @Contact    : qichun.tang@bupt.edu.cn
import unittest

import numpy as np

from ultraopt import fmin
from ultraopt.hdl import hdl2cs


def evaluate(config):
    return np.random.rand()


class TestETPE(unittest.TestCase):
    def test_conditions_and_fobidden(self):
        HDL = {
            "model(choice)": {
                "linearsvc": {
                    "max_iter": {"_type": "int_quniform", "_value": [300, 3000, 100], "_default": 600},
                    "penalty": {"_type": "choice", "_value": ["l1", "l2"], "_default": "l2"},
                    "dual": {"_type": "choice", "_value": [True, False], "_default": False},
                    "loss": {"_type": "choice", "_value": ["hinge", "squared_hinge"], "_default": "squared_hinge"},
                    "C": {"_type": "loguniform", "_value": [0.01, 10000], "_default": 1.0},
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
                    "__activate": {
                        "kernel": {
                            "rbf": ["gamma"],
                            "sigmoid": ["gamma", "coef0"],
                            "poly": ["degree", "gamma", "coef0"]
                        }
                    }
                },
                "mock": {
                    "C": {"_type": "loguniform", "_value": [0.01, 10000], "_default": 1.0},
                    "kernel": {"_type": "choice", "_value": ["rbf", "poly", "sigmoid"], "_default": "rbf"},
                    "degree": {"_type": "int_uniform", "_value": [2, 5], "_default": 3},
                    "gamma": {"_type": "loguniform", "_value": [1e-05, 8], "_default": 0.1},
                    "coef0": {"_type": "quniform", "_value": [-1, 1], "_default": 0},
                    "shrinking": {"_type": "choice", "_value": [True, False], "_default": True},
                    "__activate": {
                        "kernel": {
                            "rbf": ["gamma"],
                            "sigmoid": ["gamma", "coef0"],
                            "poly": ["degree", "gamma", "coef0"]
                        }
                    },
                    "max_iter": {"_type": "int_quniform", "_value": [300, 3000, 100], "_default": 600},
                    "penalty": {"_type": "choice", "_value": ["l1", "l2"], "_default": "l2"},
                    "dual": {"_type": "choice", "_value": [True, False], "_default": False},
                    "loss": {"_type": "choice", "_value": ["hinge", "squared_hinge"], "_default": "squared_hinge"},
                    "__forbidden": [
                        {"penalty": "l1", "loss": "hinge"},
                        {"penalty": "l2", "dual": False, "loss": "hinge"},
                        {"penalty": "l1", "dual": False},
                        {"penalty": "l1", "dual": True, "loss": "squared_hinge"},
                    ]
                },
            }
        }
        config_space = hdl2cs(HDL)
        fmin(evaluate, config_space, "ETPE", n_iterations=30)

    def test_multi_rest_config_space(self):
        HDL = {
            "feature_engineer(choice)": {
                "feature_selection(choice)": {
                    "wrapper(choice)": {
                        "RandomForest": {
                            "n_iterations": {"_type": "int_quniform", "_value": [10, 100, 10]},
                            "max_depth": {"_type": "int_quniform", "_value": [3, 7, 2]},
                        },
                        "LinearRegression": {
                            "C": {"_type": "loguniform", "_value": [0.01, 10000], "_default": 1.0},
                        },
                    },
                    "filter": {
                        "score_func": {"_type": "choice", "_value": ["pearsonr", "spearmanr"]}
                    }
                },
                "PolynomialFeatures": {
                    "degree": {"_type": "int_uniform", "_value": [2, 3]},
                    "interaction_only": {"_type": "choice", "_value": [True, False]},
                },
                "decomposition(choice)": {
                    "PCA": {
                        "n_components": {"_type": "uniform", "_value": [0.8, 0.95]},
                        "whiten": {"_type": "choice", "_value": [True, False]},
                    },
                    "KernelPCA": {
                        "n_components": {"_type": "uniform", "_value": [0.8, 0.95]},
                        "whiten": {"_type": "choice", "_value": [True, False]},
                    },
                    "ICA": {}
                }
            }
        }
        config_space = hdl2cs(HDL)
        fmin(evaluate, config_space, "ETPE", n_iterations=30)
