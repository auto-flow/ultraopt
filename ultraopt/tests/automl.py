#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-02
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np
import sklearn.metrics
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold  # 采用分层抽样
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from ultraopt.hdl import layering_config, hdl2cs

_ = [LinearSVC, RandomForestClassifier, KNeighborsClassifier, sklearn.metrics]
__all__ = ["HDL", "evaluator", "config_space"]

HDL = {
    'classifier(choice)': {
        "LinearSVC": {
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
        "RandomForestClassifier": {
            "n_estimators": {"_type": "int_quniform", "_value": [10, 200, 10], "_default": 100},
            "criterion": {"_type": "choice", "_value": ["gini", "entropy"], "_default": "gini"},
            "max_features": {"_type": "choice", "_value": ["sqrt", "log2"], "_default": "sqrt"},
            "min_samples_split": {"_type": "int_uniform", "_value": [2, 20], "_default": 2},
            "min_samples_leaf": {"_type": "int_uniform", "_value": [1, 20], "_default": 1},
            "bootstrap": {"_type": "choice", "_value": [True, False], "_default": True},
            "random_state": 42
        },
        "KNeighborsClassifier": {
            "n_neighbors": {"_type": "int_loguniform", "_value": [1, 100], "_default": 3},
            "weights": {"_type": "choice", "_value": ["uniform", "distance"], "_default": "uniform"},
            "p": {"_type": "choice", "_value": [1, 2], "_default": 2},
        },
    }
}

config_space = hdl2cs(HDL)

default_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


class Evaluator():
    def __init__(self,
                 X, y,
                 metric="accuracy",
                 cv=default_cv):
        # 初始化
        self.X = X
        self.y = y
        self.metric = metric
        self.cv = cv

    def __call__(self, config: dict, budget: float = 1) -> float:
        layered_dict = layering_config(config)
        AS_HP = layered_dict['classifier'].copy()
        AS, HP = AS_HP.popitem()
        ML_model = eval(AS)(**HP)
        # scores = cross_val_score(ML_model, self.X, self.y, cv=self.cv, scoring=self.metric)
        # -------------------------------------------------------------
        # 采用在对【 5折交叉验证中的训练集 】进行采样的方法，采样率为 budget
        sample_ratio = budget
        scores = []
        for i, (train_ix, valid_ix) in enumerate(self.cv.split(X, y)):
            rng = np.random.RandomState(i)
            size = int(train_ix.size * sample_ratio)
            train_ix = rng.choice(train_ix, size, replace=False)
            X_train = X[train_ix, :]
            y_train = y[train_ix]
            X_valid = X[valid_ix, :]
            y_valid = y[valid_ix]
            ML_model.fit(X_train, y_train)
            y_pred = ML_model.predict(X_valid)
            score = eval(f"sklearn.metrics.{self.metric}_score")(y_valid, y_pred)
            scores.append(score)
        # -------------------------------------------------------------
        score = np.mean(scores)
        return 1 - score


X, y = load_digits(return_X_y=True)
evaluator = Evaluator(X, y)
