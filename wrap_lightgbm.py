#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

import logging

import lightgbm
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target

logger = logging.getLogger(__name__)


class LGBMEstimator(BaseEstimator):
    is_classification = None

    def __init__(
            self,
            n_estimators=2000,
            objective=None,
            boosting_type="gbdt",
            # objective="binary",
            learning_rate=0.01,
            max_depth=31,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            random_state=0,
            # cat_smooth=35,
            lambda_l1=0.1,
            lambda_l2=0.2,
            subsample_for_bin=40000,
            # min_data_in_leaf=4,
            min_child_weight=0.01,
            early_stopping_rounds=250,
            verbose=-1,
            n_jobs=1,
            warm_start=True
    ):
        self.warm_start = warm_start
        assert self.is_classification is not None, NotImplementedError
        self.n_jobs = n_jobs
        self.objective = objective
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.min_child_weight = min_child_weight
        self.subsample_for_bin = subsample_for_bin
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.random_state = random_state
        self.bagging_freq = bagging_freq
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.boosting_type = boosting_type
        self.n_estimators = n_estimators
        self.model = None
        self.current_iterations = 0
        self.early_stopped = False

    def fit(self, X, y, X_valid=None, y_valid=None, categorical_feature="auto",
            sample_weight=None, **kwargs):
        X = check_array(X)
        y = check_array(y, ensure_2d=False, dtype="float")
        if X_valid is not None:
            X_valid = check_array(X_valid)
        if y_valid is not None:
            y_valid = check_array(y_valid, ensure_2d=False, dtype="float")
        if self.objective is None:
            if self.is_classification:
                target_type = type_of_target(y)
                if target_type == "binary":
                    self.objective = "binary"
                elif target_type == "multiclass":
                    self.objective = "multiclass"
                else:
                    raise ValueError(f"Invalid target_type {target_type}!")
            else:
                self.objective = "regression"
        param = dict(
            verbose=-1,
            boosting_type=self.boosting_type,
            objective=self.objective,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            random_state=self.random_state,
            lambda_l1=self.lambda_l1,
            lambda_l2=self.lambda_l2,
            subsample_for_bin=self.subsample_for_bin,
            min_child_weight=self.min_child_weight,
            num_threads=self.n_jobs
        )
        if self.objective == "multiclass":
            param.update({"num_class": int(np.max(y) + 1)})
        num_boost_round = self.n_estimators - self.current_iterations
        if num_boost_round <= 0:
            logger.warning(f"num_boost_round = {num_boost_round}, <=0, "
                           f"n_estimators = {self.n_estimators}, "
                           f"current_iterations = {self.current_iterations}")
            return self
        if self.early_stopped:
            logger.info(
                f"{self.__class__.__name__} is early_stopped, best_iterations = {self.model.best_iteration}")
            return self
        train_set = lightgbm.Dataset(
            X, y,
            categorical_feature=categorical_feature,
            weight=sample_weight)
        valid_sets = [train_set]
        if X_valid is not None and y_valid is not None:
            valid_sets.append(lightgbm.Dataset(X_valid, y_valid))
        else:
            self.early_stopping_rounds = None
        self.model = lightgbm.train(
            param,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose,
            init_model=self.model,
            **kwargs
        )
        self.current_iterations = self.n_estimators
        if y_valid is not None and getattr(self.model, "best_iteration", np.inf) < self.n_estimators:
            self.early_stopped = True
        return self

    def predict(self, X):
        X = check_array(X)
        if self.is_classification:
            return self.predict_proba(X).argmax(axis=1)
        else:
            return self.model.predict(X, num_iteration=self.model.best_iteration)

    def predict_proba(self, X):
        X = check_array(X)
        y_prob = self.model.predict(X, num_iteration=self.model.best_iteration)
        if self.objective == "binary":
            y_prob = y_prob[:, None]
            y_prob = np.hstack([1 - y_prob, y_prob])
        return y_prob


class LGBMClassifier(LGBMEstimator, ClassifierMixin):
    is_classification = True


class LGBMRegressor(LGBMEstimator, RegressorMixin):
    is_classification = False

    def predict_proba(self, X):
        raise NotImplementedError
