#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
import warnings
from copy import deepcopy
from time import time
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

from tabular_nn.tabular_nn import TabularNNTrainer, TabularNN
from tabular_nn.utils.logging_ import get_logger

warnings.simplefilter(action='ignore', category=FutureWarning)

PER_RUN_TIME_LIMIT = 300  # 5 minutes


class TabularNNEstimator(BaseEstimator):
    is_classification = None

    # todo: constraint cpu usage
    def __init__(
            self,
            max_layer_width=2056,
            min_layer_width=32,
            dropout_hidden=0.1,
            dropout_output=0.2,
            af_hidden="relu",
            af_output="linear",
            layer1=256,
            layer2=128,
            use_bn=True,
            lr=1e-2,
            max_epoch=128,
            random_state=1000,
            batch_size=1024,
            optimizer="adam",
            early_stopping_rounds=16,
            early_stopping_tol=0,
            verbose=-1,
            n_jobs=-1,
            class_weight=None,
            normalize=True,
            per_run_time_limit=PER_RUN_TIME_LIMIT
    ):
        self.per_run_time_limit = per_run_time_limit
        assert self.is_classification is not None, NotImplementedError
        self.normalize = normalize
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.early_stopping_tol = early_stopping_tol
        self.early_stopping_rounds = early_stopping_rounds
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.random_state = random_state
        self.max_epoch = max_epoch
        self.lr = lr
        self.use_bn = use_bn
        self.layer2 = layer2
        self.layer1 = layer1
        self.dropout_output = dropout_output
        self.af_output = af_output
        self.af_hidden = af_hidden
        self.dropout_hidden = dropout_hidden
        self.min_layer_width = min_layer_width
        self.max_layer_width = max_layer_width
        self.init_variables()
        if self.is_classification:
            n_class = None
        else:
            n_class = 1
        self.nn_param = dict(
            use_bn=self.use_bn,
            dropout_output=self.dropout_output,
            dropout_hidden=self.dropout_hidden,
            layers=(self.layer1, self.layer2),
            af_hidden=self.af_hidden,
            af_output=self.af_output,
            max_layer_width=self.max_layer_width,
            min_layer_width=self.min_layer_width
        )
        self.tabular_nn_trainer = TabularNNTrainer(
            lr=self.lr,
            max_epoch=self.max_epoch,
            n_class=n_class,
            nn_params=self.nn_param,
            random_state=self.rng,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight
        )

    def init_variables(self):
        self.scaler = StandardScaler(copy=True)
        self.rng = np.random.RandomState(self.random_state)
        self.logger = get_logger(self)
        self.model = None
        self.learning_curve = [
            [],  # train_sizes_abs [0]
            [],  # train_scores    [1]
            [],  # test_scores     [2]
        ]
        self.performance_history = np.full(self.early_stopping_rounds, -np.inf)
        self.iteration_history = np.full(self.early_stopping_rounds, 0, dtype="int32")
        N = len(self.performance_history)
        self.best_estimators = np.zeros([N], dtype="object")
        if self.is_classification:
            self.score_func = accuracy_score
        else:
            self.score_func = r2_score
        self.early_stopped = False
        self.time_limit_early_stopped = False
        self.tles_states = []
        self.best_iteration = 0

    def fit(self, X, y, X_valid=None, y_valid=None, categorical_feature: Optional[List[int]] = None):
        if self.early_stopped:
            return self
        X = check_array(X)
        y = check_array(y, ensure_2d=False, dtype="float")
        if self.normalize and (not self.is_classification):
            self.scaler.fit(y[:, None])
            y = self.scaler.transform(y[:, None]).flatten()
            if y_valid is not None:
                y_valid = self.scaler.transform(y_valid[:, None]).flatten()
        if X_valid is not None:
            X_valid = check_array(X_valid)
        if y_valid is not None:
            y_valid = check_array(y_valid, ensure_2d=False, dtype="float")

        if categorical_feature is not None:
            cat_indexes = check_array(categorical_feature, ensure_2d=False, dtype="int", ensure_min_samples=0)
        else:
            cat_indexes = np.array([])
        if self.best_estimators is None:
            self.init_variables()
        self.tabular_nn_trainer.max_epoch = self.max_epoch
        self.start_time = time()
        self.model = self.tabular_nn_trainer.train(
            self.model, TabularNN, X, y, X_valid, y_valid,
            self.callback, cat_indexes=cat_indexes
        )
        if self.early_stopped:
            index = int(np.lexsort((self.iteration_history, -self.performance_history))[0])
            self.best_iteration = int(self.iteration_history[index]) + 1
            best_estimator = self.best_estimators[index]
            self.model = best_estimator
            self.logger.info(f"{self.__class__.__name__} is early_stopped, "
                             f"best_iteration = {self.best_iteration}, "
                             f"best_performance in validation_set = {self.performance_history[index]:.3f}")
            self.best_estimators = None  # do not train any more
        else:
            if not self.time_limit_early_stopped:
                self.best_iteration = self.max_epoch
            else:
                self.best_iteration = int(self.iteration_history.max()) + 1
        # stash state
        self.tles_states.append(self.time_limit_early_stopped)
        # reset state
        self.time_limit_early_stopped = False
        return self

    def predict(self, X):
        y_pred = self._predict(self.model, X)
        if self.normalize and (not self.is_classification):
            return self.scaler.inverse_transform(y_pred)
        return y_pred

    def _predict(self, model, X):
        X = check_array(X)
        y_pred = model(X).detach().numpy()
        if self.is_classification:
            y_pred = y_pred.argmax(axis=1)
            return y_pred
        return y_pred

    def callback(self, epoch_index, model, X, y, X_valid, y_valid) -> bool:
        if self.early_stopped:
            return self.early_stopped
        model.eval()
        # todo: 用 logging_level  来控制
        should_print = self.verbose > 0 and epoch_index % self.verbose == 0
        train_score = self.score_func(y, self._predict(model, X))
        can_early_stopping = True
        if X_valid is not None:
            valid_score = self.score_func(y_valid, self._predict(model, X_valid))
        else:
            valid_score = None
            can_early_stopping = False
        score_func_name = self.score_func.__name__
        msg = f"epoch_index = {epoch_index}, " \
            f"TrainSet {score_func_name} = {train_score:.3f}"
        if valid_score is not None:
            msg += f", ValidSet {score_func_name} = {valid_score:.3f} . "
        if should_print:
            self.logger.info(msg)
        else:
            self.logger.debug(msg)
        self.learning_curve[0].append(epoch_index)
        self.learning_curve[1].append(train_score)
        self.learning_curve[2].append(valid_score)
        if can_early_stopping:
            if np.any(valid_score - self.early_stopping_tol > self.performance_history):
                index = epoch_index % self.early_stopping_rounds
                self.best_estimators[index] = deepcopy(model)
                self.performance_history[index] = valid_score
                self.iteration_history[index] = epoch_index
            else:
                self.early_stopped = True
        if time() - self.start_time > self.per_run_time_limit:
            self.time_limit_early_stopped = True
            self.best_iteration = epoch_index
        return self.early_stopped or self.time_limit_early_stopped


class TabularNNClassifier(TabularNNEstimator, ClassifierMixin):
    is_classification = True

    def predict_proba(self, X):
        X = check_array(X)
        return self.model(X).detach().numpy()


class TabularNNRegressor(TabularNNEstimator, RegressorMixin):
    is_classification = False
