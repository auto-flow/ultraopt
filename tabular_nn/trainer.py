#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
from math import ceil
from time import time
from typing import Optional, Callable

import numpy as np
import torch
from frozendict import frozendict
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from torch import nn
from torch.nn.functional import cross_entropy, mse_loss

from tabular_nn.utils.data import check_n_jobs
from tabular_nn.utils.logging_ import get_logger


class Trainer():
    def __init__(
            self,
            lr=1e-2, max_epoch=25, n_class=None, nn_params=frozendict(),
            random_state=1000, batch_size=1024, optimizer="adam", n_jobs=-1,
            class_weight=None
    ):
        self.class_weight = class_weight
        self.n_jobs = check_n_jobs(n_jobs)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.random_state = random_state
        self.nn_params = nn_params
        self.n_class = n_class
        self.max_epoch = max_epoch
        self.lr = lr
        self.rng = check_random_state(random_state)
        self.logger = get_logger(self)

    def train(
            self,
            init_model, nn_cls,
            X: np.ndarray,
            y: np.ndarray,
            X_valid: Optional[np.ndarray] = None,
            y_valid: Optional[np.ndarray] = None,
            callback: Optional[Callable[[int, nn.Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray], bool]] = None,
            **kwargs
    ):
        torch.manual_seed(self.rng.randint(0, 10000))
        torch.set_num_threads(self.n_jobs)
        self.parsing_kwargs(X, y, kwargs)
        if self.n_class is None:
            if type_of_target(y.astype("float")) == "continuous":
                self.n_class = 1
            else:
                self.n_class = np.unique(y).size

        nn_params = dict(self.nn_params)
        if init_model is None:
            tnn: nn.Module = self.instancing_nn(
                nn_cls, nn_params, self.n_class)
        else:
            tnn = init_model
        if self.optimizer == "adam":
            nn_optimizer = torch.optim.Adam(tnn.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            nn_optimizer = torch.optim.SGD(tnn.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
        start_time = time()
        if self.n_class >= 2:
            y_tensor = torch.from_numpy(y).long()
        else:
            y_tensor = torch.from_numpy(y).double()
        if self.n_class >= 2 and self.class_weight == "balanced":
            # fixme: am I right?
            unique, counts = np.unique(y, return_counts=True)
            counts = counts / np.min(counts)
            weight = torch.from_numpy(1 / counts).double()
        else:
            weight = None
        init_epoch = getattr(tnn, "max_epoch", 0)
        for epoch_index in range(init_epoch, self.max_epoch):
            tnn.train(True)
            # batch
            permutation = self.rng.permutation(len(y))
            batch_ixs = []
            for i in range(ceil(len(y) / self.batch_size)):
                start = min(i * self.batch_size, len(y))
                end = min((i + 1) * self.batch_size, len(y))
                batch_ix = permutation[start:end]
                if end - start < self.batch_size:
                    diff = self.batch_size - (end - start)
                    diff = min(diff, start)
                    batch_ix = np.hstack([batch_ix, self.rng.choice(permutation[:start], diff, replace=False)])
                batch_ixs.append(batch_ix)
            for batch_ix in batch_ixs:
                nn_optimizer.zero_grad()
                outputs = self.get_output(tnn, X[batch_ix, :])
                if self.n_class >= 2:
                    loss = cross_entropy(outputs.double(), y_tensor[batch_ix], weight=weight)
                elif self.n_class == 1:
                    loss = mse_loss(outputs.flatten().double(), y_tensor[batch_ix])
                else:
                    raise ValueError
                loss.backward()
                nn_optimizer.step()
            tnn.max_epoch = epoch_index + 1
            if callback is not None:
                if callback(epoch_index, tnn, X, y, X_valid, y_valid) == True:
                    break
        end = time()
        self.logger.info(f"{tnn.__class__.__name__} training time = {end - start_time:.2f}s")
        tnn.eval()
        return tnn

    def instancing_nn(self, nn_cls, nn_params, n_class):
        return nn_cls(n_class=n_class, **nn_params)

    def parsing_kwargs(self, X, y, kwargs):
        pass

    def get_output(self, model, array):
        return model(array)
