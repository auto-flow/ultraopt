#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
from math import ceil
from time import time
from typing import Optional

import numpy as np
import torch
from frozendict import frozendict
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import check_random_state
from tabular_nn.base_tnn import BaseTNN
from tabular_nn.utils.data import check_n_jobs
from tabular_nn.utils.logging_ import get_logger
from torch.nn.functional import cross_entropy, mse_loss


class Trainer():
    def __init__(
            self,
            lr=1e-2, max_epoch=25, nn_params=frozendict(),
            random_state=1000, batch_size=1024, optimizer="adam", n_jobs=-1,
            class_weight=None, verbose=0., weight_decay=5e-4
    ):
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.class_weight = class_weight
        self.n_jobs = check_n_jobs(n_jobs)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.random_state = random_state
        self.nn_params = nn_params
        self.max_epoch = max_epoch
        self.lr = lr
        self.rng = check_random_state(random_state)
        self.logger = get_logger(self)

    def train(
            self,
            init_model,
            X: np.ndarray,
            y: np.ndarray,
            X_valid: Optional[np.ndarray] = None,  # todo: 有机会搞这个？ 或者交叉验证放在这个模块里面搞？
            y_valid: Optional[np.ndarray] = None,
            label_reg_mask=None,
            clf_n_classes=(0,),
    ):
        if y.ndim == 1:
            y = y[:, np.newaxis]
        torch.manual_seed(self.rng.randint(0, 10000))
        torch.set_num_threads(self.n_jobs)
        tnn: BaseTNN = init_model
        if self.optimizer == "adam":
            nn_optimizer = torch.optim.Adam(tnn.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            nn_optimizer = torch.optim.SGD(tnn.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
        start_time = time()
        reg_label = y[:, label_reg_mask]
        clf_label = y[:, ~label_reg_mask]
        n_reg_label = reg_label.shape[1]
        n_clf_label = clf_label.shape[1]
        if reg_label.shape[0] == 0:
            reg_label = None
        else:
            reg_label = torch.from_numpy(reg_label).float()
        if clf_label.shape[0] == 0:
            clf_label = None
        else:
            clf_label = torch.from_numpy(clf_label).long()
        if n_clf_label and self.class_weight == "balanced":
            # fixme: am I right?
            unique, counts = np.unique(y, return_counts=True)
            counts = counts / np.min(counts)
            weight = torch.from_numpy(1 / counts).double()
        else:
            weight = None
        init_epoch = getattr(tnn, "max_epoch", 0)
        tnn.train(True)
        for epoch_index in range(init_epoch, self.max_epoch):
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
                outputs = self.get_output(tnn, X[batch_ix, :]).float()
                loss = torch.tensor(0).float()
                label_ix = 0
                for k in range(n_clf_label):
                    output_slice = outputs[:, label_ix: label_ix + clf_n_classes[k]]
                    loss += cross_entropy(
                        output_slice,
                        clf_label[batch_ix, k],
                        weight=weight)
                    label_ix += clf_n_classes[k]
                for k in range(n_reg_label):
                    loss += mse_loss(outputs[:, label_ix].flatten(), reg_label[batch_ix, k])
                tnn.get_reg_loss(loss)
                loss.backward()
                nn_optimizer.step()
            tnn.max_epoch = epoch_index + 1
            # ------------------------------------------------------------------------------------
            tnn.eval()
            outputs = self.get_output(tnn, X).float()
            outputs = outputs.detach().numpy()
            label_ix = 0
            acc_scores = []
            r2_scores = []
            msg = f"epoch_index = {epoch_index:02d}, acc = ["
            for k in range(n_clf_label):
                output_slice = outputs[:, label_ix: label_ix + clf_n_classes[k]]
                y_pred = np.argmax(output_slice, axis=1)
                y_true = clf_label[:, k].detach().numpy()
                score = accuracy_score(y_true, y_pred)
                label_ix += clf_n_classes[k]
                acc_scores.append(score)
                msg += f"{score:.3f} "
            msg += "], r2 = ["
            for k in range(n_reg_label):
                y_pred = outputs[:, label_ix]
                y_true = reg_label[:, k].detach().numpy()
                score = r2_score(y_true, y_pred)
                label_ix += 1
                r2_scores.append(score)
                msg += f"{score:.3f} "
            msg += "]"
            should_print = self.verbose > 0 and epoch_index % self.verbose == 0
            if should_print:
                # self.logger.info(msg)
                print(msg)

        end = time()
        self.logger.info(f"{tnn.__class__.__name__} training time = {end - start_time:.2f}s")
        tnn.eval()
        return tnn

    def get_output(self, model, array):
        return model(array)
