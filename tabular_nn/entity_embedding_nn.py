#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
from copy import deepcopy
from itertools import chain

import numpy as np
import torch
from torch import nn

from tabular_nn.base_tnn import BaseTNN, get_embed_dims
from tabular_nn.trainer import Trainer
from tabular_nn.utils.logging_ import get_logger


class EntityEmbeddingNN(BaseTNN):
    def __init__(
            self,
            n_uniques: np.ndarray,
            A=10, B=5,
            dropout1=0.1,
            dropout2=0.1,
            dropout3=0.1,
            n_class=2
    ):
        super(EntityEmbeddingNN, self).__init__()
        self.dropout3 = dropout3
        self.logger = get_logger(self)
        self.epoch = 0
        self.n_class = n_class
        self.dropout2 = dropout2
        self.dropout1 = dropout1
        self.n_uniques = n_uniques
        self.A = A
        self.B = B
        self.embed_dims = get_embed_dims(n_uniques)
        sum_ = np.log(self.embed_dims).sum()
        self.n_layer1 = min(1000,
                            int(A * (n_uniques.size ** 0.5) * sum_ + 1))
        self.n_layer2 = int(self.n_layer1 / B) + 2
        self.embedding_blocks = nn.ModuleList([
            nn.Embedding(int(n_unique), int(embed_dim))
            for n_unique, embed_dim in zip(self.n_uniques, self.embed_dims)
        ])
        embed_dims_size = self.embed_dims.sum()
        layer1 = self.get_block(embed_dims_size, self.n_layer1, False, dropout1, "leaky_relu")
        layer2 = self.get_block(self.n_layer1, self.n_layer2, False, dropout2, "leaky_relu")
        layer3 = self.get_block(self.n_layer2, self.n_class, False, dropout3, "leaky_relu")
        self.deep_net = nn.Sequential(
            layer1,
            layer2,
            layer3
        )
        self.wide_net = self.get_block(embed_dims_size, self.n_class, False, dropout3, "leaky_relu")
        output_modules = []
        # if self.n_class > 1:
        #     output_modules.append(nn.Softmax(dim=1))
        self.output_layer = nn.Sequential(*output_modules)
        self.initializing_modules(chain(
            self.deep_net.modules(), self.wide_net.modules(),
            self.output_layer.modules(), self.embedding_blocks.modules()
        ))

    def forward(self, X: np.ndarray):
        embeds = []
        if self.embedding_blocks is not None:
            for i in range(X.shape[1]):
                col_vec = deepcopy(X[:, i])
                col_vec[col_vec >= self.n_uniques[i]] = 0
                embeds.append(
                    self.embedding_blocks[i](torch.from_numpy(col_vec.astype("int64")))
                )
        cat_embeds = torch.cat(embeds, dim=1)
        outputs = self.deep_net(cat_embeds) + self.wide_net(cat_embeds)
        activated = self.output_layer(outputs)
        return embeds, activated


class TrainEntityEmbeddingNN(Trainer):
    def get_output(self, model, array):
        _, activated = model(array)
        return activated

    def parsing_kwargs(self, X, y, kwargs):
        self.n_uniques = kwargs.get("n_uniques")

    def instancing_nn(self, nn_cls, nn_params, n_class):
        return nn_cls(n_class=n_class, n_uniques=self.n_uniques, **nn_params)
