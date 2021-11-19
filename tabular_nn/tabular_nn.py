#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
from copy import deepcopy
from typing import List, Union

import numpy as np
import torch
from torch import nn

from tabular_nn.base_tnn import BaseTNN, get_embed_dims
from tabular_nn.trainer import Trainer
from tabular_nn.utils.logging_ import get_logger


class TabularNN(BaseTNN):
    def __init__(
            self,
            n_uniques: np.ndarray,
            vector_dim: int,
            cat_indexes: Union[List[int], np.ndarray],
            max_layer_width=2056,
            min_layer_width=32,
            dropout_hidden=0.1,
            af_hidden="relu",
            af_output="linear",
            dropout_output=0.2,
            layers=(256, 128),
            n_class=2,
            use_bn=True
    ):
        super(TabularNN, self).__init__()
        self.logger = get_logger(self)
        self.af_output = af_output
        self.af_hidden = af_hidden
        self.max_epoch = 0
        self.use_bn = use_bn
        assert len(cat_indexes) == len(n_uniques)
        self.layers = layers
        self.min_layer_width = min_layer_width
        self.max_layer_width = max_layer_width
        self.cat_indexes = np.array(cat_indexes, dtype="int")
        self.n_class = n_class
        self.dropout_output = dropout_output
        self.dropout_hidden = dropout_hidden
        self.n_uniques = n_uniques
        num_features = len(n_uniques) + vector_dim
        prop_vector_features = vector_dim / num_features
        msg = ""
        if vector_dim > 0:
            numeric_embed_dim = int(np.clip(
                round(layers[0] * prop_vector_features * np.log10(vector_dim + 10)),
                min_layer_width, max_layer_width
            ))
            msg += f"numeric_block = {vector_dim}->{numeric_embed_dim}; "
            self.numeric_block = nn.Sequential(
                nn.Linear(vector_dim, numeric_embed_dim),
                self.get_activate_function(self.af_hidden)
            )
        else:
            numeric_embed_dim = 0
            msg += f"numeric_block is None; "
            self.numeric_block = None
        if len(n_uniques) > 0:
            self.embed_dims = get_embed_dims(self.n_uniques)
            self.embedding_blocks = self.get_embedding_blocks(self.n_uniques, self.embed_dims)
            emb_arch = ', '.join(
                [f'{n_unique:d}->{embed_dim:d}' for n_unique, embed_dim in zip(self.n_uniques, self.embed_dims)])
            msg += f"embedding_blocks = {emb_arch}; "
        else:
            msg += f"embedding_blocks is None; "
            self.embed_dims = np.array([])
            self.embedding_blocks = None
        after_embed_dim = int(self.embed_dims.sum() + numeric_embed_dim)
        deep_net_modules = []
        layers_ = [after_embed_dim] + list(layers)
        layers_len = len(layers_)
        for i in range(1, layers_len):
            in_features = layers_[i - 1]
            out_features = layers_[i]
            msg += f"layer{i} = {in_features}->{out_features}; "
            dropout_rate = self.dropout_hidden
            block = self.get_block(
                in_features, out_features,
                use_bn=self.use_bn, dropout_rate=dropout_rate, af_name=self.af_hidden)
            deep_net_modules.append(block)
        self.logger.info(msg)
        deep_net_modules.append(
            self.get_block(
                layers_[-1], self.n_class,
                self.use_bn, dropout_rate=self.dropout_output, af_name=self.af_output
            ))
        self.deep_net = nn.Sequential(*deep_net_modules)
        self.wide_net = self.get_block(
            after_embed_dim, n_class,
            use_bn=self.use_bn, dropout_rate=self.dropout_output, af_name=self.af_output
        )
        output_modules = []
        # 分类任务的logits不需要softmax
        # if self.n_class > 1:
        #     output_modules.append(nn.Softmax(dim=1))
        self.output_layer = nn.Sequential(*output_modules)
        modules = [
            self.deep_net.modules(),
            self.wide_net.modules(),
            self.output_layer.modules(),
        ]
        if self.embedding_blocks is not None:
            modules.append(self.embedding_blocks)
        if self.numeric_block is not None:
            modules.append(self.numeric_block)
        self.initializing_modules(modules)



    def forward(self, X: np.ndarray):
        embeds = []
        if self.embedding_blocks is not None:
            for i, col in enumerate(self.cat_indexes):
                col_vec = deepcopy(X[:, col])
                col_vec[col_vec >= self.n_uniques[i]] = 0
                embeds.append(
                    self.embedding_blocks[i](torch.from_numpy(col_vec.astype("int64")))
                )
        num_indexed = np.setdiff1d(np.arange(X.shape[1]), self.cat_indexes)
        if self.numeric_block is not None:
            embeds.append(self.numeric_block(torch.from_numpy(X[:, num_indexed].astype("float32"))))
        cat_embeds = torch.cat(embeds, dim=1)
        outputs = self.deep_net(cat_embeds) + self.wide_net(cat_embeds)
        activated = self.output_layer(outputs)
        return activated


class TabularNNTrainer(Trainer):
    def parsing_kwargs(self, X, y, kwargs):
        cat_indexes = kwargs.get("cat_indexes")
        cat_indexes = np.array(cat_indexes, dtype="int")
        self.n_uniques = (X[:, cat_indexes].max(axis=0) + 1).astype("int")
        self.vector_dim = X.shape[1] - len(cat_indexes)
        self.cat_indexes = cat_indexes

    def instancing_nn(self, nn_cls, nn_params, n_class):
        return nn_cls(
            self.n_uniques, self.vector_dim, self.cat_indexes,
            n_class=n_class,
            **nn_params
        )
