#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn

import os

import numpy as np
from torch import nn


def get_embed_dims(n_uniques):
    exp_ = np.exp(-n_uniques * 0.05)
    ans = np.round(5 * (1 - exp_) + 1).astype("int")
    max_dim = int(os.getenv('MAX_DIM', '100'))
    return np.clip(ans, None, max_dim)


def get_ord_embed_dims(n_uniques):
    return np.ones(n_uniques.shape, dtype='int')


class BaseTNN(nn.Module):
    def __init__(self):
        super(BaseTNN, self).__init__()

    def get_embedding_blocks(self, n_uniques, embed_dims):
        return nn.ModuleList([
            nn.Embedding(int(n_unique), int(embed_dim))
            for n_unique, embed_dim in zip(n_uniques, embed_dims)
        ])

    def get_reg_loss(self, loss):
        return loss

    def get_activate_function(self, af_name: str):
        af_name = af_name.lower()
        if af_name == "relu":
            return nn.ReLU(inplace=True)
        elif af_name == "leaky_relu":
            return nn.LeakyReLU(inplace=True)
        elif af_name == "elu":
            return nn.ELU(inplace=True)
        elif af_name == "linear":
            return nn.Identity()
        elif af_name == "tanh":
            return nn.Tanh()
        elif af_name == "sigmoid":
            return nn.Sigmoid()
        elif af_name == "softplus":
            return nn.Softplus()
        else:
            raise ValueError(f"Unknown activate function name {af_name}")

    def get_block(self, in_features, out_features, use_bn, dropout_rate, af_name):
        seq = []
        seq.append(nn.Linear(in_features, out_features))
        if use_bn:
            seq.append(nn.BatchNorm1d(out_features))
        seq.append(self.get_activate_function(af_name))
        if dropout_rate > 0:
            seq.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*seq)

    def initializing_modules(self, modules):
        for m in modules:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                m.bias.data.zero_()
