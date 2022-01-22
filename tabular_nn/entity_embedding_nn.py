#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
import os
from itertools import chain

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from tabular_nn.base_tnn import BaseTNN, get_embed_dims
from tabular_nn.trainer import Trainer
from tabular_nn.utils.logging_ import get_logger
from torch import nn


class LearnableScaler(nn.Module):
    def __init__(self, input, momentum=0.9):
        super(LearnableScaler, self).__init__()
        self.momentum = momentum
        self.insize = input
        self.weight = nn.Parameter(torch.ones(self.insize))
        self.running_mean = torch.zeros(self.insize)
        self.init = True

    def forward(self, input):
        if self.training:
            mean = torch.mean(input, dim=0)
            if self.init:
                self.running_mean = mean
                self.init = False
            else:
                self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            # get output
        else:
            mean = self.running_mean
        y = self.weight * (input - mean)
        return y


def _n_false(n, m):
    '''https://www.thinbug.com/q/55330169'''
    return np.array([False] * n + [True] * (m - n))


n_false = np.vectorize(_n_false, signature='(),()->(n)')


class OrdinalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None, momentum=0.9):
        super(OrdinalEmbedding, self).__init__(
            num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
            scale_grad_by_freq, sparse, _weight)
        self.momentum = momentum
        self.running_mean = (torch.zeros(embedding_dim))
        self.init = True
        # self.loss = torch.tensor(0, requires_grad=True,dtype=torch.float)

    def forward(self, input: torch.Tensor):
        assert input.ndimension() == 1
        N = input.shape[0]
        K = self.num_embeddings
        M = self.embedding_dim
        # [N, K, M]
        repeated_arrange = torch.arange(K, dtype=torch.long)[None, :].repeat(N, 1)
        all_emb = super(OrdinalEmbedding, self).forward(repeated_arrange)
        # return all_emb.sum(dim=1)
        # return super(OrdinalEmbedding, self).forward(input)
        self.loss = torch.tensor(0, dtype=torch.float)
        if os.getenv('ORD_EMB_REG') == 'abs':
            all_emb = all_emb.abs_()
        if os.getenv('ORD_EMB_REG') == 'cos':
            K_emb = super(OrdinalEmbedding, self).forward(torch.arange(K, dtype=torch.long))  # [K, M]
            ones = torch.ones_like(K_emb)
            cosine_similarity = torch.cosine_similarity(K_emb, ones.detach(), dim=1)
            loss = torch.max(-cosine_similarity + 0.75, torch.zeros_like(cosine_similarity))
            loss = torch.sum(loss)
            self.loss = loss
        if os.getenv('ORD_EMB_REG') == 'abs_loss':
            K_emb = super(OrdinalEmbedding, self).forward(torch.arange(K, dtype=torch.long))  # [K, M]
            loss = torch.max(-K_emb, torch.zeros_like(K_emb)) + \
                   torch.max(0.1 - torch.abs(K_emb), torch.zeros_like(K_emb))
            loss = torch.sum(loss)
            self.loss = loss
        mask = repeated_arrange >= input[:, None].repeat(1, K)
        mask = mask[:, :, None].repeat([1, 1, M])
        all_emb[mask] = 0
        out = torch.sum(all_emb, dim=1)
        if self.training:
            mean = torch.mean(out, dim=0)
            if self.init:
                self.running_mean = mean
                self.init = False
            else:
                self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            # get output
        else:
            mean = self.running_mean
        out -= mean.detach()
        return out


class EntityEmbeddingNN(BaseTNN):
    def __init__(
            self,
            n_choices_list: np.ndarray,
            n_sequences_list: np.ndarray,
            n_cont_variables: int,
            A=10, B=5,
            dropout1=0.1,
            dropout2=0.1,
            dropout3=0.1,
            n_class=2
    ):
        super(EntityEmbeddingNN, self).__init__()
        self.n_cat_variables = len(n_choices_list)
        self.n_ord_variables = len(n_sequences_list)
        self.n_sequences_list = n_sequences_list
        self.n_cont_variables = n_cont_variables
        self.dropout3 = dropout3
        self.logger = get_logger(self)
        self.epoch = 0
        self.n_class = n_class
        self.dropout2 = dropout2
        self.dropout1 = dropout1
        self.n_choices_list = n_choices_list
        self.A = A
        self.B = B
        self.cat_embed_dims = get_embed_dims(n_choices_list)
        # self.ord_embed_dims = get_ord_embed_dims(n_sequences_list)
        self.ord_embed_dims = get_embed_dims(n_sequences_list)
        embed_dims_size = self.cat_embed_dims.sum() + self.ord_embed_dims.sum()
        sum_ = np.log(self.cat_embed_dims.tolist() + self.ord_embed_dims.tolist()).sum()
        if n_cont_variables:
            sum_ += np.log(n_cont_variables)
        n_discrete_variables = self.n_cat_variables + self.n_ord_variables
        self.n_layer1 = int(np.clip(int(A * (n_discrete_variables ** 0.5) * sum_ + 1), 100, 1000))
        self.n_layer2 = int(self.n_layer1 / B) + 2
        self.cat_embedding_blocks = nn.ModuleList([
            nn.Embedding(int(n_unique), int(embed_dim))
            for n_unique, embed_dim in zip(self.n_choices_list, self.cat_embed_dims)
        ])
        self.ord_embedding_blocks = nn.ModuleList([
            OrdinalEmbedding(int(n_unique), int(embed_dim))
            for n_unique, embed_dim in zip(self.n_sequences_list, self.ord_embed_dims)
        ])
        # todo: 对于每个cont变量都安排一个scaler，学习一个乘积系数
        self.cont_scaler = LearnableScaler(n_cont_variables)
        hybrid_input_size = embed_dims_size + n_cont_variables
        layer1 = self.get_block(hybrid_input_size, self.n_layer1, False, dropout1, "tanh")
        layer2 = self.get_block(self.n_layer1, self.n_layer2, False, dropout2, "leaky_relu")
        layer3 = self.get_block(self.n_layer2, self.n_class, False, dropout3, "linear")
        self.deep_net = nn.Sequential(
            layer1,
            layer2,
            layer3
        )
        self.wide_net = nn.Sequential(
            self.get_block(hybrid_input_size, self.n_layer2, False, dropout2, "leaky_relu"),
            self.get_block(self.n_layer2, self.n_class, False, dropout3, "linear"),
        )
        output_modules = []
        self.output_layer = nn.Sequential(*output_modules)
        self.initializing_modules(chain(
            self.deep_net.modules(), self.wide_net.modules(),
            self.output_layer.modules(),
        ))
        # todo: 初始化  cat_embedding_blocks

    def forward(self, X: np.ndarray):
        hybrid_embeds_list = []
        cat_embeds = []
        ord_embeds = []
        index = 0
        if self.n_cat_variables:
            X_cat = X[:, index:index + self.n_cat_variables]
            index += self.n_cat_variables
            cat_nan_mask = np.isnan(X_cat)
            X_cat = SimpleImputer(strategy='most_frequent').fit_transform(X_cat)
            mask_list = []
            for i in range(X_cat.shape[1]):
                col_vec = X_cat[:, i]
                cat_embeds.append(
                    self.cat_embedding_blocks[i](torch.from_numpy(col_vec.astype("int64")))
                )
                mask_list.append(np.tile(cat_nan_mask[:, i][:, np.newaxis], [1, self.cat_embed_dims[i]]))
            torch_cat_nan_mask = torch.from_numpy(np.hstack(mask_list))
            cat_embed = torch.cat(cat_embeds, dim=1)
            cat_embed[torch_cat_nan_mask] = 0
            hybrid_embeds_list.append(cat_embed)
        if self.n_ord_variables:
            X_ord = X[:, index:index + self.n_ord_variables]
            index += self.n_ord_variables
            ord_nan_mask = np.isnan(X_ord)
            X_ord = SimpleImputer(strategy='most_frequent').fit_transform(X_ord)
            mask_list = []
            for i in range(X_ord.shape[1]):
                col_vec = X_ord[:, i]
                ord_embeds.append(
                    self.ord_embedding_blocks[i](torch.from_numpy(col_vec.astype("int64")))
                )
                mask_list.append(np.tile(ord_nan_mask[:, i][:, np.newaxis], [1, self.ord_embed_dims[i]]))
            torch_ord_nan_mask = torch.from_numpy(np.hstack(mask_list))
            ord_embed = torch.cat(ord_embeds, dim=1)
            ord_embed[torch_ord_nan_mask] = 0
            hybrid_embeds_list.append(ord_embed)
        if self.n_cont_variables:
            X_cont = X[:, index:index + self.n_cont_variables]
            index += self.n_cont_variables
            cont_nan_mask = torch.from_numpy(np.isnan(X_cont).astype('bool'))
            X_cont = SimpleImputer().fit_transform(X_cont)
            X_cont_bn_out = self.cont_scaler(torch.from_numpy(X_cont.astype('float32')))
            X_cont_bn_out[cont_nan_mask] = 0
            hybrid_embeds_list.append(X_cont_bn_out)

        hybrid_embeds = torch.cat(hybrid_embeds_list, dim=1)
        outputs = self.deep_net(hybrid_embeds) + self.wide_net(hybrid_embeds)
        activated = self.output_layer(outputs)
        return cat_embeds, ord_embeds, activated

    def get_reg_loss(self, loss):
        for block in self.ord_embedding_blocks:
            block: OrdinalEmbedding
            loss += block.loss
        return loss


class TrainEntityEmbeddingNN(Trainer):
    def get_output(self, model, array):
        _, _, activated = model(array)
        return activated
