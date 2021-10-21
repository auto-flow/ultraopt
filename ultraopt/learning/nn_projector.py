#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-05
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from torch import nn


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MyNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=5, output_dim=6):
        super(MyNN, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.encoder = nn.Linear(self.input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.input_dim)
        self.fc_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.active_func = nn.Tanh()

    def forward(self, X):
        return self.fc_layer(self.active_func(self.encoder(X)))

    def encoder_forward(self, X):
        return self.encoder(X)

    def decoder_forward(self, X):
        return self.decoder(X)


class NN_projector(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins_in_y=6):
        self.n_bins_in_y = n_bins_in_y
        self.fitted = False

    def fit(self, X, y):

        self.scaler = StandardScaler().fit(X)
        bins = pd.qcut(y, self.n_bins_in_y, labels=list(range(self.n_bins_in_y))).astype(int)
        # 构造出n_bins_in_y - 1个二分类任务
        labels = np.hstack([(bins > i).astype(int)[:, None] for i in range(self.n_bins_in_y - 1)])
        X_scaled = self.scaler.transform(X)
        from sklearn.neural_network import MLPClassifier
        from time import time
        n_feats = X.shape[1]
        mlp = MLPClassifier(hidden_layer_sizes=[n_feats], activation='tanh', random_state=0, max_iter=1000,
                            solver="adam", alpha=1e-4)
        start_time = time()
        mlp.fit(X_scaled, labels)
        cost_time = time() - start_time
        # print(np.count_nonzero(mlp.predict(X_scaled)!=labels))
        # print(cost_time)

        w1, w2 = mlp.coefs_
        self.w = w1
        self.w_inv = np.matrix(w1).I
        self.fitted = True
        # b1, b2 = mlp.intercepts_
        # y_pred_logits = tanh(X_scaled @ w1 + b1) @ w2 + b2
        # y_prob = sigmoid(y_pred_logits)
        # y_pred = (y_prob > 0.5).astype(int)
        return self

    def transform(self, X):
        if not self.fitted:
            return X
        X_scaled = self.scaler.transform(X)
        return X_scaled @ self.w

    def inverse_transform(self, X):
        if not self.fitted:
            return X
        X_befor_scale = X @ self.w_inv
        return self.scaler.inverse_transform(X_befor_scale)
