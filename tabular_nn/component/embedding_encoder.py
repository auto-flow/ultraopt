#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
import warnings
from time import time
from typing import List, Optional, Tuple

import category_encoders.utils as util
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils._random import check_random_state
from sklearn.utils.multiclass import type_of_target
from tabular_nn.entity_embedding_nn import TrainEntityEmbeddingNN, EntityEmbeddingNN
from tabular_nn.utils.data import pairwise_distance
from tabular_nn.utils.logging_ import get_logger

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_array(obj):
    if isinstance(obj, np.ndarray):
        return obj
    return obj.detach().numpy()


class EmbeddingEncoder(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            cat_cols=[],
            cont_cols=[],
            ord_cols=[],
            n_choices_list=[],
            n_sequences_list=[],
            lr=1e-2,
            max_epoch=25,
            A=10,
            B=5,
            dropout1=0.1,
            dropout2=0.1,
            weight_decay=5e-4,
            random_state=1000,
            verbose=1,
            n_jobs=-1,
            class_weight=None,
            batch_size=1024,
            optimizer="adam",
            # normalize=True,
            copy=True,
            early_stopping_rounds=10,
            update_epoch=10,
            update_accepted_samples=10,
            update_used_samples=100,
            category_encoder='embedding'
    ):
        assert category_encoder in {'embedding', 'one-hot'}
        self.category_encoder = category_encoder
        self.n_sequences_list = n_sequences_list
        self.n_choices_list = n_choices_list
        self.ord_cols = ord_cols
        self.weight_decay = weight_decay
        self.cont_cols = cont_cols
        self.update_used_samples = update_used_samples
        self.update_epoch = update_epoch
        self.update_accepted_samples = update_accepted_samples
        # self.accepted_samples = accepted_samples
        # self.warm_start = warm_start
        self.early_stopping_rounds = early_stopping_rounds
        self.copy = copy
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.random_state = random_state
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.lr = lr
        self.dropout2 = dropout2
        self.dropout1 = dropout1
        self.B = B
        self.A = A
        self.max_epoch = max_epoch
        # self.return_df = return_df
        self.drop_cols = []
        self.cat_cols = cat_cols
        self._dim = None
        self.feature_names = None
        self.model: Optional[EntityEmbeddingNN] = None
        self.logger = get_logger(self)
        self.nn_params = {
            "A": self.A,
            "B": self.B,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2,
        }
        self.rng: np.random.RandomState = check_random_state(self.random_state)
        self.trainer = TrainEntityEmbeddingNN(
            lr=self.lr,
            max_epoch=self.max_epoch,
            nn_params=self.nn_params,
            random_state=self.rng,
            batch_size=batch_size,
            optimizer=optimizer,
            n_jobs=self.n_jobs,
            class_weight=class_weight,
            verbose=self.verbose,
            weight_decay=self.weight_decay
        )
        self.label_scaler = StandardScaler(copy=True)
        self.keep_going = False
        self.iter = 0
        self.is_classification = None
        self.samples_db = [pd.DataFrame(), np.array([])]
        self.transform_matrix = None
        self.stage = ""
        self.refit_times = 0
        self.transform_matrix_status = "No Updated"
        self.is_initial_fit = True
        self.fitted = False
        self.pretrained_emb = {}

    @property
    def continuous_variables_weight(self):
        if 'continuous_scaler' in self.pretrained_emb:
            cont_table = self.pretrained_emb['continuous_scaler']
            return np.array(cont_table.loc[self.cont_cols, 'weight'])
        if self.n_cont_variables == 0:
            return []
        return self.model.cont_scaler.weight.detach().numpy()

    @property
    def continuous_variables_mean(self):
        if 'continuous_scaler' in self.pretrained_emb:
            cont_table = self.pretrained_emb['continuous_scaler']
            return np.array(cont_table.loc[self.cont_cols, 'mean'])
        if self.n_cont_variables == 0:
            return []
        return self.model.cont_scaler.running_mean.detach().numpy()

    def get_initial_final_observations(self):
        return [pd.DataFrame(), np.zeros([0, self.n_labels])]

    def initial_fit(self, X: pd.DataFrame, y: np.ndarray):
        self.n_columns = X.shape[1]
        self.original_columns = X.columns
        self.cat_col_idxs = np.arange(self.n_columns)[X.columns.isin(self.cat_cols)]
        self.n_labels = y.shape[1]
        self.label_reg_mask = []
        self.clf_n_classes = []
        for i in range(self.n_labels):
            if (type_of_target(y[:, i]) == "continuous"):
                self.label_reg_mask.append(True)
            else:
                self.clf_n_classes.append(int(y[:, i].max() + 1))
                self.label_reg_mask.append(False)
        self.label_reg_mask = np.array(self.label_reg_mask)
        # if self.cat_cols:
        self.label_scaler.fit(y[:, self.label_reg_mask])
        self.fitted = True
        self.model = None
        self.cat_ohes = [
            OneHotEncoder(categories=[list(range(n_choices))], sparse=False).fit(np.arange(n_choices)[:, None])
            for n_choices in self.n_choices_list]
        self.ord_ohes = [
            OneHotEncoder(categories=[list(range(n_sequences))], sparse=False).fit(np.arange(n_sequences)[:, None])
            for n_sequences in self.n_sequences_list]

    @property
    def n_cont_variables(self):
        return len(self.cont_cols)

    def _fit(self, X: pd.DataFrame, y: np.ndarray):
        # fixme: 借助其他变量，不仅仅是cat
        X_cat = X[self.cat_cols]
        X_cont = X[self.cont_cols]
        X_ord = X[self.ord_cols]
        y[:, self.label_reg_mask] = self.label_scaler.transform(y[:, self.label_reg_mask])
        # 2. train_entity_embedding_nn
        if self.model is None:
            self.model = EntityEmbeddingNN(
                np.array(self.n_choices_list),
                np.array(self.n_sequences_list),
                self.n_cont_variables,
                n_class=int(np.sum(self.label_reg_mask) + np.sum(self.clf_n_classes))
            )
        if self.category_encoder != 'embedding':
            return
        self.model = self.trainer.train(
            self.model, np.hstack([X_cat, X_ord, X_cont]), y, None, None,
            label_reg_mask=self.label_reg_mask,
            clf_n_classes=self.clf_n_classes,
        )

    def fit(self, X, y: np.ndarray = None, **kwargs):
        # first check the type
        X = util.convert_input(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if self.is_initial_fit:
            self.logger.debug('Initial fitting')
            self.initial_fit(X, y)
        self.start_time = time()
        if self.is_initial_fit:
            self.trainer.max_epoch = self.max_epoch
            self._fit(X, y)
            self.samples_db[0] = X
            self.samples_db[1] = y
            self.cat_matrix_list, self.ord_matrix_list = self.get_transform_matrix()
            self.transform_matrix_status = "Updated"
            self.stage = "Initial fitting"
            self.is_initial_fit = False
            self.final_observations = self.get_initial_final_observations()
            # todo : early_stopping choose best model
        else:
            self.model.max_epoch = 0
            self.trainer.max_epoch = self.update_epoch
            # update final_observations
            self.final_observations[0] = pd.concat([self.final_observations[0], X], axis=0).reset_index(drop=True)
            self.final_observations[1] = np.vstack([self.final_observations[1], y])
            observations = self.final_observations[0].shape[0]
            if observations < self.update_accepted_samples:
                self.logger.debug(f"only have {observations} observations, didnt training model.")
                self.transform_matrix_status = "No Updated"
                # stage and transform_matrix dont update
            else:
                n_used_samples = min(self.update_used_samples - observations, self.samples_db[0].shape[0])
                indexes = self.rng.choice(np.arange(self.samples_db[0].shape[0]), n_used_samples, False)
                # origin samples_db + final_observations -> X, y
                X_ = pd.concat([self.samples_db[0].loc[indexes, :], self.final_observations[0]]).reset_index(drop=True)
                y_ = np.vstack([self.samples_db[1][indexes], self.final_observations[1]])
                # fitting (using previous model)
                self._fit(X_, y_)
                # update samples_db by final_observations
                self.samples_db[0] = pd.concat([self.samples_db[0], self.final_observations[0]], axis=0). \
                    reset_index(drop=True)
                self.samples_db[1] = np.vstack([self.samples_db[1], self.final_observations[1]])
                # clear final_observations
                self.final_observations = self.get_initial_final_observations()
                self.refit_times += 1
                self.cat_matrix_list, self.ord_matrix_list = self.get_transform_matrix()
                self.transform_matrix_status = "Updated"
                self.stage = f"refit-{self.refit_times}-times"
        return self

    def fake_input(self, X_cat, X_ord):
        N = X_cat.shape[0]
        return np.hstack([X_cat, X_ord, np.zeros([N, self.n_cont_variables])])

    def output_embs(self, X_cat, X_ord):
        X_cat = X_cat.values if isinstance(X_cat, pd.DataFrame) else X_cat
        X_ord = X_ord.values if isinstance(X_ord, pd.DataFrame) else X_ord
        if self.category_encoder == 'embedding':
            cat_embeds, ord_embeds, _ = self.model(self.fake_input(X_cat, X_ord))
        else:
            cat_embeds = [cat_ohe.transform(X_cat[:, [i]]) for i, cat_ohe in enumerate(self.cat_ohes)]
            ord_embeds = [ord_ohe.transform(X_ord[:, [i]]) for i, ord_ohe in enumerate(self.ord_ohes)]
        out_cat_embeds, out_ord_embeds = [], []
        for i, cat_embed in enumerate(cat_embeds):
            col = self.cat_cols[i]
            if col in self.pretrained_emb:
                out_cat_embeds.append(self.pretrained_emb[col][X_cat[:, i].astype('int')])
            else:
                out_cat_embeds.append(get_array(cat_embed))
        for i, ord_embed in enumerate(ord_embeds):
            col = self.ord_cols[i]
            if col in self.pretrained_emb:
                out_ord_embeds.append(self.pretrained_emb[col][X_ord[:, i].astype('int')])
            else:
                out_ord_embeds.append(get_array(ord_embed))
        return out_cat_embeds, out_ord_embeds

    def get_transform_matrix(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        all_list = self.n_sequences_list + self.n_choices_list
        N = max(all_list) if all_list else 0
        # cat
        M = len(self.n_choices_list)
        X_cat = np.zeros([N, M])
        for i, n_choices in enumerate(self.n_choices_list):
            X_cat[:, i][:n_choices] = np.arange(n_choices)
        # ord
        M = len(self.n_sequences_list)
        X_ord = np.zeros([N, M])
        for i, n_sequences in enumerate(self.n_sequences_list):
            X_ord[:, i][:n_sequences] = np.arange(n_sequences)
        # forward
        cat_embeds, ord_embeds = self.output_embs(X_cat, X_ord)
        cat_embeds = [cat_embed[:self.n_choices_list[i], :]
                      for i, cat_embed in enumerate(cat_embeds)]
        ord_embeds = [ord_embed[:self.n_sequences_list[i], :]
                      for i, ord_embed in enumerate(ord_embeds)]
        return cat_embeds, ord_embeds

    def transform(self, X, return_df=True):
        # first check the type
        X = util.convert_input(X)
        index = X.index
        X.index = range(X.shape[0])
        X_cat = X[self.cat_cols]
        X_ord = X[self.ord_cols]
        cat_embeds, ord_embeds = self.output_embs(X_cat, X_ord)
        values = X.values
        data = []
        cat_ix = 0
        ord_ix = 0
        columns = []
        self.type_seq = []
        idx = 0
        for i, col in enumerate(X.columns):
            if col in self.cat_cols:
                emb = cat_embeds[cat_ix]
                K = emb.shape[1]
                cat_ix += 1
                columns += [f"{col}_{i}" for i in range(K)]
                data.append(emb)
                self.type_seq.append(['cat', (idx, idx + K)])
                idx += K
            elif col in self.ord_cols:
                emb = ord_embeds[ord_ix]
                K = emb.shape[1]
                ord_ix += 1
                columns += [f"{col}_{i}" for i in range(K)]
                data.append(emb)
                self.type_seq.append(['ord', (idx, idx + K)])
                idx += K
            else:
                columns.append(col)
                data.append(values[:, i][:, np.newaxis])
                self.type_seq.append(['cont', (idx)])
                idx += 1
        X = pd.DataFrame(np.hstack(data), columns=columns, index=index)
        if return_df:
            return X
        else:
            return X.values

    def inverse_transform(self, X: np.ndarray):
        cat_ix = 0
        ord_ix = 0
        results = []
        for i, (type_, info) in enumerate(self.type_seq):
            if type_ == 'cat':
                embed = X[:, info[0]:info[1]]
                distance = pairwise_distance(embed, self.cat_matrix_list[cat_ix])
                cat_ix += 1
                result = distance.argmin(axis=1)
            elif type_ == 'ord':
                embed = X[:, info[0]:info[1]]
                distance = pairwise_distance(embed, self.ord_matrix_list[ord_ix])
                ord_ix += 1
                result = distance.argmin(axis=1)
            else:
                result = X[:, info]
            results.append(result[:, np.newaxis])
        return np.hstack(results)
