#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
import category_encoders.utils as util
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from tabular_nn.utils.data import pairwise_distance


def get_equidistance_matrix(N):
    '''

    Parameters
    ----------
    N

    Returns
    -------
    matrix, N rows, N - 1 columns

    '''
    return np.vstack([np.eye(N - 1), np.ones([1, N - 1]) * ((1 - np.sqrt(N)) / (N - 1))])


class EquidistanceEncoder(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            cols=None,
            return_df=True,
    ):
        self.return_df = return_df
        self.cols = cols
        self.fitted = False

    def fit(self, X, y=None, **kwargs):
        X = util.convert_input(X)
        self.n_columns = X.shape[1]
        if self.cols is None:
            self.cols = util.get_obj_cols(X)
        else:
            self.cols = util.convert_cols_to_list(self.cols)
        self.col_idxs = np.arange(self.n_columns)[X.columns.isin(self.cols)]
        self.n_choices_list = []
        self.le_list = []
        self.equidistance_matrix_list = []
        for col in self.cols:
            n_choices = X[col].nunique()
            self.n_choices_list.append(n_choices)
            self.le_list.append(LabelEncoder().fit(X[col][~np.isnan(X[col])]))
            self.equidistance_matrix_list.append(get_equidistance_matrix(n_choices))
        self.fitted = True
        return self

    def transform(self, X):
        X = util.convert_input(X)
        result_df_list = []
        cur_columns = []
        col2idx = dict(zip(self.cols, range(len(self.cols))))
        N = X.shape[0]
        index = X.index
        X.index = range(X.shape[0])
        for column in X.columns:
            if column in self.cols:
                if len(cur_columns) > 0:
                    result_df_list.append(X[cur_columns])
                    cur_columns = []
                idx = col2idx[column]
                n_choices = self.n_choices_list[idx]
                col_vector = np.zeros(shape=(N, n_choices - 1), dtype="float32")
                mask = (~np.isnan(X[column]))
                if np.any(mask):
                    mask_vector = self.le_list[idx].transform(X[column][mask])
                    col_vector[mask] = self.equidistance_matrix_list[idx][mask_vector]
                # idx = col2idx[column]
                # embed = X_embeds[idx]
                new_columns = [f"{column}_{i}" for i in range(col_vector.shape[1])]
                embed = pd.DataFrame(col_vector, columns=new_columns)
                result_df_list.append(embed)
            else:
                cur_columns.append(column)
        if len(cur_columns) > 0:
            result_df_list.append(X[cur_columns])
            cur_columns = []
        X = pd.concat(result_df_list, axis=1)
        X.index = index
        if self.return_df:
            return X
        else:
            return X.values

    def inverse_transform(self, X):
        # X: np.ndarray = check_array(X, force_all_finite=True)
        X = np.array(X)
        assert self.n_columns == X.shape[1] - (np.sum(self.n_choices_list) - 2 * len(self.n_choices_list))
        results = np.zeros([X.shape[0], 0], dtype="float")
        cur_cnt = 0
        col_idx2idx = dict(zip(self.col_idxs, range(len(self.cols))))
        for origin_col_idx in range(self.n_columns):
            if origin_col_idx in self.col_idxs:
                idx = col_idx2idx[origin_col_idx]
                next_cnt = cur_cnt + self.n_choices_list[idx] - 1
                embed = X[:, cur_cnt:next_cnt]
                distance = pairwise_distance(embed, self.equidistance_matrix_list[idx])
                le_output = distance.argmin(axis=1)
                le_input = self.le_list[idx].inverse_transform(le_output)
                result = le_input
                cur_cnt = next_cnt
            else:
                result = X[:, cur_cnt]
                cur_cnt += 1
            results = np.hstack([results, result[:, None]])
        return results


if __name__ == '__main__':
    # test 1
    rng = np.random.RandomState(42)
    X = rng.rand(10, 5)
    X[:, 0] = rng.randint(0, 5, [10])
    X[:, 1] = rng.randint(0, 5, [10])
    X[:, 4] = rng.randint(0, 8, [10])
    X[:, 3] = rng.randint(0, 8, [10])
    encoder = EquidistanceEncoder(cols=[0, 1, 3, 4], return_df=True).fit(X)
    X_trans = encoder.transform(X)
    X_inv = encoder.inverse_transform(X_trans)
    assert np.all(X_inv == X)

    # test 2
    rng = np.random.RandomState(42)
    X = rng.rand(10, 5)
    X[:, 0] = rng.randint(0, 5, [10])
    X[:, 1] = rng.randint(0, 5, [10])
    X[:, 2] = rng.randint(0, 5, [10])
    X[:, 4] = rng.randint(0, 8, [10])
    X[:, 3] = rng.randint(0, 8, [10])
    encoder = EquidistanceEncoder(cols=[0, 1, 2, 3, 4], return_df=True).fit(X)
    X_trans = encoder.transform(X)
    X_inv = encoder.inverse_transform(X_trans)
    assert np.all(X_inv == X)

    # test 3
    rng = np.random.RandomState(42)
    X = rng.rand(10, 5)
    encoder = EquidistanceEncoder(cols=[], return_df=True).fit(X)
    X_trans = encoder.transform(X)
    X_inv = encoder.inverse_transform(X_trans)
    assert np.all(X_inv == X)

    # test 4
    rng = np.random.RandomState(42)
    X = rng.rand(10, 5)
    X[:, 0] = rng.randint(0, 5, [10])
    X[:, 2] = rng.randint(0, 5, [10])
    X[:, 4] = rng.randint(0, 8, [10])
    encoder = EquidistanceEncoder(cols=[0, 2, 4], return_df=True).fit(X)
    X_trans = encoder.transform(X)
    X_inv = encoder.inverse_transform(X_trans)
    assert np.all(X_inv == X)

    print(X_inv)

    #
