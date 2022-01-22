#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-12-16
# @Contact    : qichun.tang@bupt.edu.cn

import numpy as np
import pandas as pd
from pyitlib import discrete_random_variable as drv
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer


class BaseSimilarity():
    def similarity_matrix(self, X, y):
        raise NotImplementedError


class SpearmanSimilarity(BaseSimilarity):
    def __init__(self, K):
        self.K = K

    def similarity_matrix(self, X, y):
        N, M = X.shape
        n_bins = round((N ** .5) * 2)
        n_bins = min(len(set(y)), n_bins)
        bins = KBinsDiscretizer(n_bins=n_bins, strategy='quantile', encode='ordinal'). \
            fit_transform(y[:, np.newaxis]).flatten().astype('int32')
        bins_set = np.unique(bins)
        n_bins = len(bins)
        # 数据不足， 返回原来的分组
        if n_bins < self.K:
            return None
            # hierarchical_groups.append([0] * n_groups)
            # continue
        X_avg = np.zeros([n_bins, M])
        for i, bin_id in enumerate(bins_set):
            # np.count_nonzero(bins == bin_id)==0
            X_avg[i, :] = np.mean(X[bins == bin_id, :], axis=0)
        # fixme : 相关系数矩阵 不准?
        spearman_corr = np.abs(pd.DataFrame(X_avg).corr(method="spearman").values)
        # spearman_corr = np.abs(pd.DataFrame(X).corr(method="spearman").values)
        return spearman_corr
        # top_right_corr_list = []
        # for i in range(M):
        #     for j in range(i + 1, M):
        #         top_right_corr_list.append(spearman_corr[i, j])
        # print(np.mean(top_right_corr_list))
        # print(np.median(top_right_corr_list))
        # print(np.std(top_right_corr_list))


def discretize(X, y, y_n_bins=None):
    N, M = X.shape
    n_bins = round((N ** .5) * 2)
    # n_bins = round(N / 2)
    X_bins = KBinsDiscretizer(n_bins=n_bins, strategy='quantile', encode='ordinal'). \
        fit_transform(X).astype('int32')
    if y_n_bins is None:
        y_bins = None
    else:
        q = np.quantile(y, q=0.15)
        y_bins = (y < q).astype(int)
    return X_bins, y_bins


class MutualInfomationSimilarity(BaseSimilarity):
    def __init__(self):
        self.y_n_bins = None

    def similarity_matrix(self, X, y):
        N, M = X.shape
        X_bins, y_bins = discretize(X, y, self.y_n_bins)
        matrix = np.zeros([M, M])
        for i in range(M):
            for j in range(i + 1, M):
                matrix[i, j] = matrix[j, i] = mutual_info_score(X_bins[:, i], X_bins[:, j])
        return matrix


class ConditionalMutualInfomationSimilarity(BaseSimilarity):
    def __init__(self, y_n_bins=2):
        self.y_n_bins = y_n_bins

    def similarity_matrix(self, X, y):
        '''https://stackoverflow.com/questions/55402338/finding-conditional-mutual-information-from-3-discrete-variable'''
        N, M = X.shape
        X_bins, y_bins = discretize(X, y, self.y_n_bins)
        matrix = np.zeros([M, M])
        for i in range(M):
            for j in range(i + 1, M):
                # matrix[i, j] = matrix[j, i] = mutual_info_score(, X_bins[:, j])
                matrix[i, j] = matrix[j, i] = drv.information_mutual_conditional(
                    X_bins[:, i], X_bins[:, j], y_bins)
        return matrix


if __name__ == '__main__':
    np.random.seed(0)
    mis = ConditionalMutualInfomationSimilarity()
    N, M = 20, 10
    X = np.random.rand(N, M)
    y = np.random.rand(N)
    print(mis.similarity_matrix(X, y))
