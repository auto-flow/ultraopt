#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_random_state
from ultraopt.tpe import top15_gamma, estimate_bw, SampleDisign
from ultraopt.tpe.kernel_density import UnivariateCategoricalKernelDensity
from ultraopt.tpe.nn_projector import NN_projector
from ultraopt.utils.config_space import add_configs_origin, sample_configurations
from ultraopt.utils.config_transformer import ConfigTransformer
from ultraopt.utils.hash import get_hash_of_array
from ultraopt.utils.logging_ import get_logger


class TreeParzenEstimator(BaseEstimator):
    def __init__(
            self,
            gamma=None, min_points_in_kde=2,
            bw_method="scott", cv_times=100, kde_sample_weight_scaler=None,
            multivariate=True,
            embed_cat_var=True,
            overlap_bagging_ratio=0.5,

            # fill_deactivated_value=False
    ):
        self.embed_cat_var = embed_cat_var
        self.multivariate = multivariate
        self.overlap_bagging_ratio = overlap_bagging_ratio
        self.min_points_in_kde = min_points_in_kde
        # self.bw_estimation = bw_estimation
        # self.min_bandwidth = min_bandwidth
        # self.bandwidth_factor = bandwidth_factor
        self.gamma = gamma or top15_gamma
        self.config_transformer: Optional[ConfigTransformer] = None
        self.logger = get_logger(self)
        self.kde_sample_weight_scaler = kde_sample_weight_scaler
        self.cv_times = cv_times
        self.bw_method = bw_method
        # self.fill_deactivated_value = fill_deactivated_value
        self.good_kde_groups = None
        self.bad_kde_groups = None

    def set_config_transformer(self, config_transformer):
        self.config_transformer = config_transformer

    def calc_groups(self, X):
        N, M = X.shape
        groups = []
        n_groups = 0
        hash2group = {}
        for i in range(M):
            col = X[:, i]
            idxs = np.arange(N)[~np.isnan(col)]
            hash = get_hash_of_array(idxs)
            if hash in hash2group:
                groups.append(hash2group[hash])
            else:
                groups.append(n_groups)
                hash2group[hash] = n_groups
                n_groups += 1
        return np.array(groups), n_groups

    def calc_overlap(self, X):
        mask = (~pd.isna(X)).astype(int)
        key2idxs = defaultdict(list)
        for i, row in enumerate(mask):
            # 用(0,1)tuple做key
            hashable_key = tuple(row)
            key2idxs[hashable_key].append(i)
        overlap_key2idxs = {k: np.array(v) for k, v in key2idxs.items()}
        return overlap_key2idxs

    def estimate_group_kde(self, active_X, active_y, nn_projector=None, uni_cat=0) -> Tuple[object, object]:
        if active_X.shape[0] < 4:  # at least have 4 samples
            return None, None
        N, M = active_X.shape
        # Each KDE contains at least 2 samples
        n_good = self.gamma(N)
        if n_good < self.min_points_in_kde or \
                N - n_good < self.min_points_in_kde:
            # Too few observation samples
            return None, None
        if nn_projector is not None and \
                active_X.shape[1] > 3 and active_X.shape[0] > nn_projector.n_bins_in_y:
            # 通过线性变换映射到一个新空间中
            X_proj = nn_projector.fit_transform(active_X, active_y)
        else:
            X_proj = active_X
        idx = np.argsort(active_y)
        X_good = X_proj[idx[:n_good]]
        X_bad = X_proj[idx[n_good:]]
        y_good = -active_y[idx[:n_good]]
        sample_weight = None
        if self.kde_sample_weight_scaler is not None and y_good.std() != 0:
            if self.kde_sample_weight_scaler == "normalize":
                scaled_y = (y_good - y_good.mean()) / y_good.std()
                scaled_y -= np.min(scaled_y)
                scaled_y /= np.max(scaled_y)
                scaled_y += 0.5
                sample_weight = scaled_y
            elif self.kde_sample_weight_scaler == "std-exp":
                scaled_y = (y_good - y_good.mean()) / y_good.std()
                sample_weight = np.exp(scaled_y)
            else:
                raise ValueError(f"Invalid kde_sample_weight_scaler '{self.kde_sample_weight_scaler}'")
        bw_good = estimate_bw(X_good, self.bw_method, self.cv_times)
        bw_bad = estimate_bw(X_bad, self.bw_method, self.cv_times)
        if uni_cat:
            klass = UnivariateCategoricalKernelDensity
        else:
            klass = KernelDensity
        return (
            klass(bandwidth=bw_good).fit(X_good, sample_weight=sample_weight),
            klass(bandwidth=bw_bad).fit(X_bad)
        )

    @property
    def can_overlap_multivariant_(self):
        return self.config_transformer.multivariate and self.n_groups != 1 and self.overlap_bagging_ratio > 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.multivariate:
            # 当时设计的时候，构造config_transformer时算了分组， 现在fit也算了分组
            self.groups, self.n_groups = self.calc_groups(X)
        else:
            self.groups, self.n_groups = self.config_transformer.groups, self.config_transformer.n_groups
        n_choices_list = self.config_transformer.n_choices_list
        # =============================================
        # =              group kde                    =
        # =============================================
        self.good_kde_groups = [None] * self.n_groups
        self.bad_kde_groups = self.good_kde_groups[:]
        # todo: 不每次都拟合一个NN
        self.nn_projectors = [NN_projector() for _ in range(self.n_groups)]
        for group in range(self.n_groups):
            group_mask = self.groups == group
            grouped_X = X[:, group_mask]
            inactive_mask = np.isnan(grouped_X[:, 0])
            active_X = grouped_X[~inactive_mask, :]
            active_y = y[~inactive_mask]
            self.good_kde_groups[group], self.bad_kde_groups[group] = \
                self.estimate_group_kde(
                    active_X, active_y,
                    uni_cat=n_choices_list[group] if not self.embed_cat_var else 0
                    # self.nn_projectors[group]
                )
        # =============================================
        # =              overlap kde                  =
        # =============================================
        if not self.can_overlap_multivariant_:
            return self
        overlap_key2idxs = self.calc_overlap(X)
        self.maskTuple_to_overlapKdePairs = {}
        for mask_tuple, idx in overlap_key2idxs.items():
            active_X = X[idx, :][:, np.array(mask_tuple).astype(bool)]
            active_y = y[idx]
            kde_pairs = self.estimate_group_kde(
                active_X, active_y,
            )
            self.maskTuple_to_overlapKdePairs[mask_tuple] = kde_pairs
        return self

    def predict_group_multivariant(self, X: np.ndarray):
        '''
        如果len(set(group)) == X.shape[1]， 分组联合概率 退化为 M个独立的概率分布
        :param X:
        :return:
        '''
        n_groups = self.n_groups
        good_log_pdf = np.zeros([X.shape[0], n_groups], dtype="float64")
        bad_log_pdf = deepcopy(good_log_pdf)
        groups = self.groups
        for group, (good_kde, bad_kde, nn_projector) in \
                enumerate(zip(self.good_kde_groups, self.bad_kde_groups, self.nn_projectors)):
            if (good_kde, bad_kde) == (None, None):
                continue
            group_mask = groups == group
            grouped_X = X[:, group_mask]
            inactive_mask = np.isnan(grouped_X[:, 0])
            active_X = grouped_X[~inactive_mask, :]
            N, M = active_X.shape
            if N == 0:
                continue
            if np.any(pd.isna(active_X)):
                self.logger.warning("ETPE contains nan, mean impute.")
                active_X = SimpleImputer(strategy="mean").fit_transform(active_X)
            X_proj = nn_projector.transform(active_X)
            good_log_pdf[~inactive_mask, group] = self.good_kde_groups[group].score_samples(X_proj)
            bad_log_pdf[~inactive_mask, group] = self.bad_kde_groups[group].score_samples(X_proj)
            # if N_deactivated > 0 and self.fill_deactivated_value:
            #     good_log_pdf[~mask, i] = np.random.choice(good_pdf_activated)
            #     bad_log_pdf[~mask, i] = np.random.choice(bad_pdf_activated)
        if not np.all(np.isfinite(good_log_pdf)):
            self.logger.warning("good_log_pdf contains NaN or inf")
        if not np.all(np.isfinite(bad_log_pdf)):
            self.logger.warning("bad_log_pdf contains NaN or inf")
        good_log_pdf[~np.isfinite(good_log_pdf)] = -10
        bad_log_pdf[bad_log_pdf == -np.inf] = -10
        bad_log_pdf[~np.isfinite(bad_log_pdf)] = 10
        result = good_log_pdf.sum(axis=1) - bad_log_pdf.sum(axis=1)
        return result

    def predict_overlap_multivariant(self, X: np.ndarray):
        overlap_key2idxs = self.calc_overlap(X)
        N = X.shape[0]
        prediction = np.array([np.nan] * N)
        for mask_tuple, idx in overlap_key2idxs.items():
            if self.maskTuple_to_overlapKdePairs.get(mask_tuple, (None, None)) == (None, None):
                continue
            good_kde, bad_kde = self.maskTuple_to_overlapKdePairs[mask_tuple]
            active_X = X[idx, :][:, np.array(mask_tuple).astype(bool)]
            prediction[idx] = good_kde.score_samples(active_X) - bad_kde.score_samples(active_X)
        return prediction

    def predict(self, X: np.ndarray):
        # 不能重叠多变量估计，直接返回 分组联合概率估计
        if not self.can_overlap_multivariant_:
            return self.predict_group_multivariant(X)
        group_multivariant_EI = self.predict_group_multivariant(X)
        # print(group_multivariant_EI)
        overlap_multivariant_EI = self.predict_overlap_multivariant(X)
        overlap_variant_occur = (np.count_nonzero(pd.isna(overlap_multivariant_EI)) < X.shape[0])
        occur_mask = (~pd.isna(overlap_multivariant_EI))
        # if overlap_variant_occur:
        # group_occur_EI = group_multivariant_EI[occur_mask]
        # overlap_occur_EI = overlap_multivariant_EI[occur_mask]
        # group_mean, group_std = np.mean(group_occur_EI), np.std(group_occur_EI)
        # group_min, group_max = np.min(group_occur_EI), np.max(group_occur_EI)
        # overlap_mean, overlap_std = np.mean(overlap_occur_EI), np.std(overlap_occur_EI)
        # overlap_min, overlap_max = np.min(overlap_occur_EI), np.max(overlap_occur_EI)
        # # overlap_multivariant_EI[occur_mask] -= overlap_mean
        # # overlap_multivariant_EI[occur_mask] /= overlap_std
        # # overlap_multivariant_EI[occur_mask] *= group_std
        # # overlap_multivariant_EI[occur_mask] += group_mean
        # overlap_multivariant_EI[occur_mask] -= overlap_min
        # overlap_multivariant_EI[occur_mask] /= (overlap_max - overlap_min)
        # overlap_multivariant_EI[occur_mask] *= (group_max - group_min)
        # overlap_multivariant_EI[occur_mask] += group_min
        overlap_multivariant_EI[~occur_mask] = group_multivariant_EI[~occur_mask]
        p = self.overlap_bagging_ratio
        assert 0 <= p <= 1
        return np.log(np.exp(overlap_multivariant_EI) * p + np.exp(group_multivariant_EI) * (1 - p))

    def __sample(self, n_candidates, random_state=None, bandwidth_factor=3, is_random_sample=False):
        if n_candidates <= 0: return []
        if is_random_sample:
            return sample_configurations(self.config_transformer.config_space, n_candidates)
        else:
            groups = np.array(self.groups)
            rng = check_random_state(random_state)
            if self.good_kde_groups is None:
                self.logger.warning("good_kde_groups is None, random sampling.")
                return sample_configurations(self.config_transformer.config_space, n_candidates)
            sampled_matrix = np.zeros([n_candidates, len(self.groups)])
            for group, good_kde in enumerate(self.good_kde_groups):
                nn_projector = self.nn_projectors[group]
                group_mask = groups == group
                if good_kde:
                    # KDE采样
                    bw = good_kde.bandwidth
                    prev_bw = bw
                    bw *= bandwidth_factor
                    good_kde.set_params(bandwidth=bw)
                    result = good_kde.sample(n_candidates, random_state=random_state)
                    # 从采样后的空间恢复到原来的空间中
                    result = nn_projector.inverse_transform(result)
                    good_kde.set_params(bandwidth=prev_bw)
                else:
                    # 随机采样(0-1)
                    result = rng.rand(n_candidates, group_mask.sum())
                sampled_matrix[:, group_mask] = result
            candidates = []
            candidates += self.config_transformer.inverse_transform(sampled_matrix)
            n_fails = n_candidates - len(candidates)
            add_configs_origin(candidates, "ETPE sampling")
            if n_fails:
                random_candidates = sample_configurations(self.config_transformer.config_space, n_fails)
                add_configs_origin(random_candidates, "Random Search")
                candidates.extend(random_candidates)
            return candidates

    def sample(self, n_candidates=20, sort_by_EI=False,
               random_state=None, bandwidth_factor=3,
               specific_sample_design=None) -> List[Configuration]:
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity
        if specific_sample_design is None:
            specific_sample_design = [
                SampleDisign(ratio=1, n_samples=0, is_random=False, bw_factor=bandwidth_factor)]
        candidates = []
        for design in specific_sample_design:
            if n_candidates <= 0:
                break
            if design.n_samples != 0:
                n_samples = design.n_samples
            else:
                n_samples = round(design.ratio * n_candidates)
                n_candidates -= n_samples
            if design.is_random:
                candidates += self.__sample(n_samples, random_state, is_random_sample=True)
            else:
                bw_factor = design.bw_factor
                assert bw_factor is not None
                candidates += self.__sample(n_samples, random_state, bandwidth_factor=bw_factor)
        if n_candidates >= 0:
            candidates += self.__sample(n_candidates, random_state, bandwidth_factor=bandwidth_factor)
        if sort_by_EI:
            # try:
            X = [candidate.get_array() for candidate in candidates]
            X_trans = self.config_transformer.transform(X)

            EI = self.predict(X_trans)
            indexes = np.argsort(-EI)
            candidates = [candidates[ix] for ix in indexes]
            # except Exception as e:
            #     self.logger.error(f"sort_by_EI failed: {e}")
        return candidates
