#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
from copy import deepcopy
from typing import List, Optional

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_random_state

from ultraopt.utils.config_space import add_configs_origin, sample_configurations
from ultraopt.utils.config_transformer import ConfigTransformer
from ultraopt.utils.hash import get_hash_of_array
from ultraopt.utils.logging_ import get_logger


def estimate_bw(data, bw_method="scott", cv_times=100):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.estimate_bandwidth.html
    ndata = data.shape[0]
    if bw_method == 'scott':
        bandwidth = ndata ** (-1 / 5) * np.std(data, ddof=1)
        bandwidth = np.clip(bandwidth, 0.01, None)
    elif bw_method == 'silverman':
        bandwidth = (ndata * 3 / 4) ** (-1 / 5) * np.std(data, ddof=1)
        bandwidth = np.clip(bandwidth, 0.01, None)
    elif bw_method == 'cv':
        if ndata <= 3:
            return estimate_bw(data)
        bandwidths = np.std(data, ddof=1) ** np.linspace(-1, 1, cv_times)
        bandwidths = np.clip(bandwidths, 0.01, None)
        grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths},
                            cv=KFold(n_splits=3, shuffle=True, random_state=0))
        grid.fit(data)
        bandwidth = grid.best_params_['bandwidth']
    elif np.isscalar(bw_method):
        bandwidth = bw_method
    else:
        raise ValueError("Unrecognized input for bw_method.")
    return bandwidth


class TreeParzenEstimator(BaseEstimator):
    def __init__(
            self,
            top_n_percent=15, min_points_in_kde=2,
            bw_method="scott", cv_times=100, kde_sample_weight_scaler=None,
            # fill_deactivated_value=False
    ):
        self.min_points_in_kde = min_points_in_kde
        # self.bw_estimation = bw_estimation
        # self.min_bandwidth = min_bandwidth
        # self.bandwidth_factor = bandwidth_factor
        self.top_n_percent = top_n_percent
        self.config_transformer: Optional[ConfigTransformer] = None
        self.logger = get_logger(self)
        self.kde_sample_weight_scaler = kde_sample_weight_scaler
        self.cv_times = cv_times
        self.bw_method = bw_method
        # self.fill_deactivated_value = fill_deactivated_value
        self.good_kdes = None
        self.bad_kdes = None

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

    def fit(self, X: np.ndarray, y: np.ndarray):
        groups, n_groups = self.calc_groups(X)
        self.groups = groups
        self.n_groups = n_groups
        good_kdes = np.zeros([n_groups], dtype=object)
        bad_kdes = deepcopy(good_kdes)
        for group in range(n_groups):
            group_mask = groups == group
            grouped_X = X[:, group_mask]
            inactive_mask = np.isnan(grouped_X[:, 0])
            active_X = grouped_X[~inactive_mask, :]
            active_y = y[~inactive_mask]
            if active_X.shape[0] < 4:  # at least have 4 samples
                continue
            N, M = active_X.shape
            # Each KDE contains at least 2 samples
            n_good = max(2, (self.top_n_percent * N) // 100)
            if n_good < self.min_points_in_kde or \
                    N - n_good < self.min_points_in_kde:
                # Too few observation samples
                continue
            idx = np.argsort(active_y)
            X_good = active_X[idx[:n_good]]
            X_bad = active_X[idx[n_good:]]
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
            good_kdes[group] = KernelDensity(bandwidth=bw_good).fit(X_good, sample_weight=sample_weight)
            bad_kdes[group] = KernelDensity(bandwidth=bw_bad).fit(X_bad)
        self.good_kdes = good_kdes
        self.bad_kdes = bad_kdes
        return self

    def predict(self, X: np.ndarray):
        n_groups = self.n_groups
        good_log_pdf = np.zeros([X.shape[0], n_groups], dtype="float64")
        bad_log_pdf = deepcopy(good_log_pdf)
        groups = self.groups
        for group, (good_kde, bad_kde) in enumerate(zip(self.good_kdes, self.bad_kdes)):
            if (not good_kde) or (not bad_kde):
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
            good_log_pdf[~inactive_mask, group] = self.good_kdes[group].score_samples(active_X)
            bad_log_pdf[~inactive_mask, group] = self.bad_kdes[group].score_samples(active_X)
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

    def sample(self, n_candidates=20, sort_by_EI=False, random_state=None, bandwidth_factor=3) -> List[Configuration]:
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity
        groups = np.array(self.groups)
        rng = check_random_state(random_state)
        if self.good_kdes is None:
            self.logger.warning("good_kdes is None, random sampling.")
            return sample_configurations(self.config_transformer.config_space, n_candidates)
        sampled_matrix = np.zeros([n_candidates, len(self.groups)])
        for group, good_kde in enumerate(self.good_kdes):
            group_mask = groups == group
            if good_kde:
                # KDE采样
                bw = good_kde.bandwidth
                prev_bw = bw
                bw *= bandwidth_factor
                good_kde.set_params(bandwidth=bw)
                result = good_kde.sample(n_candidates, random_state=random_state)
                good_kde.set_params(bandwidth=prev_bw)
            else:
                # 随机采样(0-1)
                result = rng.rand(n_candidates, group_mask.sum())
            sampled_matrix[:, group_mask] = result
        candidates = self.config_transformer.inverse_transform(sampled_matrix)
        n_fails = n_candidates - len(candidates)
        add_configs_origin(candidates, "ETPE sampling")
        if n_fails:
            random_candidates = sample_configurations(self.config_transformer.config_space, n_fails)
            add_configs_origin(random_candidates, "Random Search")
            candidates.extend(random_candidates)
        if sort_by_EI:
            try:
                X = [candidate.get_array() for candidate in candidates]
                X_trans = self.config_transformer.transform(X)

                EI = self.predict(X_trans)
                indexes = np.argsort(-EI)
                candidates = [candidates[ix] for ix in indexes]
            except Exception as e:
                self.logger.error(f"sort_by_EI failed: {e}")
        return candidates
