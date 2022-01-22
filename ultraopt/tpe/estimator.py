#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.utils import check_random_state
from ultraopt.tpe import SpearmanSimilarity
from ultraopt.tpe import top15_gamma, SampleDisign
# from ultraopt.tpe.kernel_density import MultivariateKernelDensity
from ultraopt.tpe.kernel_density import NormalizedKernelDensity
from ultraopt.tpe.kernel_density import UnivariateCategoricalKernelDensity
from ultraopt.transform.config_transformer import ConfigTransformer
from ultraopt.utils.config_space import add_configs_origin, sample_configurations
from ultraopt.utils.hash import get_hash_of_array
from ultraopt.utils.logging_ import get_logger
from ultraopt.utils.misc import uniform_segmentation

warnings.filterwarnings('ignore')


class TreeParzenEstimator(BaseEstimator):
    def __init__(
            self,
            gamma=None, min_points_in_kde=2,
            bw_method="scott", cv_times=100, kde_sample_weight_scaler=None,
            multivariate=True,
            embed_catVar=True,
            overlap_bagging_ratio=0.5,
            limit_max_groups=True,
            max_groups=8,
            similarity=SpearmanSimilarity(K=10)
            # fill_deactivated_value=False
    ):
        self.similarity = similarity
        self.max_groups = max_groups
        self.limit_max_groups = limit_max_groups
        self.embed_catVar = embed_catVar
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
        # variable groups
        self.good_kde_groups = None
        self.bad_kde_groups = None
        self.groups = None
        self.n_groups = None
        self.hierarchical_groups_seq = []
        self.group_step = 0
        self.meta_learning = None

    def set_config_transformer(self, config_transformer: ConfigTransformer):
        self.config_transformer: ConfigTransformer = config_transformer

    def set_meta_learning(self, meta_learning: 'MetaLearning'):
        self.meta_learning = meta_learning

    def calc_groups_by_hierarchy(self, X):
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

    def estimate_group_kde(
            self, active_X=None, active_y=None, X_good=None, X_bad=None, uni_cat=0,
            bw_factor=None
    ) -> Tuple[object, object]:
        # 得到X_good, X_bad
        if active_X is not None:
            if active_X.shape[0] < 4:  # at least have 4 samples
                return None, None
            N, D = active_X.shape
            # Each KDE contains at least 2 samples
            n_good = self.gamma(N)
            if n_good < self.min_points_in_kde or \
                    N - n_good < self.min_points_in_kde:
                # Too few observation samples
                return None, None
            X_proj = active_X
            idx = np.argsort(active_y)
            X_good = X_proj[idx[:n_good]]
            X_bad = X_proj[idx[n_good:]]
        else:
            assert X_good is not None
            assert X_bad is not None
            if X_good.shape[0] < 2 or X_bad.shape[0] < 2:
                return None, None
        if uni_cat:
            klass = UnivariateCategoricalKernelDensity
        else:
            klass = NormalizedKernelDensity
        # todo: sample weight
        good_kde, bad_kde = klass(), klass()
        if hasattr(good_kde, 'bw_factor'):
            good_kde.bw_factor = bw_factor
            bad_kde.bw_factor = bw_factor
        return (
            good_kde.fit(X_good, bw_factor=bw_factor),
            bad_kde.fit(X_bad, bw_factor=bw_factor)
        )

    def variable_clustering_algorithm(self, cur_X, similarity_matrix):
        M = cur_X.shape[1]
        if M <= max(self.max_groups, 3):
            return None, None
        new_group_alloc_list = uniform_segmentation(M, self.max_groups)
        n_clusters = new_group_alloc_list.size
        if n_clusters == 1:
            return None, None

        corr_pairs = []
        for i in range(M):
            for j in range(i + 1, M):
                corr_pairs.append([similarity_matrix[i, j], i, j])
        corr_pairs = sorted(corr_pairs)[::-1]
        varId2cluster = {}
        cluster2varId = defaultdict(list)
        cluster2quota = new_group_alloc_list.copy()
        # 第一轮：建立初始cluster
        for cluster_id in range(n_clusters):
            for corr, i, j in corr_pairs:
                if i not in varId2cluster and j not in varId2cluster:
                    varId2cluster[i] = cluster_id
                    varId2cluster[j] = cluster_id
                    cluster2varId[cluster_id] += [i, j]
                    cluster2quota[cluster_id] -= 2
                    break
        # 第二轮：
        for i in range(M):
            if i in varId2cluster:
                continue
            best_cluster = None
            best_corr = -np.inf
            for cluster_id in range(n_clusters):
                if cluster2quota[cluster_id] == 0:
                    continue
                corr = np.mean(similarity_matrix[np.array(cluster2varId[cluster_id]), i])
                if corr > best_corr:
                    best_cluster = cluster_id
                    best_corr = corr
            varId2cluster[i] = best_cluster
            cluster2varId[best_cluster] += [i]
            cluster2quota[best_cluster] -= 1
        return n_clusters, varId2cluster

    def adaptive_multivariate_grouping(
            self,
            groups: np.ndarray, n_groups: int,
            X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, int, List[List]]:
        groups = groups.copy()
        hierarchical_groups = []
        cur_group = 0
        for _ in range(n_groups):
            group_mask = groups == cur_group
            cur_X = X[:, group_mask]
            inactive_mask = np.isnan(cur_X[:, 0])
            cur_X = cur_X[~inactive_mask, :]
            cur_y = y[~inactive_mask]
            M = cur_X.shape[1]
            similarity_matrix = self.similarity.similarity_matrix(cur_X, cur_y)
            if similarity_matrix is None:
                hierarchical_groups.append([0] * n_groups)
                continue
            n_clusters, varId2cluster = self.variable_clustering_algorithm(cur_X, similarity_matrix)
            # n_clusters, varId2cluster = self.cluster_algo_hierarchical_clustering(cur_X, spearman_corr)
            if n_clusters is None:
                hierarchical_groups.append([0] * n_groups)
                continue
            # 部分合并的情况
            nxt_group = cur_group + (n_clusters - 1)
            groups[groups > cur_group] += (n_clusters - 1)
            n_groups += (n_clusters - 1)
            agg_groups_ = []
            for i in range(M):
                agg_groups_.append(varId2cluster[i])
            agg_groups = np.array(agg_groups_)
            groups[group_mask] += agg_groups
            hierarchical_groups.append(agg_groups_)
            cur_group = nxt_group
        return groups, n_groups, hierarchical_groups

    def calc_variable_groups(self, X, y=None):
        if y is None:
            self.groups, self.n_groups = self.calc_groups_by_hierarchy(X)
        if self.multivariate:
            # 当时设计的时候，构造config_transformer时算了分组， 现在fit也算了分组
            self.groups, self.n_groups = self.calc_groups_by_hierarchy(X)
            if self.limit_max_groups:
                self.groups, self.n_groups, hierarchical_groups = \
                    self.adaptive_multivariate_grouping(self.groups, self.n_groups, X, y)
                self.hierarchical_groups_seq.append(hierarchical_groups)
        else:
            # 考虑Embedding encoder的影响
            n_variables = self.config_transformer.n_variables_embedded_list
            n_variables = [x for x in n_variables if x]
            groups = []
            n_groups = 0
            for v in n_variables:
                groups += [n_groups] * v
                n_groups += 1
            self.groups, self.n_groups = np.array(groups), n_groups

    def fit(self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            X_bad: Optional[np.ndarray] = None,
            bw_factor=None,
            ):
        self.n_samples = X.shape[0] + (0 if X_bad is None else X_bad.shape[0])
        if self.n_groups is None:
            self.calc_variable_groups(X, y)
        elif self.group_step % 10 == 0:
            self.calc_variable_groups(X, y)
        self.group_step += 1
        n_choices_list = self.config_transformer.n_choices_list
        self.good_kde_groups = [None] * self.n_groups
        self.bad_kde_groups = self.good_kde_groups[:]
        for group in range(self.n_groups):
            group_mask = self.groups == group
            grouped_X = X[:, group_mask]
            inactive_mask = np.isnan(grouped_X[:, 0])
            active_X = grouped_X[~inactive_mask, :]
            if y is not None:
                active_y = y[~inactive_mask]
            else:
                active_y = None
            if X_bad is not None:
                grouped_X_bad = X_bad[:, group_mask]
                inactive_mask_bad = np.isnan(grouped_X_bad[:, 0])
                active_X_bad = grouped_X_bad[~inactive_mask_bad, :]
            else:
                active_X_bad = None
            uni_cat = n_choices_list[group] if not self.embed_catVar else 0
            if X_bad is None:
                self.good_kde_groups[group], self.bad_kde_groups[group] = \
                    self.estimate_group_kde(
                        active_X=active_X, active_y=active_y,
                        uni_cat=uni_cat, bw_factor=bw_factor
                    )
            else:
                self.good_kde_groups[group], self.bad_kde_groups[group] = \
                    self.estimate_group_kde(
                        X_good=active_X, X_bad=active_X_bad, bw_factor=bw_factor
                    )
        return self

    def predict_group_multivariant(self, X: np.ndarray, return_each_varGroups=False):
        '''
        如果len(set(group)) == X.shape[1]， 分组联合概率 退化为 M个独立的概率分布
        :param X:
        :return:
        '''
        n_groups = self.n_groups
        good_log_pdf = np.zeros([X.shape[0], n_groups], dtype="float64")
        bad_log_pdf = deepcopy(good_log_pdf)
        groups = self.groups
        for group, (good_kde, bad_kde) in \
                enumerate(zip(self.good_kde_groups, self.bad_kde_groups)):
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
            X_proj = active_X
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
        if return_each_varGroups:
            return good_log_pdf - bad_log_pdf
        return good_log_pdf.sum(axis=1) - bad_log_pdf.sum(axis=1)

    def predict(self, X: np.ndarray):
        logEI = self.predict_group_multivariant(X)
        if self.meta_learning is None:
            return logEI
        return self.meta_learning.merge_predict(X, logEI, self.n_samples)

    def __sample(self, n_candidates, random_state, bandwidth_factor,
                 is_random_sample, optimize_each_varGroups):
        # todo: 用CMA-ES采样
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity
        if n_candidates <= 0: return []
        if is_random_sample:
            if optimize_each_varGroups:
                # fixme : 对inactive变量的处理
                return []
            return sample_configurations(self.config_transformer.config_space, n_candidates)
        else:
            groups = np.array(self.groups)
            rng = check_random_state(random_state)
            if self.good_kde_groups is None:
                self.logger.warning("good_kde_groups is None, random sampling.")
                return sample_configurations(self.config_transformer.config_space, n_candidates)
            sampled_matrix = np.zeros([n_candidates, len(self.groups)])
            for group, good_kde in enumerate(self.good_kde_groups):
                group_mask = groups == group
                if good_kde:
                    # KDE采样
                    bw = good_kde.bandwidth
                    prev_bw = deepcopy(bw)
                    bw *= bandwidth_factor
                    good_kde.set_params(bandwidth=bw)
                    result = good_kde.sample(n_candidates, random_state=random_state)
                    # 从采样后的空间恢复到原来的空间中
                    good_kde.set_params(bandwidth=prev_bw)
                else:
                    # 随机采样(0-1)
                    result = rng.rand(n_candidates, group_mask.sum())
                sampled_matrix[:, group_mask] = result
            candidates = []
            # todo: 在这里处理不激活变量
            candidates += self.config_transformer.inverse_transform(
                sampled_matrix, return_vector=optimize_each_varGroups)
            if not optimize_each_varGroups:
                # 避免对inactive变量的处理
                n_fails = n_candidates - len(candidates)
                add_configs_origin(candidates, "ETPE sampling")
                if n_fails:
                    random_candidates = sample_configurations(self.config_transformer.config_space, n_fails)
                    add_configs_origin(random_candidates, "Random Search")
                    candidates.extend(random_candidates)
            return candidates

    def sample(self, n_candidates=20, sort_by_EI=False,
               random_state=None, bandwidth_factor=3,
               specific_sample_design=None,
               optimize_each_varGroups=False
               ) -> List[Configuration]:
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
                candidates += self.__sample(n_samples, random_state, -1, True, optimize_each_varGroups)
            else:
                bw_factor = design.bw_factor
                assert bw_factor is not None
                candidates += self.__sample(n_samples, random_state, bw_factor, False, optimize_each_varGroups)
        if n_candidates >= 0:
            candidates += self.__sample(n_candidates, random_state, bandwidth_factor, False, optimize_each_varGroups)
        if sort_by_EI:
            # try:
            if optimize_each_varGroups:
                X = np.array(candidates)
                X_trans = self.config_transformer.transform(X)
            else:
                X = np.array([candidate.get_array() for candidate in candidates])
                X_trans = self.config_transformer.transform(X)
            if optimize_each_varGroups:  # [n_candidates, n_groups]
                EI_groups = self.predict_group_multivariant(
                    X_trans, return_each_varGroups=True)
                for group in range(self.n_groups):
                    group_mask = (self.groups == group)
                    X_trans[:, group_mask] = X_trans[:, group_mask][np.argsort(-EI_groups[:, group]), :]
                candidates = self.config_transformer.inverse_transform(X_trans)
            else:
                EI = self.predict(X_trans)
                indexes = np.argsort(-EI)
                candidates = [candidates[ix] for ix in indexes]
            # except Exception as e:
            #     self.logger.error(f"sort_by_EI failed: {e}")
        return candidates
