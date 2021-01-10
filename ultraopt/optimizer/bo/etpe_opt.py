#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-15
# @Contact    : qichun.tang@bupt.edu.cn
from copy import deepcopy

import numpy as np
from tabular_nn import EmbeddingEncoder
from tabular_nn import EquidistanceEncoder

from ultraopt.learning.tpe import TreeParzenEstimator
from ultraopt.optimizer.base_opt import BaseOptimizer
from ultraopt.utils.config_space import add_configs_origin, initial_design_2, sample_configurations
from ultraopt.utils.config_transformer import ConfigTransformer


class ETPEOptimizer(BaseOptimizer):
    def __init__(
            self,
            # model related
            top_n_percent=15, min_points_in_kde=2,
            bw_method="scott", cv_times=100, kde_sample_weight_scaler=None,
            # several hyper-parameters
            gamma1=0.96, gamma2=3, max_bw_factor=4, min_bw_factor=1, max_try=3,
            min_points_in_model=20, min_n_candidates=8,
            n_candidates=None, n_candidates_factor=4, sort_by_EI=True,
            # Embedding Encoder
            embedding_encoder="default"
    ):
        super(ETPEOptimizer, self).__init__()
        self.min_bw_factor = min_bw_factor
        self.max_bw_factor = max_bw_factor
        self.embedding_encoder = embedding_encoder
        self.gamma1 = gamma1
        self.min_n_candidates = min_n_candidates
        self.max_try = max_try
        self.gamma2 = gamma2
        self.min_points_in_model = min_points_in_model
        self._bw_factor = max_bw_factor - min_bw_factor
        self.sort_by_EI = sort_by_EI
        self.n_candidates_factor = n_candidates_factor
        self.n_candidates = n_candidates
        self.tpe = TreeParzenEstimator(
            top_n_percent=top_n_percent,
            min_points_in_kde=min_points_in_kde,
            bw_method=bw_method,
            cv_times=cv_times,
            kde_sample_weight_scaler=kde_sample_weight_scaler
        )

    def initialize(self, config_space, budgets=(1,), random_state=42, initial_points=None, budget2obvs=None):
        super(ETPEOptimizer, self).initialize(config_space, budgets, random_state, initial_points, budget2obvs)
        if not self.embedding_encoder:
            # do not use embedding_encoder, use One Hot Encoder
            encoder = EquidistanceEncoder()
        elif isinstance(self.embedding_encoder, str):
            if self.embedding_encoder == "default":
                encoder = EmbeddingEncoder(
                    max_epoch=100, early_stopping_rounds=50, n_jobs=1, verbose=0)
            else:
                raise ValueError(f"Invalid Indicate string '{self.embedding_encoder}' for embedding_encoder'")
        else:
            encoder = self.embedding_encoder
        # todo: 如果自动构建了Embedding encoder， 后续需要保证initial point覆盖所有的类别
        # todo: auto_enrich_initial_points
        self.config_transformer = ConfigTransformer(impute=None, encoder=encoder)
        self.config_transformer.fit(config_space)
        if len(self.config_transformer.high_r_cols) == 0:
            self.config_transformer.encoder = None
        if self.embedding_encoder is None:
            vectors = np.array([config.get_array() for config in sample_configurations(self.config_space, 5000)])
            self.config_transformer.fit_encoder(vectors)
        self.budget2epm = {budget: None for budget in budgets}
        if self.n_candidates is None:
            self.n_candidates = max(
                self.config_transformer.n_variables_embedded * self.n_candidates_factor,
                self.min_n_candidates
            )
        # 初始化样本
        # todo: 考虑热启动时初始化得到的观测
        self.initial_design_configs = initial_design_2(self.config_space, self.min_points_in_model, self.rng)
        self.initial_design_ix = 0
        updated_min_points_in_model = len(self.initial_design_configs)
        if updated_min_points_in_model != self.min_points_in_model:
            self.logger.debug(f"Update min_points_in_model from {self.min_points_in_model} "
                              f"to {updated_min_points_in_model}")
            self.min_points_in_model = updated_min_points_in_model

    def tpe_sampling(self, epm, budget):
        info_dict = {"model_based_pick": True}
        for try_id in range(self.max_try):
            samples = epm.sample(
                n_candidates=self.n_candidates,
                sort_by_EI=self.sort_by_EI,
                random_state=self.rng,
                bandwidth_factor=self._bw_factor + self.min_bw_factor
            )
            for i, sample in enumerate(samples):
                if self.is_config_exist(budget, sample):
                    self.logger.debug(f"The sample already exists and needs to be resampled. "
                                      f"It's the {i}-th time sampling in thompson sampling. ")
                else:
                    add_configs_origin(sample, "ETPE sampling")
                    return sample, info_dict
            old_db = self._bw_factor
            self._bw_factor = (self._bw_factor + self.min_bw_factor) * self.gamma2 - self.min_bw_factor
            self.logger.warning(f"After {try_id + 1} times sampling, all samples exist in observations. "
                                f"Update bandwidth_factor from {old_db:.4f} to {self._bw_factor:.4f} by "
                                f"multiply gamma2 ({self.gamma2}).")
        sample = self.config_space.sample_configuration()
        add_configs_origin(sample, "Random Search")
        info_dict = {"model_based_pick": False}
        return sample, info_dict

    def _get_config(self, budget, max_budget):
        # choose model from max-budget
        epm = self.budget2epm[max_budget]
        # random sampling
        if epm is None:
            # return self.pick_random_initial_config(budget)
            info_dict = {"model_based_pick": False}
            if self.initial_design_ix < len(self.initial_design_configs):
                config = self.initial_design_configs[self.initial_design_ix]
                add_configs_origin(config, "Initial Design")
                self.initial_design_ix += 1
                return self.process_config_info_pair(config, info_dict, budget)
            else:
                return self.pick_random_initial_config(budget)
        # model based pick
        config, info_dict = self.tpe_sampling(epm, budget)
        self._bw_factor *= self.gamma1
        return self.process_config_info_pair(config, info_dict, budget)

    def get_available_max_budget(self):
        budgets = [budget for budget in self.budget2epm.keys() if budget > 0]
        sorted_budgets = sorted(budgets)
        for budget in sorted(budgets, reverse=True):
            if budget <= 0:
                continue
            if self.budget2epm[budget] is not None:
                return budget
        return sorted_budgets[0]

    def get_available_min_budget(self):
        budgets = [budget for budget in self.budget2epm.keys() if budget > 0]
        for budget in sorted(budgets):
            return budget
        return None

    def _new_result(self, budget, vectors: np.ndarray, losses: np.ndarray):
        if len(losses) < self.min_points_in_model:
            return
        vectors = self.budget2obvs[budget]["vectors"]
        losses = np.array(self.budget2obvs[budget]["losses"])
        # fit embedding encoder
        if self.has_embedding_encoder:
            if self.config_transformer.encoder.fitted:
                X = np.array([vectors[-1]])
                y = np.array([losses[-1]])
                self.config_transformer.fit_encoder(X, y)
            else:
                X = np.array(vectors)
                y = np.array(losses)
                self.config_transformer.fit_encoder(X, y)
            # todo: plot
        # fit epm
        if self.budget2epm[budget] is None:
            # new epm
            epm = deepcopy(self.tpe)
            epm.set_config_transformer(self.config_transformer)
        else:
            epm = self.budget2epm[budget]
        X_obvs = self.config_transformer.transform(vectors)
        self.budget2epm[budget] = epm.fit(X_obvs, losses)

    @property
    def has_embedding_encoder(self):
        return isinstance(self.config_transformer.encoder, EmbeddingEncoder) and \
               len(self.config_transformer.high_r_cols) > 0
