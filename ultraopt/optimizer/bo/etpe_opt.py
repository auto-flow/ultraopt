#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-15
# @Contact    : qichun.tang@bupt.edu.cn
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from tabular_nn import EmbeddingEncoder
from tabular_nn import EquidistanceEncoder
from ultraopt.optimizer.base_opt import BaseOptimizer
from ultraopt.tpe import SampleDisign
from ultraopt.tpe.estimator import TreeParzenEstimator
from ultraopt.transform.config_transformer import ConfigTransformer
from ultraopt.utils.config_space import add_configs_origin, initial_design_2, sample_configurations
from ultraopt.utils.misc import content_to_dfMap


class ETPEOptimizer(BaseOptimizer):
    def __init__(
            self,
            # TPE model related
            gamma=None, min_points_in_kde=2,
            multivariate=True,
            embed_catVar=True,
            # if config space is all joint and dim > 10, and hyperparameter are Homogeneous, use it
            limit_max_groups='auto',
            max_groups=3,  # optimal meta-parameter by a lot of experiments
            optimize_each_varGroups=False,
            # fixme : 这个超参和 limit_max_groups 机制都值得深入探究
            overlap_bagging_ratio=0,
            bw_method="scott", cv_times=100, kde_sample_weight_scaler=None,
            # several hyper-parameters
            max_bw_factor=4, min_bw_factor=1.2,
            anneal_steps=15,
            min_points_in_model=20, min_n_candidates=8,
            n_candidates=None, n_candidates_factor=4, sort_by_EI=True,
            specific_sample_design=(
                    SampleDisign(ratio=0.1, is_random=True),
                    SampleDisign(ratio=0.2, bw_factor=3),
            ),
            # embedding
            embedding_encoder="default",
            pretrained_emb=None,
            pretrained_emb_expire_iter=-1
    ):
        super(ETPEOptimizer, self).__init__()
        self.pretrained_emb_expire_iter = pretrained_emb_expire_iter
        self.optimize_each_varGroups = optimize_each_varGroups
        self.max_groups = max_groups
        self.specific_sample_design = specific_sample_design
        self.embed_catVar = embed_catVar
        assert isinstance(overlap_bagging_ratio, (int, float)) and 0 <= overlap_bagging_ratio <= 1
        self.overlap_bagging_ratio = overlap_bagging_ratio
        self.multivariate = multivariate
        self.min_bw_factor = min_bw_factor
        self.max_bw_factor = max_bw_factor
        self.embedding_encoder = embedding_encoder
        # self.lambda_ = lambda_
        # fixme: 推公式确认一下min_bw_factor
        assert anneal_steps >= 0
        if anneal_steps > 0:
            self.lambda_ = np.exp((1 / anneal_steps) * np.log(min_bw_factor / max_bw_factor))
        else:
            self.lambda_ = 1
        if pretrained_emb is not None:
            if isinstance(pretrained_emb, str):
                pretrained_emb = content_to_dfMap(Path(pretrained_emb).read_text())
        self.pretrained_emb = pretrained_emb
        self.min_n_candidates = min_n_candidates
        self.MAX_TRY = 3  # fixme: 硬编码
        self.min_points_in_model = min_points_in_model
        self._bw_factor = max_bw_factor
        self.sort_by_EI = sort_by_EI
        self.n_candidates_factor = n_candidates_factor
        self.n_candidates = n_candidates
        self.limit_max_groups = limit_max_groups
        self.max_groups = max_groups
        self.tpe = TreeParzenEstimator(
            gamma=gamma,
            min_points_in_kde=min_points_in_kde,
            bw_method=bw_method,
            cv_times=cv_times,
            kde_sample_weight_scaler=kde_sample_weight_scaler,
            overlap_bagging_ratio=overlap_bagging_ratio,
            multivariate=multivariate,
            embed_catVar=embed_catVar,
            limit_max_groups=limit_max_groups,
            max_groups=max_groups
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

        self.config_transformer = ConfigTransformer(
            impute=None, encoder=encoder, pretrained_emb=self.pretrained_emb
        )
        self.config_transformer.fit(config_space)
        if len(self.config_transformer.high_r_cols) == 0:
            self.config_transformer.encoder = None
        encoder = self.config_transformer.encoder
        # fixme: 对于EmbeddingEncoder 需要这么做吗
        if isinstance(encoder, EquidistanceEncoder):
            vectors = np.array([config.get_array() for config in sample_configurations(self.config_space, 5000)])
            self.config_transformer.fit_encoder(vectors)
        self.budget2epm = {budget: None for budget in budgets}
        if self.n_candidates is None:
            # 对于结构空间，通过采样的方法得到一个平均的变量长度
            N = 100
            cs = deepcopy(self.config_transformer.config_space)
            vec = np.array(
                [config.get_array() for config in cs.sample_configuration(N)])
            mask = ~pd.isna(vec)
            n_variables_embedded_list = np.array(self.config_transformer.n_variables_embedded_list)
            n_variables = round(np.sum([np.sum(n_variables_embedded_list[row]) for row in mask]) / N)
            self.n_candidates = max(
                n_variables * self.n_candidates_factor,
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
        # process 'limit_max_groups'
        assert isinstance(self.limit_max_groups, (str, int))
        if isinstance(self.limit_max_groups, (str)): assert self.limit_max_groups == 'auto'
        if self.limit_max_groups == 'auto':
            ct = self.config_transformer
            # is high-dimension?
            self.limit_max_groups = True
            if ct.n_variables_embedded < 10:
                self.limit_max_groups = False
            # is all joint?
            if ct.n_top_levels != ct.n_variables:
                self.limit_max_groups = False
            # is Homogeneous
            is_Homogeneous = (int(ct.n_ordinal > 0) + int(ct.n_categorical > 0) + int(ct.n_numerical > 0)) == 1
            if not is_Homogeneous:
                self.limit_max_groups = False
            signal = 'enable' if self.limit_max_groups else 'disable'
            self.logger.info(f'{signal} limit_max_groups.')
            if self.limit_max_groups:
                assert self.max_groups >= 3

    def tpe_sampling(self, epm, budget):
        info_dict = {"model_based_pick": True}
        for try_id in range(self.MAX_TRY):
            samples = epm.sample(
                n_candidates=round(self.n_candidates),
                sort_by_EI=self.sort_by_EI,
                random_state=self.rng,
                bandwidth_factor=self._bw_factor,
                specific_sample_design=self.specific_sample_design,
                optimize_each_varGroups=self.optimize_each_varGroups
            )
            for i, sample in enumerate(samples):
                if self.is_config_exist(budget, sample):
                    self.logger.debug(f"The sample already exists and needs to be resampled. "
                                      f"It's the {i}-th time sampling in thompson sampling. ")
                else:
                    add_configs_origin(sample, "ETPE sampling")
                    return sample, info_dict
            # fixme: 放大策略
            self._bw_factor = self.max_bw_factor
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
        self._bw_factor = max(self.lambda_ * self._bw_factor, self.min_bw_factor)
        # self._bw_factor *= self.lambda_
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
        nn_info = self.budget2obvs[budget]["nn_info"]
        y_ = np.hstack([losses[:, None]] + [np.array(v)[:, None] for k, v in nn_info.items()])
        if y_.ndim == 1:
            y_ = y_[:, np.newaxis]
        # fit embedding encoder
        if self.has_embedding_encoder:
            if self.config_transformer.encoder.fitted:
                X = np.array([vectors[-1]])
                y = y_[y_.shape[0] - 1, :][np.newaxis, :]
                self.config_transformer.fit_encoder(X, y)
            else:
                X = np.array(vectors)
                y = y_
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
        # TPE自适应分组联合概率的逻辑应该写在这
        self.budget2epm[budget] = epm.fit(X_obvs, losses)
        # 预训练Embedding过期
        if self.pretrained_emb_expire_iter > 0:
            if len(self.budget2obvs[1]['losses']) > self.pretrained_emb_expire_iter and \
                    isinstance(self.config_transformer.encoder, EmbeddingEncoder):
                self.config_transformer.encoder.pretrained_emb = {}

    @property
    def has_embedding_encoder(self):
        return isinstance(self.config_transformer.encoder, EmbeddingEncoder) and \
               len(self.config_transformer.high_r_cols) > 0
