#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-11-30
# @Contact    : qichun.tang@bupt.edu.cn
from copy import deepcopy

import numpy as np
from ConfigSpace import Configuration
from sklearn.utils import check_random_state
from ultraopt.tpe import SampleDisign
from ultraopt.tpe.estimator import TreeParzenEstimator
from ultraopt.transform.config_transformer import ConfigTransformer


def extract_matrix(cs, configs):
    return np.array(
        [(config if isinstance(config, Configuration) else Configuration(cs, values=config)).get_array() for config in
         configs])


class MetaLearning():
    def __init__(
            self,
            good_configs, bad_configs, weight=1,
            random_startups=0,
            bw_factor=None,
            specific_sample_design=(
                    SampleDisign(ratio=0.5, is_random=True),
                    SampleDisign(ratio=0.5, bw_factor=4),
            )
    ):
        self.bw_factor = bw_factor
        self.random_startups = random_startups
        self.specific_sample_design = specific_sample_design
        self.weight = weight
        self.bad_configs = bad_configs
        self.good_configs = good_configs

    def init(self, tpe: TreeParzenEstimator, config_transformer: ConfigTransformer, min_points_in_model, random_state,
             n_candidates):
        self.min_points_in_model = min_points_in_model
        tpe = deepcopy(tpe)
        tpe.set_config_transformer(config_transformer)
        cs = config_transformer.config_space
        self.config_space=cs
        X_good = extract_matrix(cs, self.good_configs)
        X_bad = extract_matrix(cs, self.bad_configs)
        tpe.fit(X=X_good, X_bad=X_bad, bw_factor=self.bw_factor)
        self.tpe = tpe
        self.X_good = X_good
        self.X_bad = X_bad
        self.rng = check_random_state(random_state)
        self.n_candidates = n_candidates
        self.random_idx = 0

    def sample(self) -> Configuration:
        if self.random_idx<self.random_startups:
            self.random_idx+=1
            return self.config_space.sample_configuration()

        return self.tpe.sample(
            n_candidates=self.n_candidates,
            sort_by_EI=True,
            random_state=self.rng,
            specific_sample_design=self.specific_sample_design
        )[0]

    def merge_predict(self, X: np.ndarray, logEI: np.ndarray, n_samples: int):
        EI_a = np.exp(self.tpe.predict(X))
        EI_b = np.exp(logEI)
        weight = self.min_points_in_model / n_samples
        weight *= self.weight
        return np.log(weight * EI_a + EI_b)
