#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : tqichun@gmail.com
from copy import deepcopy

from skopt.learning.forest import ExtraTreesRegressor

from oxygen.config_generators.base_cg import BaseConfigGenerator
from oxygen.utils.config_transformer import ConfigurationTransformer
from oxygen.utils.loss_transformer import LossTransformer, LogScaledLossTransformer, ScaledLossTransformer


class SamplingSortConfigGenerator(BaseConfigGenerator):
    def __init__(
            self, config_space, budgets, random_state=42, initial_points=None, budget2obvs=None,
            epm=None, min_points_in_model=15, config_transformer=None,
            n_samples=5000, loss_transformer=None, use_local_search=False,
    ):
        super(SamplingSortConfigGenerator, self).__init__(config_space, budgets, random_state, initial_points,
                                                          budget2obvs)
        # ----------member variables-----------------
        self.use_local_search = use_local_search
        self.n_samples = n_samples
        self.min_points_in_model = min_points_in_model
        # ----------components-----------------
        # experiment performance model
        self.epm = epm if epm is not None else ExtraTreesRegressor()
        self.emp_copied = deepcopy(self.epm)
        # config transformer
        self.config_transformer = config_transformer if config_transformer is not None else ConfigurationTransformer()
        # loss transformer
        if loss_transformer is None:
            self.loss_transformer = LossTransformer()
        elif loss_transformer == "log_scaled":
            self.loss_transformer = LogScaledLossTransformer()
        elif loss_transformer == "scaled":
            self.loss_transformer = ScaledLossTransformer()
        else:
            raise NotImplementedError

