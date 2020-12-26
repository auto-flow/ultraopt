#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : qichun.tang@bupt.edu.cn
from skopt.learning.forest import ExtraTreesRegressor, RandomForestRegressor
from skopt.learning.gbrt import GradientBoostingQuantileRegressor

from ultraopt.optimizer.bo.sampling_sort_opt import SamplingSortOptimizer
from ultraopt.utils.config_transformer import ConfigTransformer


class ForestOptimizer(SamplingSortOptimizer):
    def __init__(
            self,
            # model related
            forest_type="ET",
            n_estimators=10, max_depth=None, max_features="auto",
            # several hyper-parameters
            use_local_search=False, loss_transformer="log_scaled",
            min_points_in_model=20, n_samples=5000,
            acq_func="EI", xi=0,
            # model related (trival)
            min_samples_leaf=1, min_weight_fraction_leaf=0, min_samples_split=2,
            max_leaf_nodes=None, min_impurity_decrease=0, bootstrap=True, oob_score=False,
            n_jobs=1, min_variance=0,

    ):

        if forest_type == "ET":
            forest_klass = ExtraTreesRegressor
        elif forest_type == "RF":
            forest_klass = RandomForestRegressor
        else:
            raise ValueError(
                f"forest_type should be 'ET' or 'RF', means 'ExtraTrees' and 'RandomForest', respectively. ")
        super(ForestOptimizer, self).__init__(
            epm=forest_klass(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                min_variance=min_variance,
                # random_state=random_state,
            ),
            config_transformer=ConfigTransformer(impute=-1, encoder=None),
            use_local_search=use_local_search,
            loss_transformer=loss_transformer,
            min_points_in_model=min_points_in_model,
            n_samples=n_samples,
            acq_func=acq_func,
            xi=xi
        )

    def initialize(self, config_space, budgets=(1,), random_state=42, initial_points=None, budget2obvs=None):
        super(ForestOptimizer, self).initialize(config_space, budgets, random_state, initial_points,
                                                budget2obvs)
        self.epm.set_params(random_state=random_state)


class GBRTOptimizer(SamplingSortOptimizer):
    def __init__(
            self,
            # model related
            n_jobs=1,
            # several hyper-parameters
            use_local_search=False, loss_transformer="log_scaled",
            min_points_in_model=20, n_samples=5000,
            acq_func="EI", xi=0
    ):
        super(GBRTOptimizer, self).__init__(
            epm=GradientBoostingQuantileRegressor(
                n_jobs=n_jobs,
            ),
            config_transformer=ConfigTransformer(impute=-1, encoder=None),
            use_local_search=use_local_search,
            loss_transformer=loss_transformer,
            min_points_in_model=min_points_in_model,
            n_samples=n_samples,
            acq_func=acq_func,
            xi=xi
        )

    def initialize(self, config_space, budgets=(1,), random_state=42, initial_points=None, budget2obvs=None):
        super(GBRTOptimizer, self).initialize(config_space, budgets, random_state, initial_points,
                                              budget2obvs)
        self.epm.set_params(random_state=random_state)
