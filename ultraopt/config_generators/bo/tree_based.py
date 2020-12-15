#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : tqichun@gmail.com
from skopt.learning.forest import ExtraTreesRegressor, RandomForestRegressor
from skopt.learning.gbrt import GradientBoostingQuantileRegressor

from ultraopt.config_generators.bo.sampling_sort_cg import SamplingSortConfigGenerator
from ultraopt.utils.config_transformer import ConfigTransformer


class ForestConfigGenerator(SamplingSortConfigGenerator):
    def __init__(
            self,
            # basic params
            config_space, budgets, random_state=42, initial_points=None, budget2obvs=None,
            # model related
            forest_type="ET",
            n_estimators=10, max_depth=None, max_features="auto",
            # several hyper-parameters
            use_local_search=False, loss_transformer="log_scaled",
            min_points_in_model=15, n_samples=5000,
            acq_func="LogEI", xi=0,
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
        super(ForestConfigGenerator, self).__init__(
            config_space=config_space,
            budgets=budgets,
            random_state=random_state,
            initial_points=initial_points,
            budget2obvs=budget2obvs,
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
                random_state=random_state,
            ),
            config_transformer=ConfigTransformer(impute=1, encoder=None),
            use_local_search=use_local_search,
            loss_transformer=loss_transformer,
            min_points_in_model=min_points_in_model,
            n_samples=n_samples,
            acq_func=acq_func,
            xi=xi
        )


class GBRTConfigGenerator(SamplingSortConfigGenerator):
    def __init__(
            self,
            # basic params
            config_space, budgets, random_state=42, initial_points=None, budget2obvs=None,
            # model related
            n_jobs=1,
            # several hyper-parameters
            use_local_search=False, loss_transformer="log_scaled",
            min_points_in_model=15, n_samples=5000,
            acq_func="LogEI", xi=0

    ):
        super(GBRTConfigGenerator, self).__init__(
            config_space=config_space,
            budgets=budgets,
            random_state=random_state,
            initial_points=initial_points,
            budget2obvs=budget2obvs,
            epm=GradientBoostingQuantileRegressor(
                n_jobs=n_jobs, random_state=random_state,
            ),
            config_transformer=ConfigTransformer(impute=1, encoder=None),
            use_local_search=use_local_search,
            loss_transformer=loss_transformer,
            min_points_in_model=min_points_in_model,
            n_samples=n_samples,
            acq_func=acq_func,
            xi=xi
        )
