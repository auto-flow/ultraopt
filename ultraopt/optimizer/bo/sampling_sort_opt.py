#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : qichun.tang@bupt.edu.cn

import itertools
from copy import deepcopy
from functools import partial
from typing import Tuple, List

import numpy as np
from ConfigSpace import Configuration
from ConfigSpace.util import get_one_exchange_neighbourhood
from skopt.learning.forest import ExtraTreesRegressor

from ultraopt.optimizer.base_opt import BaseOptimizer
from ultraopt.optimizer.bo.config_evaluator import ConfigEvaluator
from ultraopt.utils.config_space import add_configs_origin
from ultraopt.utils.config_transformer import ConfigTransformer
from ultraopt.utils.loss_transformer import LossTransformer, LogScaledLossTransformer, ScaledLossTransformer

get_one_exchange_neighbourhood = partial(get_one_exchange_neighbourhood, stdev=0.05, num_neighbors=8)


class SamplingSortOptimizer(BaseOptimizer):
    def __init__(
            self,
            # model related
            epm=None, config_transformer=None,
            # several hyper-parameters
            use_local_search=False, loss_transformer="log_scaled",
            min_points_in_model=15, n_samples=5000,
            acq_func="LogEI", xi=0
    ):
        super(SamplingSortOptimizer, self).__init__()
        # ----------member variables-----------------
        self.xi = xi
        self.acq_func = acq_func
        self.use_local_search = use_local_search
        self.n_samples = n_samples
        self.min_points_in_model = min_points_in_model
        # ----------components-----------------
        # experiment performance model
        self.epm = epm if epm is not None else ExtraTreesRegressor()
        # config transformer
        self.config_transformer = config_transformer if config_transformer is not None else ConfigTransformer()
        # loss transformer
        if loss_transformer is None:
            self.loss_transformer = LossTransformer()
        elif loss_transformer == "log_scaled":
            self.loss_transformer = LogScaledLossTransformer()
        elif loss_transformer == "scaled":
            self.loss_transformer = ScaledLossTransformer()
        else:
            raise NotImplementedError

    def initialize(self, config_space, budgets=(1,), random_state=42, initial_points=None, budget2obvs=None):
        super(SamplingSortOptimizer, self).initialize(config_space, budgets, random_state, initial_points, budget2obvs)
        self.budget2epm = {budget: None for budget in budgets}
        self.config_transformer.fit(config_space)
        self.budget2confevt = {}
        for budget in budgets:
            config_evaluator = ConfigEvaluator(self.budget2epm, budget, self.acq_func, {"xi": self.xi})
            self.budget2confevt[budget] = config_evaluator
        self.update_weight_cnt = 0

    def _new_result(self, budget, vectors: np.ndarray, losses: np.ndarray):
        if len(losses) < self.min_points_in_model:
            return
        X_obvs = self.config_transformer.transform(vectors)
        y_obvs = self.loss_transformer.fit_transform(losses)
        if self.budget2epm[budget] is None:
            epm = deepcopy(self.epm)
        else:
            epm = self.budget2epm[budget]
        self.budget2epm[budget] = epm.fit(X_obvs, y_obvs)

    def _get_config(self, budget, max_budget):
        # choose model from max-budget
        epm = self.budget2epm[max_budget]
        # random sampling
        if epm is None:
            return self.pick_random_initial_config(budget)
        # model based pick
        info_dict = {"model_based_pick": True}
        # using config_evaluator evaluate random samples
        configs = self.config_space.sample_configuration(self.n_samples)
        losses, configs_sorted = self.evaluate(configs, max_budget, return_loss_config=True)
        add_configs_origin(configs_sorted, "Random Search (Sorted)")
        if self.use_local_search:
            start_points = self.get_local_search_initial_points(max_budget, 10, configs_sorted)
            local_losses, local_configs = self.local_search(start_points,
                                                            max_budget)
            add_configs_origin(local_configs, "Local Search")
            concat_losses = np.hstack([losses.flatten(), local_losses.flatten()])
            concat_configs = configs + local_configs
            random_var = self.rng.rand(len(concat_losses))
            indexes = np.lexsort((random_var.flatten(), concat_losses))
            concat_configs_sorted = [concat_configs[i] for i in indexes]
            concat_losses = concat_losses[indexes]
        else:
            concat_losses, concat_configs_sorted = losses, configs_sorted
        # 选取获益最大，且没有出现过的一个配置
        for i, config in enumerate(concat_configs_sorted):
            if self.is_config_exist(budget, config):
                self.logger.debug(f"The sample already exists and needs to be resampled. "
                                 f"It's the {i}-th time sampling in bayesian sampling. ")
            else:
                return self.process_config_info_pair(config, info_dict, budget)
        return self.process_all_configs_exist(info_dict, budget)

    def get_available_max_budget(self):
        budgets = [budget for budget in self.budget2epm.keys() if budget > 0]
        sorted_budgets = sorted(budgets)
        for budget in sorted(budgets, reverse=True):
            if budget <= 0:
                continue
            if self.budget2epm[budget] is not None:
                return budget
        return sorted_budgets[0]

    def get_local_search_initial_points(self, budget, num_points, additional_start_points):
        # 对之前的样本做评价
        # 1. 按acq排序，前num_points的历史样本
        config_evaluator = self.budget2confevt[budget]
        configs_previous_runs = self.budget2obvs[budget]["configs"]
        X_trans = self.transform(configs_previous_runs)
        y_opt = np.min(self.budget2obvs[budget]["losses"])
        rewards = config_evaluator(X_trans, y_opt)
        # 只取前num_points的样本
        random_var = self.rng.rand(len(rewards))
        indexes = np.lexsort((random_var.flatten(), -rewards.flatten()))
        configs_previous_runs_sorted_by_acq = [configs_previous_runs[ix] for ix in indexes[:num_points]]
        # 2. 按loss排序，前num_points的历史样本
        losses = np.array(self.budget2obvs[budget]["losses"])
        random_var = self.rng.rand(len(losses))
        indexes = np.lexsort((random_var.flatten(), losses.flatten()))
        configs_previous_runs_sorted_by_loss = [configs_previous_runs[ix] for ix in indexes[:num_points]]
        additional_start_points = additional_start_points[:num_points]
        init_points = []
        init_points_as_set = set()
        for cand in itertools.chain(
                configs_previous_runs_sorted_by_acq,
                configs_previous_runs_sorted_by_loss,
                additional_start_points,
        ):
            if cand not in init_points_as_set:
                init_points.append(cand)
                init_points_as_set.add(cand)

        return init_points

    def get_y_opt(self, budget):
        y_opt = np.min(self.budget2obvs[budget]["losses"])
        return y_opt

    def transform(self, configs: List[Configuration]):
        X = np.array([config.get_array() for config in configs], dtype="float32")
        X_trans = self.config_transformer.transform(X)
        return X_trans

    def local_search(
            self,
            start_points: List[Configuration],
            budget,
    ) -> Tuple[np.ndarray, List[Configuration]]:
        y_opt = self.get_y_opt(budget)
        # Compute the acquisition value of the incumbents
        num_incumbents = len(start_points)
        acq_val_incumbents_, incumbents = self.evaluate(deepcopy(start_points), budget, y_opt,
                                                        return_loss_config=True)
        acq_val_incumbents: list = acq_val_incumbents_.tolist()

        # Set up additional variables required to do vectorized local search:
        # whether the i-th local search is still running
        active = [True] * num_incumbents
        # number of plateau walks of the i-th local search. Reaching the maximum number is the stopping criterion of
        # the local search.
        n_no_plateau_walk = [0] * num_incumbents
        # tracking the number of steps for logging purposes
        local_search_steps = [0] * num_incumbents
        # tracking the number of neighbors looked at for logging purposes
        neighbors_looked_at = [0] * num_incumbents
        # tracking the number of neighbors generated for logging purposse
        neighbors_generated = [0] * num_incumbents
        # how many neighbors were obtained for the i-th local search. Important to map the individual acquisition
        # function values to the correct local search run
        # todo
        self.vectorization_min_obtain = 2
        self.n_steps_plateau_walk = 10
        self.vectorization_max_obtain = 64
        # todo
        obtain_n = [self.vectorization_min_obtain] * num_incumbents
        # Tracking the time it takes to compute the acquisition function
        times = []

        # Set up the neighborhood generators
        neighborhood_iterators = []
        for i, inc in enumerate(incumbents):
            neighborhood_iterators.append(get_one_exchange_neighbourhood(
                inc, seed=self.rng.randint(low=0, high=100000)))
            local_search_steps[i] += 1
        # Keeping track of configurations with equal acquisition value for plateau walking
        neighbors_w_equal_acq = [[]] * num_incumbents

        num_iters = 0
        while np.any(active):

            num_iters += 1
            # Whether the i-th local search improved. When a new neighborhood is generated, this is used to determine
            # whether a step was made (improvement) or not (iterator exhausted)
            improved = [False] * num_incumbents
            # Used to request a new neighborhood for the incumbent of the i-th local search
            new_neighborhood = [False] * num_incumbents

            # gather all neighbors
            neighbors = []
            for i, neighborhood_iterator in enumerate(neighborhood_iterators):
                if active[i]:
                    neighbors_for_i = []
                    for j in range(obtain_n[i]):
                        try:
                            n = next(neighborhood_iterator)  # n : Configuration
                            neighbors_generated[i] += 1
                            neighbors_for_i.append(n)
                        except StopIteration:
                            obtain_n[i] = len(neighbors_for_i)
                            new_neighborhood[i] = True
                            break
                    neighbors.extend(neighbors_for_i)

            if len(neighbors) != 0:
                acq_val = self.evaluate(neighbors, budget, return_loss=True)
                if np.ndim(acq_val.shape) == 0:
                    acq_val = [acq_val]

                # Comparing the acquisition function of the neighbors with the acquisition value of the incumbent
                acq_index = 0
                # Iterating the all i local searches
                for i in range(num_incumbents):
                    if not active[i]:
                        continue
                    # And for each local search we know how many neighbors we obtained
                    for j in range(obtain_n[i]):
                        # The next line is only true if there was an improvement and we basically need to iterate to
                        # the i+1-th local search
                        if improved[i]:
                            acq_index += 1
                        else:
                            neighbors_looked_at[i] += 1

                            # Found a better configuration
                            if acq_val[acq_index] < acq_val_incumbents[i]:
                                self.logger.debug(
                                    "Local search %d: Switch to one of the neighbors (after %d configurations).",
                                    i,
                                    neighbors_looked_at[i],
                                )
                                incumbents[i] = neighbors[acq_index]
                                acq_val_incumbents[i] = acq_val[acq_index]
                                new_neighborhood[i] = True
                                improved[i] = True
                                local_search_steps[i] += 1
                                neighbors_w_equal_acq[i] = []
                                obtain_n[i] = 1
                            # Found an equally well performing configuration, keeping it for plateau walking
                            elif acq_val[acq_index] == acq_val_incumbents[i]:
                                neighbors_w_equal_acq[i].append(neighbors[acq_index])

                            acq_index += 1

            # Now we check whether we need to create new neighborhoods and whether we need to increase the number of
            # plateau walks for one of the local searches. Also disables local searches if the number of plateau walks
            # is reached (and all being switched off is the termination criterion).
            for i in range(num_incumbents):
                if not active[i]:
                    continue
                if obtain_n[i] == 0 or improved[i]:
                    obtain_n[i] = 2
                else:
                    obtain_n[i] = obtain_n[i] * 2
                    obtain_n[i] = min(obtain_n[i], self.vectorization_max_obtain)
                if new_neighborhood[i]:
                    if not improved[i] and n_no_plateau_walk[i] < self.n_steps_plateau_walk:
                        if len(neighbors_w_equal_acq[i]) != 0:
                            incumbents[i] = neighbors_w_equal_acq[i][0]
                            neighbors_w_equal_acq[i] = []
                        n_no_plateau_walk[i] += 1
                    if n_no_plateau_walk[i] >= self.n_steps_plateau_walk:
                        active[i] = False
                        continue

                    neighborhood_iterators[i] = get_one_exchange_neighbourhood(
                        incumbents[i], seed=self.rng.randint(low=0, high=100000),
                    )

        self.logger.debug(
            "Local searches took %s steps and looked at %s configurations. Computing the acquisition function in "
            "vectorized for took %f seconds on average.",
            local_search_steps, neighbors_looked_at, np.mean(times),
        )
        # todo: origin 标注来自局部搜索
        return np.array(acq_val_incumbents), incumbents
        # return [(a, i) for a, i in zip(acq_val_incumbents, incumbents)]

    def evaluate(self, configs: List[Configuration], budget, y_opt=None,
                 return_loss_config_pairs=False, return_loss=False, return_loss_config=False):
        config_evaluator = self.budget2confevt[budget]
        if isinstance(configs, Configuration):
            configs = [configs]
        X_trans = self.transform(configs)
        if y_opt is None:
            y_opt = self.get_y_opt(budget)
        rewards = config_evaluator(X_trans, y_opt)
        random_var = self.rng.rand(len(rewards))
        indexes = np.lexsort((random_var.flatten(), -rewards.flatten()))
        rewards_sorted = rewards[indexes]
        configs_sorted = [configs[ix] for ix in indexes]
        if return_loss_config_pairs:
            return list(zip(-rewards_sorted, configs_sorted))
        if return_loss:
            return -rewards
        if return_loss_config:
            return -rewards_sorted, configs_sorted
        return configs_sorted

    # todo: develop and test weight update
