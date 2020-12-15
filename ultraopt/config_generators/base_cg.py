#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy

import numpy as np
from ConfigSpace import Configuration
from sklearn.utils.validation import check_random_state

from ultraopt.structure import Job
from ultraopt.utils.config_space import add_configs_origin
from ultraopt.utils.logging_ import get_logger


class BaseConfigGenerator():
    def __init__(
            self, config_space, budgets, random_state=42, initial_points=None, budget2obvs=None
    ):
        self.initial_points = initial_points
        self.random_state = random_state
        self.config_space = config_space
        self.config_space.seed(random_state)
        self.budgets = budgets
        if budget2obvs is None:
            budget2obvs = self.get_initial_budget2obvs(self.budgets)
        self.budget2obvs = budget2obvs
        # other variable
        self.rng = check_random_state(self.random_state)
        self.initial_points_index = 0
        self.logger = get_logger(self)

    @classmethod
    def get_initial_budget2obvs(cls, budgets):
        return {budget: {"losses": [], "configs": [], "vectors": [], "locks": []} for budget in budgets}

    def new_result(self, job: Job, update_model=True, should_update_weight=0):
        ##############################
        ### 1. update observations ###
        ##############################
        if job.result is None:
            # One could skip crashed results, but we decided to
            # assign a +inf loss and count them as bad configurations
            loss = np.inf
        else:
            # same for non numeric losses.
            # Note that this means losses of minus infinity will count as bad!
            loss = job.result["loss"] if np.isfinite(job.result["loss"]) else np.inf
        budget = job.kwargs["budget"]
        config_dict = job.kwargs["config"]
        config_info = job.kwargs["config_info"]
        config = Configuration(self.config_space, config_dict)
        # add lock (It may be added twice, but it does not affect)
        self.budget2obvs[budget]["locks"].append(config.get_array().copy())
        self.budget2obvs[budget]["configs"].append(deepcopy(config))
        self.budget2obvs[budget]["vectors"].append(config.get_array())
        self.budget2obvs[budget]["losses"].append(loss)
        losses = np.array(self.budget2obvs[budget]["losses"])
        vectors = np.array(self.budget2obvs[budget]["vectors"])
        ###################################################################
        ### 2. Judge whether the EPM training conditions are satisfied  ###
        ###################################################################
        if not update_model:
            return
        self.new_result_(budget, vectors, losses, update_model, should_update_weight)

    def new_result_(self, budget, vectors: np.ndarray, losses: np.ndarray, update_model=True, should_update_weight=0):
        raise NotImplementedError

    def get_config(self, budget):
        # get max_budget
        # calc by budget2epm
        max_budget = self.get_available_max_budget()
        # initial points
        if self.initial_points is not None and self.initial_points_index < len(self.initial_points):
            while True:
                if self.initial_points_index >= len(self.initial_points):
                    break
                initial_point_dict = self.initial_points[self.initial_points_index]
                initial_point = Configuration(self.config_space, initial_point_dict)
                self.initial_points_index += 1
                initial_point.origin = "User Defined"
                if not self.is_config_exist(budget, initial_point):
                    self.logger.info(f"Using initial points [{self.initial_points_index - 1}]")
                    return self.process_config_info_pair(initial_point, {}, budget)
        return self.get_config_(budget, max_budget)

    def get_config_(self, budget, max_budget):
        raise NotImplementedError

    def is_config_exist(self, budget, config: Configuration):
        vectors_list = []
        budgets = [budget_ for budget_ in list(self.budget2obvs.keys()) if budget_ >= budget]
        for budget_ in budgets:
            vectors = np.array(self.budget2obvs[budget_]["locks"])
            if vectors.size:
                vectors_list.append(vectors)
        if len(vectors_list) == 0:
            return False
        vectors = np.vstack(vectors_list)
        if np.any(np.array(vectors.shape) == 0):
            return False
        vectors[np.isnan(vectors)] = -1
        vector = config.get_array().copy()
        vector[np.isnan(vector)] = -1
        if np.any(np.all(vector == vectors, axis=1)):
            return True
        return False

    def get_available_max_budget(self):
        raise NotImplementedError

    def process_config_info_pair(self, config: Configuration, info_dict: dict, budget):
        self.budget2obvs[budget]["locks"].append(config.get_array().copy())
        info_dict = deepcopy(info_dict)
        if config.origin is None:
            config.origin = "unknown"
        info_dict.update({
            "origin": config.origin
        })
        return config.get_dictionary(), info_dict

    def process_all_configs_exist(self, info_dict, budget):
        seed = self.rng.randint(1, 8888)
        self.config_space.seed(seed)
        config = self.config_space.sample_configuration()
        add_configs_origin(config, "Initial Design")
        info_dict.update({"sampling_different_samples_failed": True, "seed": seed})
        return self.process_config_info_pair(config, info_dict, budget)

    def pick_random_initial_config(self, budget, max_sample=1000):
        i = 0
        info_dict = {"model_based_pick": False}
        while i < max_sample:
            i += 1
            config = self.config_space.sample_configuration()
            add_configs_origin(config, "Initial Design")
            if self.is_config_exist(budget, config):
                self.logger.info(f"The sample already exists and needs to be resampled. "
                                 f"It's the {i}-th time sampling in random sampling. ")
            else:
                return self.process_config_info_pair(config, info_dict, budget)
        return self.process_all_configs_exist(info_dict, budget)
