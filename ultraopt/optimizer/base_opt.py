#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
from collections import defaultdict
from copy import deepcopy
from math import inf
from time import time
from typing import Tuple, Union, List, Dict

import numpy as np
from ConfigSpace import Configuration
from sklearn.utils.validation import check_random_state

from ultraopt.structure import Job
from ultraopt.utils.config_space import add_configs_origin, get_dict_from_config
from ultraopt.utils.hash import get_hash_of_config
from ultraopt.utils.logging_ import get_logger


def runId_info():
    return {"start_time": time(), "end_time": -1}


class BaseOptimizer():
    def __init__(self):
        self.logger = get_logger(self)
        self.is_init = False
        self.configId2config: Dict[str, dict] = {}
        self.runId2info: Dict[Tuple[str, float], dict] = defaultdict(runId_info)

    def initialize(self, config_space, budgets=(1,), random_state=42, initial_points=None, budget2obvs=None):
        if self.is_init:
            return
        self.is_init = True
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

    @classmethod
    def get_initial_budget2obvs(cls, budgets):
        return {budget: {"losses": [], "configs": [], "vectors": [], "locks": []} for budget in budgets}

    def tell(self, config: Union[dict, Configuration], loss: float, budget: float = 1, update_model=True):
        config = get_dict_from_config(config)
        job = Job(get_hash_of_config(config))
        job.kwargs = {
            "budget": budget,
            "config": config,
            "config_info": {}
        }
        job.result = {
            "loss": loss
        }
        self.new_result(job, update_model=update_model)

    def new_result(self, job: Job, update_model=True):

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
        configId = get_hash_of_config(config_dict)
        runId = (configId, budget)
        if runId in self.runId2info:
            self.runId2info[runId]["end_time"] = time()
            self.runId2info[runId]["loss"] = loss
        else:
            self.logger.error(f"runId {runId} not in runId2info, it's impossible!!!")
        # config_info = job.kwargs["config_info"]
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
        self._new_result(budget, vectors, losses)

    def _new_result(self, budget, vectors: np.ndarray, losses: np.ndarray):
        raise NotImplementedError

    def ask(self, budget=1, n_points=None, strategy="cl_min") -> Union[List[Tuple[dict, dict]], Tuple[dict, dict]]:
        if n_points is None:
            return self.get_config(budget)
        supported_strategies = ["cl_min", "cl_mean", "cl_max"]

        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError(
                "n_points should be int > 0, got " + str(n_points)
            )

        if strategy not in supported_strategies:
            raise ValueError(
                "Expected parallel_strategy to be one of " +
                str(supported_strategies) + ", " + "got %s" % strategy
            )

        opt = deepcopy(self)
        config_info_pairs = []
        for i in range(n_points):
            start_time = time()
            config, config_info = opt.get_config(budget=budget)
            config_info_pairs.append((config, config_info))
            losses = opt.budget2obvs[budget]["losses"]
            if strategy == "cl_min":
                y_lie = np.min(losses) if losses else 0.0  # CL-min lie
            elif strategy == "cl_mean":
                y_lie = np.mean(losses) if losses else 0.0  # CL-mean lie
            elif strategy == "cl_max":
                y_lie = np.max(losses) if losses else 0.0  # CL-max lie
            else:
                raise NotImplementedError
            opt.tell(config, y_lie)
            self.register_config(config, budget, start_time=start_time)
        return config_info_pairs

    def get_config(self, budget) -> Tuple[dict, dict]:
        # get max_budget
        # calc by budget2epm
        start_time = time()
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
                    self.logger.debug(f"Using initial points [{self.initial_points_index - 1}]")
                    return self.process_config_info_pair(initial_point, {}, budget)
        config, config_info = self._get_config(budget, max_budget)
        self.register_config(config, budget, start_time)
        return config, config_info

    def register_config(self, config, budget, start_time=None):
        configId = get_hash_of_config(config)
        runId = (configId, budget)
        if runId in self.runId2info:  # don't set second time
            return
        self.configId2config[configId] = config
        info = self.runId2info[runId]  # auto set start_time
        if start_time:
            info["start_time"] = start_time

    def _get_config(self, budget, max_budget):
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

    def pick_random_initial_config(self, budget, max_sample=1000, origin="Initial Design"):
        i = 0
        info_dict = {"model_based_pick": False}
        while i < max_sample:
            i += 1
            config = self.config_space.sample_configuration()
            add_configs_origin(config, origin)
            if self.is_config_exist(budget, config):
                self.logger.debug(f"The sample already exists and needs to be resampled. "
                                  f"It's the {i}-th time sampling in random sampling. ")
            else:
                return self.process_config_info_pair(config, info_dict, budget)
        return self.process_all_configs_exist(info_dict, budget)

    def reset_time(self):
        min_time = inf
        for info in self.runId2info.values():
            min_time = min(info["start_time"], min_time)
        for info in self.runId2info.values():
            info["start_time"] -= min_time
            info["end_time"] -= min_time

    def resume_time(self):
        max_time = -inf
        for info in self.runId2info.values():
            max_time = max(info["end_time"], max_time)
        delta_time = time() - max_time
        for info in self.runId2info.values():
            info["start_time"] += delta_time
            info["end_time"] += delta_time
