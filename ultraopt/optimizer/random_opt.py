#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn

import numpy as np

from ultraopt.optimizer.base_opt import BaseOptimizer


class RandomOptimizer(BaseOptimizer):

    def _new_result(self, budget, vectors: np.ndarray, losses: np.ndarray):
        pass

    def _get_config(self, budget, max_budget):
        return self.pick_random_initial_config(budget, origin="Random Search")

    def get_available_max_budget(self):
        for budget in reversed(sorted(self.budgets)):
            if self.budget2obvs[budget]["losses"]:
                return budget
        return self.budgets[0]
