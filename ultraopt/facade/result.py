#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-19
# @Contact    : qichun.tang@bupt.edu.cn
from copy import deepcopy
from functools import lru_cache

import numpy as np
from terminaltables import AsciiTable

from ultraopt.facade.utils import get_wanted
from ultraopt.optimizer.base_opt import BaseOptimizer
from ultraopt.utils.misc import pbudget, get_import_error


class FMinResult():
    def __init__(self, optimizer: BaseOptimizer):
        self.optimizer = optimizer
        self.configs_table = []
        self.hyperparameters = [hp.name for hp in optimizer.config_space.get_hyperparameters()]
        self.is_multi_fidelity = len(optimizer.budgets) > 1
        self.budget2obvs = optimizer.budget2obvs
        self.budget2info = {}
        for budget, obvs in self.budget2obvs.items():
            losses = obvs["losses"]
            configs = obvs["configs"]
            if len(losses):
                ix = np.argmin(losses)
                loss = losses[ix]
                config = configs[ix]
            else:
                loss = None
                config = None
            self.budget2info[budget] = {"loss": loss, "config": config, "num_configs": len(configs)}
        for hyperparameter in self.hyperparameters:
            row = []
            row.append(hyperparameter)
            for budget in self.budget2info:
                config = self.budget2info[budget]["config"]
                nil = "-"
                if config is not None:
                    val = config.get(hyperparameter, None)
                else:
                    val = config = nil
                if val is None:
                    val = nil
                if isinstance(val, float):
                    val = f"{val:.4f}"
                elif not isinstance(val, str):
                    val = str(val)
                row.append(val)
            self.configs_table.append(row)
        self.configs_title = ["HyperParameters"] + ["" if i else "Optimal Value" for i, _ in
                                                    enumerate(self.budget2info)]
        self.max_budget, self.best_loss, self.best_config = get_wanted(self.optimizer)

    @lru_cache(None)
    def get_str(self):
        # todo: 更好看的打印
        table_data = ([self.configs_title] +
                      self.configs_table +
                      [["Optimal Loss"] + [f"{self.budget2info[budget]['loss']:.4f}" for budget in self.budget2info]] +
                      [["Num Configs"] + [str(self.budget2info[budget]["num_configs"]) for budget in self.budget2info]])
        if self.is_multi_fidelity:
            M = 3
            table_data.insert(-2, ["Budgets"] + [
                f"{pbudget(budget)} (max)" if budget == self.max_budget else pbudget(budget)
                for budget in self.budget2info])
        else:
            M = 2

        raw_table = AsciiTable(
            table_data
            # title="Result of UltraOpt's fmin"
        ).table
        lines = raw_table.splitlines()
        title_line = lines[1]
        st = title_line.index("|", 1)
        col = "Optimal Value"
        L = len(title_line)
        lines[0] = "+" + "-" * (L - 2) + "+"
        new_title_line = title_line[:st + 1] + (" " + col + " " * (L - st - 3 - len(col))) + "|"
        lines[1] = new_title_line
        bar = "\n" + lines.pop() + "\n"
        finals = lines[-M:]
        prevs = lines[:-M]
        render_table = "\n".join(prevs) + bar + bar.join(finals) + bar
        return render_table

    def __str__(self):
        return self.get_str()

    __repr__ = __str__

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def plot_hi(self, budget=None, target_name="loss", loss2target_func=None, return_data_only=False):
        if budget is None:
            budget = self.max_budget
        losses = deepcopy(self.budget2obvs[budget]["losses"])
        data = deepcopy([config.get_dictionary() for config in self.budget2obvs[budget]["configs"]])
        if loss2target_func is not None:
            targets = [loss2target_func(loss) for loss in losses]
        else:
            targets =losses
        for config, target in zip(data, targets):
            config[target_name]=target
        if return_data_only:
            return data
        try:
            import hiplot as hip
        except Exception:
            raise get_import_error("hiplot")
        return hip.Experiment.from_iterable(data).display()

