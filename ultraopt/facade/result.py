#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-19
# @Contact    : qichun.tang@bupt.edu.cn
import itertools
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
from terminaltables import AsciiTable

from ultraopt.facade.utils import get_wanted
from ultraopt.optimizer.base_opt import BaseOptimizer
from ultraopt.utils.misc import pbudget, get_import_error
from ultraopt.viz import plot_convergence


class FMinResult():
    def __init__(self, optimizer: BaseOptimizer):
        self.optimizer = deepcopy(optimizer)
        self.configs_table = []
        self.hyperparameters = [hp.name for hp in optimizer.config_space.get_hyperparameters()]
        self.is_multi_fidelity = len(optimizer.budgets) > 1
        self.budget2obvs = optimizer.budget2obvs
        self.budget2info = {}
        self.optimizer.reset_time()
        self.runId2info = self.optimizer.runId2info
        self.budgets = sorted(list(self.budget2obvs.keys()))

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
            targets = losses
        for config, target in zip(data, targets):
            config[target_name] = target
        if return_data_only:
            return data
        try:
            import hiplot as hip
        except Exception:
            raise get_import_error("hiplot")
        return hip.Experiment.from_iterable(data).display()

    def plot_convergence(
            self,
            budget=None,
            xlabel="Number of iterations $n$",
            ylabel=r"$\min f(x)$ after $n$ iterations",
            ax=None, name=None, alpha=0.2, yscale=None,
            color=None, true_minimum=None,
            **kwargs):
        """Plot one or several convergence traces.

        Parameters
        ----------
        args[i] :  `OptimizeResult`, list of `OptimizeResult`, or tuple
            The result(s) for which to plot the convergence trace.

            - if `OptimizeResult`, then draw the corresponding single trace;
            - if list of `OptimizeResult`, then draw the corresponding convergence
              traces in transparency, along with the average convergence trace;
            - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
              an `OptimizeResult` or a list of `OptimizeResult`.

        ax : `Axes`, optional
            The matplotlib axes on which to draw the plot, or `None` to create
            a new one.

        true_minimum : float, optional
            The true minimum value of the function, if known.

        yscale : None or string, optional
            The scale for the y-axis.

        Returns
        -------
        ax : `Axes`
            The matplotlib axes.
        """
        if budget is None:
            budget = self.max_budget
        losses = deepcopy(self.budget2obvs[budget]["losses"])

        n_calls = len(losses)
        iterations = range(1, n_calls + 1)
        mins = [np.min(losses[:i]) for i in iterations]
        max_mins = max(mins)
        cliped_losses = np.clip(losses, None, max_mins)
        return plot_convergence(iterations, mins, cliped_losses, xlabel, ylabel, ax, name, alpha, yscale, color,
                                true_minimum, **kwargs)

    plot_convergence_over_iter = plot_convergence

    def plot_convergence_over_time(
            self,
            xlabel="time [s]",
            ylabel=r"$\min f(x)$ over time",
            ax=None, names=None, alpha=0.2, yscale=None,
            colors=None, true_minimum=None,
            **kwargs):
        budget2TimesLosses = defaultdict(lambda: {"times": [], "losses": []})
        for (configId, budget), info in self.runId2info.items():
            end_time = info["end_time"]
            loss = info["loss"]
            budget2TimesLosses[budget]["times"].append(end_time)
            budget2TimesLosses[budget]["losses"].append(loss)
        budgets = self.budgets
        max_mins = -float("inf")

        budget2data = {}

        for i, (budget) in enumerate(budgets):
            TimesLosses = budget2TimesLosses[budget]
            times = TimesLosses["times"]
            idx = np.argsort(times)
            times = np.array(TimesLosses["times"])[idx]
            losses = np.array(TimesLosses["losses"])[idx]
            mins = [np.min(losses[:i]) for i in range(1, len(losses) + 1)]
            max_mins = max(max(mins), max_mins)
            budget2data[budget] = {
                "x": times,
                "y1": mins,
                "y2": losses,
            }
        for i, (budget) in enumerate(budgets):
            y2 = np.clip(budget2data[budget]["y2"], None, max_mins)
            color = None
            if colors:
                color = colors[i]
            name = f"budget={pbudget(budget)}"
            if names:
                name = names[i]
            ax = plot_convergence(budget2data[budget]["x"], budget2data[budget]["y1"], y2, xlabel, ylabel, ax, name,
                                  alpha, yscale, color,
                                  true_minimum, **kwargs)
        return ax

    def plot_concurrent_over_time(self, ax=None, num_points=512, alpha=0.5):
        data = []
        for info in self.runId2info.values():
            end_time = info["end_time"]
            start_time = info["start_time"]
            data.append([start_time, end_time])
        data = np.array(data)
        ts = np.linspace(data.min(), data.max(), num_points)
        n_workers = np.array([((data[:, 0] <= t) * (data[:, 1] > t)).sum() for t in ts])
        if ax is None:
            ax = plt.gca()
        ax.plot(ts, n_workers)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('number of concurrent runs')
        ax.set_title('Number of Concurrent Runs over Time')
        ax.grid(alpha=alpha)
        return ax

    def plot_finished_over_time(self, ax=None, alpha=0.5):
        budgets = self.budgets

        if ax is None:
            ax = plt.gca()
        times = {}
        for budget in budgets:
            times[budget] = [0]

        for (_, budget), info in self.runId2info.items():
            times[budget].append(info["end_time"])

        for budget in budgets:
            times[budget].sort()

        for budget in budgets:
            ax.plot(times[budget], np.arange(len(times[budget])), label=f'budget = {budget}')

        ax.set_xlabel('time [s]')
        ax.set_ylabel('number of finished runs')
        ax.set_title('Number of Finished Runs over Time')
        ax.legend()
        ax.grid(alpha=alpha)

        return ax

    def plot_correlation_across_budgets(self, ax=None):
        if ax is None:
            ax = plt.gca()
        configId2budgets = defaultdict(list)
        for (configId, budget) in self.runId2info.keys():
            configId2budgets[configId].append(budget)
        budgets = self.budgets

        loss_pairs = {}
        for b in budgets[:-1]:
            loss_pairs[b] = {}

        for b1, b2 in itertools.combinations(budgets, 2):
            loss_pairs[b1][b2] = []

        for configId, bs in configId2budgets.items():
            bs.sort()
            if len(bs) < 2: continue
            for b1, b2 in itertools.combinations(bs, 2):
                loss_pairs[float(b1)][float(b2)].append((
                    self.runId2info[(configId, b1)]["loss"],
                    self.runId2info[(configId, b2)]["loss"],
                ))

        rhos = np.eye(len(budgets) - 1)
        rhos.fill(np.nan)

        ps = np.eye(len(budgets) - 1)
        ps.fill(np.nan)

        for i in range(len(budgets) - 1):
            for j in range(i + 1, len(budgets)):
                correlation, pvalue = sps.spearmanr(loss_pairs[budgets[i]][budgets[j]])
                rhos[i][j - 1] = correlation
                ps[i][j - 1] = pvalue

        cax = ax.matshow(rhos, vmin=-1, vmax=1)
        plt.colorbar(cax)

        ax.set_yticks(range(len(budgets) - 1))
        ax.set_yticklabels([pbudget(b) for b in budgets[:-1]])

        ax.set_xticks(range(len(budgets) - 1))
        ax.set_xticklabels([pbudget(b) for b in budgets[1:]])

        ax.set_title('Rank correlation of the loss across the budgets')

        for i in range(len(budgets) - 1):
            for j in range(i + 1, len(budgets)):
                plt.text(j - 1, i, r'$\rho_{spearman} = %f$' + f"{rhos[i][j - 1]:.4f}\n"
                         + f'$p = {ps[i][j - 1]:.4f}$\n$n = {len(loss_pairs[budgets[i]][budgets[j]])}$',
                         horizontalalignment='center', verticalalignment='center')
        return ax
