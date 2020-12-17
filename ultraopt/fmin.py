#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-17
# @Contact    : tqichun@gmail.com
import importlib
import inspect
from typing import Callable, Union, Optional, List, Type
from uuid import uuid4

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration

from ultraopt.core.master import Master
from ultraopt.core.nameserver import NameServer
from ultraopt.core.worker import Worker
from ultraopt.hdl import HDL2CS
from ultraopt.optimizer.base_opt import BaseOptimizer
from ultraopt.utils import progress
from ultraopt.utils.config_space import get_dict_from_config
# 设计目标：单机并行、多保真优化
from ultraopt.utils.net import get_a_free_port


def fmin(
        eval_func: Callable,
        config_space: Union[ConfigurationSpace, dict],
        optimizer: Union[BaseOptimizer, str, Type] = "TPE",
        initial_points: Union[None, List[Configuration], List[dict]] = None,
        random_state=42,
        n_iterations=100,
        n_jobs=1,
        budgets: Optional[List[float]] = None,
        stage_configs: Optional[List[float]] = None,
        multi_fidelity: Optional[str] = None,
        previous_budget2obvs=None,
        run_id=None,
        ns_host="127.0.0.1",
        ns_port=9090
):
    # ------------   config_space   ---------------#
    if isinstance(config_space, dict):
        cs_ = HDL2CS()(config_space)
    elif isinstance(config_space, ConfigurationSpace):
        cs_ = config_space
    else:
        raise NotImplementedError
    # ------------      budgets     ---------------#
    if budgets is None:
        budgets_ = [1]
    else:
        budgets_ = list(budgets)
    # ------------ optimizer ---------------#
    if inspect.isclass(optimizer):
        if not issubclass(optimizer, BaseOptimizer):
            raise ValueError(f"optimizer {optimizer} is not subclass of BaseOptimizer")
        opt_ = optimizer()
    elif isinstance(optimizer, BaseOptimizer):
        opt_ = optimizer
    elif isinstance(optimizer, str):
        try:
            opt_ = getattr(importlib.import_module("ultraopt.optimizer"),
                           f"{optimizer}Optimizer")()
        except Exception:
            raise ValueError(f"Invalid optimizer string-indicator: {optimizer}")
    else:
        raise NotImplementedError
    # fixme: 返回值的设置
    # x(config): 最优点
    # best_loss: 最小的loss
    # budget2obvs: 所有的观测结果
    warm_start_optimizer(optimizer, previous_budget2obvs)
    progress_callback = progress.default_callback
    # non-parallelism debug mode
    if n_jobs == 1 and (not multi_fidelity):
        budgets_ = [1]
        opt_.initialize(cs_, budgets_, random_state, initial_points)
        with progress_callback(
                initial=0, total=n_iterations
        ) as progress_ctx:
            for iter in range(n_iterations):
                config, _ = opt_.ask()
                loss = eval_func(config)
                opt_.tell(config, loss)
                _, best_loss, _ = get_wanted(opt_)
                progress_ctx.postfix = f"best loss: {best_loss:.3f}"
                progress_ctx.update(1)
    else:
        # start name-server
        if run_id is None:
            run_id = uuid4().hex
        NS = NameServer(run_id=run_id, host=ns_host, port=get_a_free_port(ns_port, ns_host))
        NS.start()
        # start n workers
        workers = [Worker(run_id=run_id, nameserver=ns_host, nameserver_port=ns_port, host=ns_host, worker_id=i)
                   for i in range(n_jobs)]
        for worker in workers:
            worker.initialize(eval_func)
            worker.run(True, "process")
        # start master
        master = Master(run_id=run_id, optimizer=opt_, nameserver=ns_host, nameserver_port=ns_port, host=ns_host)

        result = master.run(n_iterations)

    max_budget, best_loss, best_config = get_wanted(opt_)
    return {
        "best_config": best_config,
        "best_loss": best_loss,
        "budget2obvs": opt_.budget2obvs,
        "optimizer": opt_
    }


def warm_start_optimizer(optimizer: BaseOptimizer, budget2obvs):
    if budget2obvs is None:
        return
    for budget, obvs in budget2obvs.items():
        L = len(obvs["losses"])
        for i in range(L):
            optimizer.tell(
                get_dict_from_config(obvs["configs"][i]),
                obvs["losses"][i],
                budget,
                update_model=(i == L - 1))


def get_wanted(opt_: BaseOptimizer):
    budget2obvs = opt_.budget2obvs
    max_budget = opt_.get_available_max_budget()
    obvs = budget2obvs[max_budget]
    losses = obvs["losses"]
    configs = obvs["configs"]
    idx = np.argmin(losses)
    best_loss = losses[idx]
    best_config = get_dict_from_config(configs[idx])
    return max_budget, best_loss, best_config
