#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-17
# @Contact    : qichun.tang@bupt.edu.cn
import importlib
import inspect
import logging
from typing import Callable, Union, Optional, List, Type
from uuid import uuid4

from ConfigSpace import ConfigurationSpace, Configuration
from joblib import Parallel, delayed

from ultraopt.async_comm.master import Master
from ultraopt.async_comm.nameserver import NameServer
from ultraopt.async_comm.worker import Worker
from ultraopt.facade.result import FMinResult
from ultraopt.facade.utils import warm_start_optimizer, get_wanted
from ultraopt.hdl import hdl2cs
from ultraopt.multi_fidelity import BaseIterGenerator, CustomIterGenerator
from ultraopt.optimizer.base_opt import BaseOptimizer
from ultraopt.utils import progress
from ultraopt.utils.misc import dump_checkpoint


def fmin(
        eval_func: Callable,
        config_space: Union[ConfigurationSpace, dict],
        optimizer: Union[BaseOptimizer, str, Type] = "ETPE",
        initial_points: Union[None, List[Configuration], List[dict]] = None,
        random_state=42,
        n_iterations=100,
        n_jobs=1,
        parallel_strategy="AsyncComm",
        auto_identify_serial_strategy=True,
        multi_fidelity_iter_generator: Optional[BaseIterGenerator] = None,
        previous_result: Union[FMinResult, BaseOptimizer, str, None] = None,
        warm_start_strategy="continue",
        show_progressbar=True,
        checkpoint_file=None,
        checkpoint_freq=10,
        verbose=0,
        run_id=None,
        ns_host="127.0.0.1",
        ns_port=0
):
    # fixme: 这合理吗
    if verbose <= 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)
    # 设计目标：单机并行、多保真优化
    # ------------   config_space   ---------------#
    if isinstance(config_space, dict):
        cs_ = hdl2cs(config_space)
    elif isinstance(config_space, ConfigurationSpace):
        cs_ = config_space
    else:
        raise NotImplementedError
    # ------------      budgets     ---------------#
    if multi_fidelity_iter_generator is None:
        budgets_ = [1]
    else:
        budgets_ = multi_fidelity_iter_generator.get_budgets()
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
    if show_progressbar:
        progress_callback = progress.default_callback
    else:
        progress_callback = progress.no_progress_callback
    # 3种运行模式：
    # 1. 串行，方便调试，不支持multi-fidelity
    # 2. AsyncComm，RPC，支持multi-fidelity
    # 3. MapReduce，不支持multi-fidelity
    # non-parallelism debug mode
    if auto_identify_serial_strategy and n_jobs == 1 and multi_fidelity_iter_generator is None:
        parallel_strategy = "Serial"
    if parallel_strategy in ["Serial", "MapReduce"]:
        budgets_ = [1]
    # initialize optimizer
    opt_.initialize(cs_, budgets_, random_state, initial_points)
    opt_ = warm_start_optimizer(opt_, previous_result, warm_start_strategy)
    if parallel_strategy == "Serial":
        with progress_callback(
                initial=0, total=n_iterations
        ) as progress_ctx:
            for counts in range(n_iterations):
                config, _ = opt_.ask()
                loss = eval_func(config)
                opt_.tell(config, loss)
                _, best_loss, _ = get_wanted(opt_)
                progress_ctx.postfix = f"best loss: {best_loss:.3f}"
                progress_ctx.update(1)
                if checkpoint_file is not None:
                    if (counts % checkpoint_freq == 0 and counts != 0) \
                            or (counts == n_iterations - 1):
                        dump_checkpoint(opt_, checkpoint_file)
    elif parallel_strategy == "AsyncComm":
        # start name-server
        if run_id is None:
            run_id = uuid4().hex
        if multi_fidelity_iter_generator is None:
            # todo: warning
            multi_fidelity_iter_generator = CustomIterGenerator([1], [1])
        NS = NameServer(run_id=run_id, host=ns_host, port=ns_port)  # get_a_free_port(ns_port, ns_host)
        _, ns_port = NS.start()
        # start n workers
        workers = [Worker(run_id=run_id, nameserver=ns_host, nameserver_port=ns_port,
                          host=ns_host, worker_id=i)
                   for i in range(n_jobs)]
        for worker in workers:
            worker.initialize(eval_func)
            worker.run(True, "thread")
        # start master
        master = Master(
            run_id, opt_, multi_fidelity_iter_generator, progress_callback=progress_callback,
            checkpoint_file=checkpoint_file, checkpoint_freq=checkpoint_freq,
            nameserver=ns_host, nameserver_port=ns_port, host=ns_host)
        result = master.run(n_iterations)
        master.shutdown(True)
        NS.shutdown()
        # todo: 将result添加到返回结果中
    elif parallel_strategy == "MapReduce":
        # todo: 支持multi-fidelity
        counts = 0
        with progress_callback(
                initial=0, total=n_iterations
        ) as progress_ctx:
            while counts < n_iterations:
                n_parallels = min(n_jobs, n_iterations - counts)
                config_info_pairs = opt_.ask(n_points=n_parallels)
                losses = Parallel(n_jobs=n_parallels)(
                    delayed(eval_func)(config)
                    for config, _ in config_info_pairs
                )
                for j, (loss, (config, _)) in enumerate(zip(losses, config_info_pairs)):
                    opt_.tell(config, loss, update_model=(j == n_parallels - 1))
                counts += n_parallels
                _, best_loss, _ = get_wanted(opt_)
                progress_ctx.postfix = f"best loss: {best_loss:.3f}"
                progress_ctx.update(n_parallels)
                iteration = counts // n_jobs
                if checkpoint_file is not None:
                    if ((iteration - 1) % checkpoint_freq == 0) \
                            or (counts == n_iterations):
                        dump_checkpoint(opt_, checkpoint_file)
    else:
        raise NotImplementedError

    # max_budget, best_loss, best_config = get_wanted(opt_)
    return FMinResult(opt_)
