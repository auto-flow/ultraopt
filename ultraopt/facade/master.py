#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-18
# @Contact    : tqichun@gmail.com
from typing import Dict, Optional, List

import numpy as np
from frozendict import frozendict

from ultraopt.core.master import Master
from ultraopt.multi_fidelity import SuccessiveHalvingIteration


class MultiFidelityMaster(Master):
    def __init__(
            self,
            run_id,
            optimizer,
            budgets: Optional[List[float]] = None,
            stage_configs: Optional[List[float]] = None,
            multi_fidelity: Optional[str] = None,
            working_directory='.',
            ping_interval=60,
            time_left_for_this_task=np.inf,
            nameserver='127.0.0.1',
            nameserver_port=None,
            host=None,
            shutdown_workers=True,
            job_queue_sizes=(-1, 0),
            dynamic_queue_size=True,
            result_logger=None,
            previous_result=None,
            incumbents: Dict[float, dict] = None,
            incumbent_performances: Dict[float, float] = None
    ):
        super(MultiFidelityMaster, self).__init__(
            run_id, optimizer, working_directory, ping_interval, time_left_for_this_task,
            nameserver, nameserver_port, host,
            shutdown_workers, job_queue_sizes, dynamic_queue_size,
            result_logger, previous_result, incumbents,
            incumbent_performances)
        self.multi_fidelity = multi_fidelity
        self.stage_configs = stage_configs
        self.budgets = budgets
        assert len(self.budgets)==len(self.stage_configs), ValueError("length of budgets and state configs should be equal.")

    def get_next_iteration(self, iteration, iteration_kwargs=frozendict()):
        if self.multi_fidelity=="SH":
            pass
        return SuccessiveHalvingIteration(
            HPB_iter=iteration,
            num_configs=ns,
            budgets=self.budgets[(-s - 1):],
            config_sampler=self.optimizer.get_config,
            **iteration_kwargs
        )


