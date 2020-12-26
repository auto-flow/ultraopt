#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-18
# @Contact    : qichun.tang@bupt.edu.cn
from ultraopt.multi_fidelity.iter.base_iter import BaseIteration
from ultraopt.structure import Job


class WarmStartIteration(BaseIteration):
    """
    iteration that imports a privious Result for warm starting
    """

    def __init__(self, result, optimizer):

        self.is_finished = False
        self.stage = 0

        id2conf = result.get_id2config_mapping()
        delta_t = - max(map(lambda r: r.timestamps['finished'], result.get_all_runs()))

        super().__init__(-1, [len(id2conf)], [None], None)

        for i, id in enumerate(id2conf):
            new_id = self.add_configuration(config=id2conf[id]['config'], config_info=id2conf[id]['config_info'])

            for r in result.get_runs_by_id(id):

                j = Job(new_id, config=id2conf[id]['config'], budget=r.budget, config_info=id2conf[id]['config_info'])

                j.result = {'loss': r.loss, 'info': r.info}
                j.error_logs = r.error_logs

                for k, v in r.timestamps.items():
                    j.timestamps[k] = v + delta_t

                self.register_result(j, skip_sanity_checks=True)
                should_update = (i == len(id2conf) - 1)
                optimizer.new_result(j, update_model=should_update, should_update_weight=-1)
        optimizer.update_weight(should_update=1)

        # mark as finished, as no more runs should be executed from these runs
        self.is_finished = True

    def fix_timestamps(self, time_ref):
        """
            manipulates internal time stamps such that the last run ends at time 0
        """

        for k, v in self.data.items():
            for kk, vv in v.timestamps.items():
                for kkk, vvv in vv.items():
                    self.data[k].timestamps[kk][kkk] += time_ref