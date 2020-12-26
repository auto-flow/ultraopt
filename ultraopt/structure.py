#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
import time


class Datum(object):
    def __init__(self, config, config_info, results=None, timestamps=None, exceptions=None, status='QUEUED', budget=0):
        self.config = config
        self.config_info = config_info
        self.results = results if not results is None else {}
        self.timestamps = timestamps if not timestamps is None else {}
        self.exceptions = exceptions if not exceptions is None else {}
        self.status = status
        self.budget = budget

    def __repr__(self):
        return ( \
                    "\nconfig:{}\n".format(self.config) + \
                    "config_info:\n{}\n".format(self.config_info) + \
                    "losses:\n"
                    '\t'.join(["{}: {}\t".format(k, v['loss']) for k, v in self.results.items()]) + \
                    "time stamps: {}".format(self.timestamps)
        )


class Job(object):
    def __init__(self, id, **kwargs):
        self.id = id

        self.kwargs = kwargs

        self.timestamps = {}

        self.result = None
        self.exception = None
        self.worker_name = None

    def time_it(self, which_time):
        self.timestamps[which_time] = time.time()

    def __repr__(self):
        return(\
            "job_id: " +str(self.id) + "\n" + \
            "kwargs: " + str(self.kwargs) + "\n" + \
            "result: " + str(self.result)+ "\n" +\
            "exception: "+ str(self.exception) + "\n"
        )