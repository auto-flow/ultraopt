#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-20
# @Contact    : qichun.tang@bupt.edu.cn
valid_optimizers = ["ETPE", "Forest", "GBRT", "Random"]
valid_parallel_strategies = ["Serial", "MapReduce", "AsyncComm"]
valid_warm_start_strategies = ["continue", "resume"]


class Configs:
    FractionalBudget = True
    AutoAdjustFractionalBudget = True
