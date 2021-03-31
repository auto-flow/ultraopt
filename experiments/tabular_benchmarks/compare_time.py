#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-02-14
# @Contact    : qichun.tang@bupt.edu.cn
import json

time_compare = json.load(open('time_compare.json'))
reduce = 0
for benchmark in time_compare:
    map = time_compare[benchmark]
    ETPE = map["UltraOpt-ETPE"]
    TPE = map["HyperOpt-TPE"]
    reduce += (TPE - ETPE) / TPE
reduce /= 4
print(reduce)
