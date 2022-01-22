#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-11-28
# @Contact    : qichun.tang@bupt.edu.cn
import os

import numpy as np
from ultraopt import fmin
from ultraopt.optimizer import ETPEOptimizer

os.environ['ORD_EMB_REG'] = 'cos'
np.random.seed(0)
HDL = {
    'x1': {
        '_type': 'ordinal',
        '_value': ['a', 'b', 'c'],
    },
    'x2': {
        '_type': 'ordinal',
        '_value': ['a', 'b', 'c', 'd'],
    },
    'x3': {
        '_type': 'ordinal',
        '_value': ['a', 'b', 'c', 'd', 'e', 'f'],
    },
    'x4': {
        '_type': 'choice',
        '_value': ['a', 'b', 'c', 'd', 'e'],
    },
    'x5': {
        '_type': 'choice',
        '_value': ['a', 'b', 'c', 'd'],
    },
    'x6': {
        '_type': 'uniform',
        '_value': [0, 1],
    },
    'x7': {
        '_type': 'choice',
        '_value': ['a','b'],
    },
    'x8': {
        '_type': 'ordinal',
        '_value': ['a','b'],
    },
}
opt = ETPEOptimizer(
    consider_ord_as_cont=True, scale_cont_var=False,
    # pretrained_emb='test_emb_table.txt'
)
ret = fmin(lambda x: np.random.rand(), HDL, opt)
losses=ret.budget2obvs[1]['losses']
configs=ret.budget2obvs[1]['configs']

ret.export_embedding_table('test_emb_table.txt')
