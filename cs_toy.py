#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-11-25
# @Contact    : qichun.tang@bupt.edu.cn
from collections import Counter
from pprint import pprint

from ConfigSpace import UniformFloatHyperparameter, ConfigurationSpace

cs = ConfigurationSpace()
u = UniformFloatHyperparameter('x', 1, 9, q=0.5, log=True)
cs.add_hyperparameter(u)
pprint(Counter([
    x.get_dictionary()['x']
    for x in cs.sample_configuration(1000)
]).most_common())
