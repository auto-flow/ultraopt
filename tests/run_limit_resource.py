#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-03-31
# @Contact    : qichun.tang@bupt.edu.cn
from ultraopt.tests.automl import HDL, evaluator
from ultraopt import fmin
import warnings
warnings.filterwarnings("ignore")

result = fmin(evaluator, HDL, n_jobs=3, limit_resource=True, verbose=1,
              time_limit=1,memory_limit=0)
print(result)
