#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-27
# @Contact    : qichun.tang@bupt.edu.cn
from ultraopt import fmin
from ultraopt.multi_fidelity import HyperBandIterGenerator
from ultraopt.tests.mock import evaluate, config_space
import pylab as plt

res = fmin(evaluate, config_space, n_jobs=3,
           multi_fidelity_iter_generator=HyperBandIterGenerator(25, 100, 2))
res.plot_convergence_over_time(yscale="log")
plt.show()
res.plot_concurrent_over_time(num_points=100)
plt.show()
res.plot_finished_over_time()
plt.show()
print()
