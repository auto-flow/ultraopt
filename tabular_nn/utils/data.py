#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np
import multiprocessing as mp


def check_n_jobs(n_jobs):
    cpu_count = mp.cpu_count()
    if n_jobs == 0:
        return 1
    elif n_jobs > 0:
        return min(cpu_count, n_jobs)
    else:
        return max(1, cpu_count + 1 + n_jobs)

def pairwise_distance(X, Y):
    A, M = X.shape
    B, _ = Y.shape
    matrix1 = X.reshape([A, 1, M])
    matrix1 = np.repeat(matrix1, B, axis=1)
    matrix2 = Y.reshape([1, B, M])
    matrix2 = np.repeat(matrix2, A, axis=0)
    distance_sqr = np.sum((matrix1 - matrix2) ** 2, axis=-1)  # A, B
    # didnt sqrt
    return distance_sqr
