#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
import math


def float_gcd(a, b):
    def is_int(x):
        return not bool(int(x) - x)

    base = 1
    while not (is_int(a) and is_int(b)):
        a *= 10
        b *= 10
        base *= 10
    return math.gcd(int(a), int(b)) / base