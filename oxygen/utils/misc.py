#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : tqichun@gmail.com
from fractions import Fraction


def pprint_budget(budget: float):
    if budget - float(int(budget)) == 0:
        return str(int(budget))
    fraction = Fraction.from_float(budget)
    return f"{fraction.numerator}/{fraction.denominator}"