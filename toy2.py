#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-16
# @Contact    : tqichun@gmail.com
import pylab as plt

start = 3
history = []
L = 100
for i in range(L):
    history.append(start)
    start *= 0.9
plt.plot(range(L), history)
plt.show()
