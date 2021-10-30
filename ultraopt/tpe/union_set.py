#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-29
# @Contact    : qichun.tang@bupt.edu.cn

class UnionSet():
    '''
    origin implementation: https://github.com/TQCAI/Algorithm/blob/master/notes/%E7%AE%97%E6%B3%95%E4%B8%8E%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84/%E6%A0%91%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84/%E5%B9%B6%E6%9F%A5%E9%9B%86/001.%20%E5%B9%B6%E6%9F%A5%E9%9B%86%E5%AE%9A%E4%B9%89.md
    '''

    def __init__(self, n):
        self.cnt = n
        self.parent = [0] * n
        for i in range(n):
            self.parent[i] = i

    def union(self, a, b):
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return
        self.parent[pa] = pb
        self.cnt -= 1

    def find(self, x) -> int:
        if x == self.parent[x]:
            return x
        # 找到根节点
        r = x
        while r != self.parent[r]:
            r = self.parent[r]
        # 路径压缩
        while x != self.parent[x]:
            t = self.parent[x]
            self.parent[x] = r
            x = t
        return r
