#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-18
# @Contact    : tqichun@gmail.com
from typing import Type

from ultraopt.multi_fidelity.iter import BaseIteration, RankReductionIteration


class BaseIterGenerator():
    def __init__(
            self,
            iter_klass: type = None
    ):
        if iter_klass is None:
            iter_klass = RankReductionIteration
        assert issubclass(iter_klass, BaseIteration)
        self.iter_klass: Type[BaseIteration] = iter_klass

    def get_next_iteration(self, iteration, **kwargs):
        raise NotImplementedError
