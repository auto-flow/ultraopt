#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : tqichun@gmail.com
from oxygen.config_generators.base_cg import BaseConfigGenerator


class SamplingSortConfigGenerator(BaseConfigGenerator):
    def __init__(
            self, config_space, budgets, random_state=42, initial_points=None, budget2obvs=None,

    ):
        super(SamplingSortConfigGenerator, self).__init__(config_space, budgets, random_state, initial_points, budget2obvs)


