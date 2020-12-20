#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-20
# @Contact    : tqichun@gmail.com
import unittest

from ultraopt.multi_fidelity import HyperBandIterGenerator


class TestHyperBandIterGenerator(unittest.TestCase):
    def test(self):
        hp_iter_gen = HyperBandIterGenerator(1 / 16, 1, 4, SH_only=True)
        res = hp_iter_gen.num_all_configs(3)
        assert res == 21 * 3
        res = hp_iter_gen.num_all_configs(5)
        assert res == 21 * 5

        hp_iter_gen = HyperBandIterGenerator(1 / 16, 1, 4)
        res = hp_iter_gen.num_all_configs(3)
        assert res == 29
        res = hp_iter_gen.num_all_configs(5)
        assert res == 55
