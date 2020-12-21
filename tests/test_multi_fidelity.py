#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-20
# @Contact    : tqichun@gmail.com
import unittest

from ultraopt.multi_fidelity import HyperBandIterGenerator, SuccessiveHalvingIterGenerator, CustomIterGenerator


class TestMultiFidelity(unittest.TestCase):
    def test_hyperband_iter_generator(self):
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

    def test_all_iter_generator(self):
        iter_gens = [
            HyperBandIterGenerator(1 / 16, 1, 4),
            SuccessiveHalvingIterGenerator(1 / 16, 1, 4),
            CustomIterGenerator([4, 2, 1], [1 / 4, 1 / 2, 1])
        ]
        for iter_gen in iter_gens:
            print(iter_gen)
            print()
            print("get_budgets", iter_gen.get_budgets())
            print("num_all_configs", iter_gen.num_all_configs(3))
            print("get_next_iteration", iter_gen.get_next_iteration(4))
