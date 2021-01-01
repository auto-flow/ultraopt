#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import ConfigSpace as CS

from ultraopt.benchmarks import AbstractBenchmark


class Rosenbrock2D(AbstractBenchmark):
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        y = 0
        d = len(x)
        for i in range(d - 1):
            y += 100 * (x[i + 1] - x[i] ** 2) ** 2
            y += (x[i] - 1) ** 2

        return {'function_value': y}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Rosenbrock2D.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Rosenbrock2D',
                'num_function_evals': 200,
                'optima': ([[1.0] * 2]),
                'bounds': [[-5.0, 10.0]] * 2,
                'f_opt': 0.0}


# Build more Rosenbrocks
dimensions = [5, 10, 20]
for d in dimensions:
    benchmark_string = """
class Rosenbrock%dD(Rosenbrock2D):

    @staticmethod
    def get_meta_information():
        return {'name': 'Rosenbrock%dD',
                'num_function_evals': 200,
                'optima': ([[1.0]*%d]),
                'bounds': [[-5.0, 10.0]]*%d,
                'f_opt': 0.0}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Rosenbrock%dD.get_meta_information()['bounds'])
        return cs""" % (d, d, d, d, d)
    exec(benchmark_string)


class MultiFidelityRosenbrock2D(Rosenbrock2D):

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, budget=100, **kwargs):
        # shift = (10 + 2 * np.log(budget / 100.)) / 10.
        # f_bias = np.log(100) - np.log(budget)
        f_bias = 0
        shift = 2 - 2 * (budget / 100)
        y = 0
        d = len(x)
        for i in range(d - 1):
            zi = x[i] - shift
            zi_next = x[i + 1] - shift

            y += 100 * (zi_next - zi ** 2) ** 2
            y += (zi - 1) ** 2
            y += f_bias

        return {'function_value': y}

    @staticmethod
    def get_meta_information():
        return {'name': 'MultiFidelityRosenbrock2D',
                'num_function_evals': 200,
                'optima': ([[1.0] * 2]),
                'bounds': [[-5.0, 10.0]] * 2,
                'f_opt': 0.0}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x, budget=100)


# Build more MultiFidelityRosenbrocks
dimensions = [5, 10, 20]
for d in dimensions:
    benchmark_string = """
class MultiFidelityRosenbrock%dD(MultiFidelityRosenbrock2D):

    @staticmethod
    def get_meta_information():
        return {'name': 'MultiFidelityRosenbrock%dD',
                'num_function_evals': 200,
                'optima': ([[1.0]*%d]),
                'bounds': [[-5.0, 10.0]]*%d,
                'f_opt': 0.0}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(MultiFidelityRosenbrock%dD.get_meta_information()['bounds'])
        return cs""" % (d, d, d, d, d)
    exec(benchmark_string)
