import numpy as np

import ConfigSpace as CS
from ultraopt.benchmarks import AbstractBenchmark



class Levy1D(AbstractBenchmark):
    """
    N-Dimensional Function, taken from here
        https://www.sfu.ca/~ssurjano/levy.html
    """

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        w0 = (1 + (x[0] - 1) / 4)
        term1 = np.power(np.sin(np.pi * w0), 2)

        term2 = 0
        for i in range(len(x) - 1):
            wi = 1 + (x[i] - 1) / 4
            term2 += np.power(wi - 1, 2) * (1 + 10 * np.power(np.sin(wi * np.pi + 1), 2))

        wd = (1 + (x[-1] - 1) / 4)
        term3 = np.power(wd - 1, 2)
        term3 *= (1 + np.power(np.sin(2 * np.pi * wd), 2))

        y = term1 + term2 + term3
        return {'function_value': y}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Levy1D.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Levy1D',
                'num_function_evals': 100,
                'optima': ([[1.0]]),
                'bounds': [[-15.0, 10.0]],
                'f_opt': 0.0}


dimensions = list(range(2, 51))
for d in dimensions:
    benchmark_string = """class Levy%dD(Levy1D):
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Levy%dD',
                'num_function_evals': 200,
                'optima': ([[1.0]*%d]),
                'bounds': [[-15.0, 10.0]]*%d,
                'f_opt': 0.0}
    
    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Levy%dD.get_meta_information()['bounds'])
        return cs""" % (d, d, d, d, d)


    exec(benchmark_string)