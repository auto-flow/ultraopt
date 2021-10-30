import ConfigSpace as CS
from ultraopt.benchmarks import AbstractBenchmark



class Camelback(AbstractBenchmark):
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        y = (4 - 2.1 * (x[0] ** 2) + ((x[0] ** 4) / 3)) * \
            (x[0] ** 2) + x[0] * x[1] + (-4 + 4 * (x[1] ** 2)) * (x[1] ** 2)
        return {'function_value': y}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(
                Camelback.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Camelback',
                'num_function_evals': 200,
                'optima': ([[0.0898, -0.7126],
                            [-0.0898, 0.7126]]),
                'bounds': [[-3.0, 3.0],
                           [-2.0, 2.0]],
                'f_opt': -1.03162842}
