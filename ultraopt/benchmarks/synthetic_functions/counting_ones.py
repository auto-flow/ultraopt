import numpy as np

import ConfigSpace as CS
from ultraopt.benchmarks import AbstractBenchmark



class CountingOnes(AbstractBenchmark):

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, budget=100, **kwargs):

        y = 0
        for h in config:
            if 'float' in h:
                samples = np.random.binomial(1, config[h], int(budget))
                y += np.mean(samples)
            else:
                y += config[h]

        return {'function_value': -y}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x)


    @AbstractBenchmark._check_configuration
    def objective_function_test(self, config, **kwargs):
        return {'function_value': -np.sum(config.get_array())}

    @staticmethod
    def get_configuration_space(n_categorical=1, n_continuous=1):
        cs = CS.ConfigurationSpace()
        for i in range(n_categorical):
            cs.add_hyperparameter(CS.CategoricalHyperparameter("cat_%d" % i, [0, 1]))
        for i in range(n_continuous):
            cs.add_hyperparameter(CS.UniformFloatHyperparameter('float_%d' % i, lower=0, upper=1))
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Counting Ones'}



if __name__ == "__main__":
    cs = CountingOnes.get_configuration_space(8,8)
    config = cs.sample_configuration()
    B = CountingOnes()
    print(config)
    for i in range(10):
        print(B.objective_function(config, budget=5))
    print(B.objective_function_test(config))
