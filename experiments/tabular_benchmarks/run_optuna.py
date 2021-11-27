#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-08
# @Contact    : qichun.tang@bupt.edu.cn
'''
--run_id=0 --benchmark=protein_structure --n_iters=100 --data_dir=/media/tqc/doc/Project/fcnet_tabular_benchmarks
'''
import argparse
import json
import os

from optuna.samplers import TPESampler
from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark, \
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--max_groups', default=0, type=int, nargs='?', help='max_groups')
parser.add_argument('--optimizer', default="ETPE", type=str, nargs='?', help='Which optimizer to use')
parser.add_argument('--mode', default="default", type=str, nargs='?', help='mode: {default, univar, univar_cat}')
parser.add_argument('--benchmark', default="protein_structure", type=str, nargs='?', help='specifies the benchmark')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="./", type=str, nargs='?', help='specifies the path to the tabular data')

args = parser.parse_args()
mode = args.mode
if args.benchmark == "protein_structure":
    b = FCNetProteinStructureBenchmark(data_dir=args.data_dir)
elif args.benchmark == "slice_localization":
    b = FCNetSliceLocalizationBenchmark(data_dir=args.data_dir)
elif args.benchmark == "naval_propulsion":
    b = FCNetNavalPropulsionBenchmark(data_dir=args.data_dir)
elif args.benchmark == "parkinsons_telemonitoring":
    b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=args.data_dir)
else:
    raise NotImplementedError

output_path = os.path.join(args.output_path, f"{args.benchmark}-optuna")


def objective_function(config: dict, budget: int = 100):
    loss, cost = b.objective_function(config, int(budget))
    return float(loss)


max_groups = args.max_groups
limit_max_groups = max_groups > 0

cs = b.get_configuration_space()

os.makedirs(os.path.join(output_path), exist_ok=True)

from optuna import Trial
from ConfigSpace import UniformFloatHyperparameter, OrdinalHyperparameter, CategoricalHyperparameter


class Evaluator():
    def __init__(self, config_space):
        self.config_space = config_space

    def __call__(self, trial: Trial):
        config = {}
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, OrdinalHyperparameter):
                config[hp.name] = trial.suggest_categorical(hp.name, hp.sequence)
            elif isinstance(hp, CategoricalHyperparameter):
                config[hp.name] = trial.suggest_categorical(hp.name, hp.choices)
        loss = b.objective_function(config, 100)
        return loss[0]


evaluator = Evaluator(config_space=cs)
import optuna

tpe = TPESampler(
    n_startup_trials=20, multivariate=True,
    seed=args.run_id)
study = optuna.create_study(sampler=tpe)
study.optimize(evaluator, n_trials=args.n_iters)
# dump(fmin_result, os.path.join(output_path, 'run_%d.pkl' % args.run_id))
res = b.get_results()
fh = open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w')
json.dump(res, fh)
fh.close()
