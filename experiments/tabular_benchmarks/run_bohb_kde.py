#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-08
# @Contact    : qichun.tang@bupt.edu.cn
import argparse
import json
import os

import hpbandster.core.nameserver as hpns
from hpbandster.core.dispatcher import Job
from hpbandster.core.worker import Worker
from hpbandster.optimizers.config_generators.bohb import BOHB
from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark, \
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--mode', default="none", type=str, nargs='?', help='mode: {none, univar, univar_cat}')
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


def compute(config, budget):
    if args.benchmark == "protein_structure" \
            or args.benchmark == "slice_localization" or args.benchmark == "naval_propulsion" \
            or args.benchmark == "parkinsons_telemonitoring":

        original_cs = b.get_configuration_space()
        c = original_cs.sample_configuration()
        c["n_units_1"] = original_cs.get_hyperparameter("n_units_1").sequence[config["n_units_1"]]
        c["n_units_2"] = original_cs.get_hyperparameter("n_units_2").sequence[config["n_units_2"]]
        c["dropout_1"] = original_cs.get_hyperparameter("dropout_1").sequence[config["dropout_1"]]
        c["dropout_2"] = original_cs.get_hyperparameter("dropout_2").sequence[config["dropout_2"]]
        c["init_lr"] = original_cs.get_hyperparameter("init_lr").sequence[config["init_lr"]]
        c["batch_size"] = original_cs.get_hyperparameter("batch_size").sequence[config["batch_size"]]
        c["activation_fn_1"] = config["activation_fn_1"]
        c["activation_fn_2"] = config["activation_fn_2"]
        c["lr_schedule"] = config["lr_schedule"]
        y, cost = b.objective_function(c, budget=int(budget))

    else:
        y, cost = b.objective_function(config, budget=int(budget))

    return ({
        'loss': float(y),
        'info': float(cost)})


class MyWorker(Worker):
    def compute(self, config, budget, **kwargs):
        return compute(config, budget)


import ConfigSpace

cs = ConfigSpace.ConfigurationSpace()
cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("n_units_1", lower=0, upper=5))
cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("n_units_2", lower=0, upper=5))
cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("dropout_1", lower=0, upper=2))
cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("dropout_2", lower=0, upper=2))
cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_fn_1", ["tanh", "relu"]))
cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_fn_2", ["tanh", "relu"]))
cs.add_hyperparameter(
    ConfigSpace.UniformIntegerHyperparameter("init_lr", lower=0, upper=5))
cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("lr_schedule", ["cosine", "const"]))
cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("batch_size", lower=0, upper=3))

output_path = os.path.join(args.output_path, f"{args.benchmark}-bohb-kde")
os.makedirs(os.path.join(output_path), exist_ok=True)
cs.seed(args.run_id)

bohb = BOHB(configspace=cs, min_points_in_model=20, random_fraction=0, bandwidth_factor=1, num_samples=24)

losses = []
for i in tqdm(range(args.n_iters)):
    config = bohb.get_config(1)[0]
    loss = compute(config, 100)['loss']
    job = Job(id=i, budget=1, config=config)
    job.result = {'loss': loss}
    bohb.new_result(job)
    losses.append(loss)

res = b.get_results()


with open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w') as fh:
    json.dump(res, fh)
