#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-08
# @Contact    : qichun.tang@bupt.edu.cn
import argparse
import json
import os

import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers.bohb import BOHB
from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark, \
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark

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


class MyWorker(Worker):
    def compute(self, config, budget, **kwargs):
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

output_path = os.path.join(args.output_path, f"{args.benchmark}-hpbandster")
os.makedirs(os.path.join(output_path), exist_ok=True)

hb_run_id = str(args.run_id)

NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
ns_host, ns_port = NS.start()

worker = MyWorker(nameserver=ns_host, nameserver_port=ns_port,
                  run_id=hb_run_id,
                  id=0)
worker.run(background=True)
cs.seed(args.run_id)
bohb = BOHB(configspace=cs,
            run_id=hb_run_id,
            # just test KDE
            eta=2, min_budget=1, max_budget=1,
            nameserver=ns_host,
            nameserver_port=ns_port,
            num_samples=64,
            random_fraction=0.01,
            bandwidth_factor=1.2,
            ping_interval=10, min_bandwidth=.3)

results = bohb.run(args.n_iters, min_n_workers=1)

bohb.shutdown(shutdown_workers=True)
NS.shutdown()

print('fuck down')

res = b.get_results()

print('fucking result')

with open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w') as fh:
    json.dump(res, fh)
