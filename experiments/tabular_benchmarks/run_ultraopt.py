#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-08
# @Contact    : qichun.tang@bupt.edu.cn
import argparse
import json
import os
import time

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark, \
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from ultraopt import fmin
from ultraopt.multi_fidelity import HyperBandIterGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--optimizer', default="ETPE", type=str, nargs='?', help='Which optimizer to use')
parser.add_argument('--mode', default="none", type=str, nargs='?', help='mode: {none, univar, univar_cat}')
parser.add_argument('--benchmark', default="protein_structure", type=str, nargs='?', help='specifies the benchmark')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
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

output_path = os.path.join(args.output_path, f"{args.benchmark}-ultraopt_{args.optimizer}")


def objective_function(config: dict, budget: int = 100):
    loss, cost = b.objective_function(config, int(budget))
    return float(loss)


cs = b.get_configuration_space()
HB = False
if args.optimizer == "BOHB":
    optimizer = "ETPE"
    iter_generator = HyperBandIterGenerator(min_budget=3, max_budget=100, eta=3)
    HB = True
elif args.optimizer == "HyperBand":
    optimizer = "Random"
    iter_generator = HyperBandIterGenerator(min_budget=3, max_budget=100, eta=3)
    HB = True
else:
    optimizer = args.optimizer
    iter_generator = None
from ultraopt.optimizer import ETPEOptimizer

if mode != "default":
    output_path += f"_{mode}"

os.makedirs(os.path.join(output_path), exist_ok=True)


if optimizer == "ETPE":
    if mode == 'univar':
        optimizer = ETPEOptimizer(multivariate=False)
    elif mode == 'univar_cat':
        optimizer = ETPEOptimizer(multivariate=False, embed_cat_var=False)
    elif mode == "default":
        optimizer = ETPEOptimizer()
    else:
        raise NotImplementedError

fmin_result = fmin(
    objective_function, cs, optimizer,
    n_iterations=args.n_iters, random_state=args.run_id,
    multi_fidelity_iter_generator=iter_generator)
print(fmin_result)
# dump(fmin_result, os.path.join(output_path, 'run_%d.pkl' % args.run_id))
res = b.get_results()
fh = open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w')
json.dump(res, fh)
fh.close()
if HB:
    time.sleep(5)
