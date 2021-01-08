#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-08
# @Contact    : qichun.tang@bupt.edu.cn
import argparse
import json
import os

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark, \
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark

from ultraopt import fmin

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--optimizer', default="ETPE", type=str, nargs='?', help='Which optimizer to use')
parser.add_argument('--benchmark', default="protein_structure", type=str, nargs='?', help='specifies the benchmark')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="./", type=str, nargs='?', help='specifies the path to the tabular data')

args = parser.parse_args()

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
os.makedirs(os.path.join(output_path), exist_ok=True)


def objective_function(config: dict, budget: int = 100):
    loss, cost = b.objective_function(config, budget)
    return loss


cs = b.get_configuration_space()
fmin(objective_function, cs, args.optimizer, n_iterations=args.n_iters, random_state=args.run_id)
res = b.get_results()
fh = open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w')
json.dump(res, fh)
fh.close()
