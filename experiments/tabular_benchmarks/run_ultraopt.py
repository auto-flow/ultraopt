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
import time

import numpy as np
from joblib import dump
from joblib import load
from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark, \
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from ultraopt import fmin
from ultraopt.multi_fidelity import HyperBandIterGenerator
from ultraopt.tpe.meta import MetaLearning

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

output_path = os.path.join(args.output_path, f"{args.benchmark}-ultraopt_{args.optimizer}")


def objective_function(config: dict, budget: int = 100):
    loss, cost = b.objective_function(config, int(budget))
    return float(loss)


max_groups = args.max_groups
limit_max_groups = max_groups > 0

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
from ultraopt.optimizer import ETPEOptimizer, ForestOptimizer, RandomOptimizer

if mode != "default":
    output_path += f"_{mode}"

if limit_max_groups:
    output_path += f"_g{max_groups}"

from ultraopt.tpe import SampleDisign

min_points_in_model = int(os.getenv('min_points_in_model', '20'))
et_file = f'{args.benchmark}_ok.txt'
pretrain = True
# pretrain=False
# scale = False
init_pts = int(os.getenv('INIT_PTS', '0'))
init_bench = os.getenv('INIT_BENCH')
print(f'init_pts={init_pts}')
print(f'init_bench={init_bench}')
initial_points = []
if init_pts:
    data = load(f'store/{init_bench}/{args.run_id}.pkl')
    initial_points = [x[1] for x in data[:init_pts]]
    output_path += f'_init_pts({init_bench},{init_pts})'

meta_learning = None
meta_weight = float(os.getenv('META_WEIGHT', '0'))
meta_config = os.getenv('META_CONFIG', '20:80')
if meta_weight:
    n_good, n_bad = meta_config.split(':')
    n_good, n_bad = int(n_good), int(n_bad)
    data = load(f'store/{init_bench}/{args.run_id % 10}.pkl')
    # data = load(f'store/{init_bench}/10000.pkl')
    configs = [x[1] for x in data]
    # good_configs = load('store/good_configs.pkl')
    good_configs = configs[:n_good]
    bad_configs = configs[-n_bad:]
    meta_learning = MetaLearning(
        good_configs, bad_configs,
        # random_startups=5,
        bw_factor=3,
        specific_sample_design=[
            # SampleDisign(ratio=1, is_random=True),
            SampleDisign(n_samples=30, is_random=True),
        ],
        weight=meta_weight)
    output_path += f'_meta_learn({init_bench},{meta_weight},{n_good},{n_bad})'

print(f'pretrain = {pretrain}')
if optimizer == "ETPE":
    if mode == 'univar':
        optimizer = ETPEOptimizer(
            multivariate=False,
            specific_sample_design=[
                SampleDisign(ratio=0.5, is_random=True)
            ]
        )
    elif mode == 'univar_cat':
        optimizer = ETPEOptimizer(
            min_points_in_model=min_points_in_model,
            multivariate=False,
            embed_catVar=False,
            specific_sample_design=[
                SampleDisign(ratio=0.5, is_random=True)
            ]
        )
    elif mode == "default":
        from tabular_nn import EmbeddingEncoder

        encoder = EmbeddingEncoder(max_epoch=10, n_jobs=1, verbose=1)
        optimizer = ETPEOptimizer(
            # limit_max_groups=limit_max_groups,
            min_points_in_model=min_points_in_model,
            limit_max_groups=max_groups > 0,
            max_groups=max_groups,
            # pretrained_emb=et_file if pretrain else None,
            scale_cont_var=False,
            consider_ord_as_cont=True,
            meta_learning=meta_learning
            # category_encoder=encoder,
        )
    else:
        raise NotImplementedError
elif optimizer == "Forest":
    if mode == "default":
        optimizer = ForestOptimizer(min_points_in_model=min_points_in_model)
    elif mode == "local_search":
        optimizer = ForestOptimizer(min_points_in_model=min_points_in_model, use_local_search=True)
elif optimizer == "Random":
    optimizer = RandomOptimizer()
ret = fmin(
    objective_function, cs, optimizer,
    n_iterations=args.n_iters, random_state=args.run_id,
    multi_fidelity_iter_generator=iter_generator,
    initial_points=initial_points
)
os.makedirs(os.path.join(output_path), exist_ok=True)

print(ret)
if os.getenv('DEBUG') == 'true':
    store_root = f'store/{args.benchmark}'
    os.system(f'mkdir -p {store_root}')
    store_file = f'{store_root}/{args.run_id}.pkl'

    losses = ret.budget2obvs[1]['losses']
    configs = ret.budget2obvs[1]['configs']
    argsort = np.argsort(losses)
    configs = [configs[i] for i in argsort]
    # losses=losses[:500]+losses[-500:]
    # configs=configs[:500]+configs[-500:]
    data = list(zip(sorted(losses), configs))
    dump(data, store_file)
    exit(0)
# dump(fmin_result, os.path.join(output_path, 'run_%d.pkl' % args.run_id))
res = b.get_results()
fh = open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w')
json.dump(res, fh)
fh.close()
if HB:
    time.sleep(5)
