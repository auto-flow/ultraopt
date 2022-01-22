import argparse
import json
import os
from copy import deepcopy

import ConfigSpace
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark, \
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--benchmark', default="protein_structure", type=str, nargs='?', help='specifies the benchmark')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="./", type=str, nargs='?', help='specifies the path to the tabular data')

args = parser.parse_args()

if args.benchmark == "nas_cifar10a":
    b = NASCifar10A(data_dir=args.data_dir)

elif args.benchmark == "nas_cifar10b":
    b = NASCifar10B(data_dir=args.data_dir)

elif args.benchmark == "nas_cifar10c":
    b = NASCifar10C(data_dir=args.data_dir)

elif args.benchmark == "protein_structure":
    b = FCNetProteinStructureBenchmark(data_dir=args.data_dir)

elif args.benchmark == "slice_localization":
    b = FCNetSliceLocalizationBenchmark(data_dir=args.data_dir)

elif args.benchmark == "naval_propulsion":
    b = FCNetNavalPropulsionBenchmark(data_dir=args.data_dir)

elif args.benchmark == "parkinsons_telemonitoring":
    b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=args.data_dir)

output_path = os.path.join(args.output_path, f"{args.benchmark}-tpe")

cs = b.get_configuration_space()

space = {}
for h in cs.get_hyperparameters():
    if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
        space[h.name] = hp.quniform(h.name, 0, len(h.sequence) - 1, q=1)
    elif type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
        space[h.name] = hp.choice(h.name, h.choices)
    elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
        space[h.name] = hp.quniform(h.name, h.lower, h.upper, q=1)
    elif type(h) == ConfigSpace.hyperparameters.UniformFloatHyperparameter:
        space[h.name] = hp.uniform(h.name, h.lower, h.upper)


def objective(x):
    config = deepcopy(x)
    for h in cs.get_hyperparameters():
        if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:

            config[h.name] = h.sequence[int(x[h.name])]

        elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:

            config[h.name] = int(x[h.name])
    y, c = b.objective_function(config)

    return {
        'config': config,
        'loss': y,
        'cost': c,
        'status': STATUS_OK}


init_pts = int(os.getenv('INIT_PTS', '0'))
init_bench = os.getenv('INIT_BENCH')
print(f'init_pts={init_pts}')
print(f'init_bench={init_bench}')
initial_points = None
if init_pts:
    from joblib import load

    data = load(f'store/{init_bench}/{args.run_id}.pkl')
    initial_points = [x[1].get_dictionary() for x in data[:init_pts]]
    output_path += f'_init_pts({init_bench},{init_pts})'

trials = Trials()
best = fmin(objective,
            space=space,
            algo=tpe.suggest,
            max_evals=args.n_iters,
            rstate=np.random.RandomState(args.run_id),
            trials=trials,
            points_to_evaluate=initial_points
            )

os.makedirs(os.path.join(output_path), exist_ok=True)

if args.benchmark == "nas_cifar10a" or args.benchmark == "nas_cifar10b" or args.benchmark == "nas_cifar10c":
    res = b.get_results(ignore_invalid_configs=True)
else:
    res = b.get_results()

fh = open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w')
json.dump(res, fh)
fh.close()
