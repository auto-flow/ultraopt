'''
smac==0.10.0
'''
import argparse
import json
import os

import numpy as np
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncDict
from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark, \
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--benchmark', default="protein_structure", type=str, nargs='?', help='specifies the benchmark')
parser.add_argument('--n_iters', default=25, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="./", type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--n_trees', default=10, type=int, nargs='?', help='number of trees for the random forest')
parser.add_argument('--random_fraction', default=.33, type=float, nargs='?', help='fraction of random configurations')
parser.add_argument('--max_feval', default=4, type=int, nargs='?',
                    help='maximum number of function evaluation per configuration')

args = parser.parse_args()

if args.benchmark == "nas_cifar10a":
    b = NASCifar10A(data_dir=args.data_dir)

elif args.benchmark == "nas_cifar10b":
    b = NASCifar10B(data_dir=args.data_dir, multi_fidelity=False)

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

output_path = os.path.join(args.output_path, f"{args.benchmark}-smac")
os.makedirs(os.path.join(output_path), exist_ok=True)

cs = b.get_configuration_space()

scenario = Scenario({
    "run_obj": "quality",
    "runcount-limit": args.n_iters,
    "cs": cs,
    "deterministic": "true",
    "initial_incumbent": "RANDOM",
    "output_dir": ""
})


def objective_function(config, **kwargs):
    y, c = b.objective_function(config)
    return float(y)


tae = ExecuteTAFuncDict(objective_function, use_pynisher=False)
smac = SMAC(scenario=scenario, tae_runner=tae, rng=np.random.RandomState(args.run_id))

# probability for random configurations

smac.solver.random_configuration_chooser.prob = args.random_fraction
smac.solver.model.rf_opts.num_trees = args.n_trees
# only 1 configuration per SMBO iteration
smac.solver.scenario.intensification_percentage = 1e-10
smac.solver.intensifier.min_chall = 1
# maximum number of function evaluations per configuration
smac.solver.intensifier.maxR = args.max_feval

smac.optimize()

if args.benchmark == "nas_cifar10a" or args.benchmark == "nas_cifar10b" or \
        args.benchmark == "nas_cifar10c":
    res = b.get_results(ignore_invalid_configs=True)
else:
    res = b.get_results()

fh = open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w')
json.dump(res, fh)
fh.close()
