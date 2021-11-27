import argparse
import json
import logging
import os
import time

import ConfigSpace

logging.basicConfig(level=logging.ERROR)

from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from ultraopt.utils.net import get_a_free_port
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
parser.add_argument('--strategy', default="sampling", type=str, nargs='?',
                    help='optimization strategy for the acquisition function')
parser.add_argument('--min_bandwidth', default=.3, type=float, nargs='?', help='minimum bandwidth for KDE')
parser.add_argument('--num_samples', default=64, type=int, nargs='?',
                    help='number of samples for the acquisition function')
parser.add_argument('--random_fraction', default=.33, type=float, nargs='?', help='fraction of random configurations')
parser.add_argument('--bandwidth_factor', default=3, type=int, nargs='?', help='factor multiplied to the bandwidth')

args = parser.parse_args()

if args.benchmark == "nas_cifar10a":
    min_budget = 4
    max_budget = 108
    b = NASCifar10A(data_dir=args.data_dir)

elif args.benchmark == "nas_cifar10b":
    b = NASCifar10B(data_dir=args.data_dir)
    min_budget = 4
    max_budget = 108

elif args.benchmark == "nas_cifar10c":
    b = NASCifar10C(data_dir=args.data_dir)
    min_budget = 4
    max_budget = 108

elif args.benchmark == "protein_structure":
    b = FCNetProteinStructureBenchmark(data_dir=args.data_dir)
    min_budget = 3
    max_budget = 100

elif args.benchmark == "slice_localization":
    b = FCNetSliceLocalizationBenchmark(data_dir=args.data_dir)
    min_budget = 3
    max_budget = 100

elif args.benchmark == "naval_propulsion":
    b = FCNetNavalPropulsionBenchmark(data_dir=args.data_dir)
    min_budget = 3
    max_budget = 100

elif args.benchmark == "parkinsons_telemonitoring":
    b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=args.data_dir)
    min_budget = 3
    max_budget = 100

output_path = os.path.join(args.output_path, f"{args.benchmark}-bohb")
os.makedirs(os.path.join(output_path), exist_ok=True)

if args.benchmark == "protein_structure" or \
        args.benchmark == "slice_localization" or args.benchmark == "naval_propulsion" \
        or args.benchmark == "parkinsons_telemonitoring":
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
else:
    cs = b.get_configuration_space()


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


hb_run_id = f'{args.run_id}'

NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
ns_host, ns_port = NS.start()

num_workers = 1

workers = []
for i in range(num_workers):
    w = MyWorker(nameserver=ns_host, nameserver_port=ns_port,
                 run_id=hb_run_id,
                 id=i)
    w.run(background=True)
    workers.append(w)

bohb = BOHB(configspace=cs,
            run_id=hb_run_id,
            eta=3, min_budget=min_budget, max_budget=max_budget,
            nameserver=ns_host,
            nameserver_port=ns_port,
            # optimization_strategy=args.strategy,
            num_samples=args.num_samples,
            random_fraction=args.random_fraction, bandwidth_factor=args.bandwidth_factor,
            ping_interval=10, min_bandwidth=args.min_bandwidth)

results = bohb.run(args.n_iters, min_n_workers=num_workers)

bohb.shutdown(shutdown_workers=True)
NS.shutdown()
time.sleep(5)

if args.benchmark == "nas_cifar10a" or args.benchmark == "nas_cifar10b" or args.benchmark == "nas_cifar10c":
    res = b.get_results(ignore_invalid_configs=True)
else:
    res = b.get_results()

fh = open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w')
json.dump(res, fh)
fh.close()
