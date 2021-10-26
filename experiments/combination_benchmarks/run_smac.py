#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-23
# @Contact    : qichun.tang@bupt.edu.cn


import json
from pathlib import Path
import os
from experiments.combination_benchmarks.common import *
from joblib import Parallel, delayed
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncDict
from ultraopt.hdl import hdl2cs

random_fraction = 0.33
n_trees = 10
max_feval = 4


def evaluate(trial):
    cs = hdl2cs(hdl)
    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": max_iter,
        "cs": cs,
        "deterministic": "false",
        "initial_incumbent": "RANDOM",
        "output_dir": "",
    })
    tae = ExecuteTAFuncDict(evaluator, use_pynisher=False)
    initial_configurations = cs.sample_configuration(n_startup_trials)
    # cs.seed(trial)
    smac = SMAC(
        scenario=scenario, tae_runner=tae,
        rng=np.random.RandomState(trial),
        initial_configurations=initial_configurations
    )
    # probability for random configurations

    smac.solver.random_configuration_chooser.prob = random_fraction
    smac.solver.model.rf_opts.num_trees = n_trees
    # only 1 configuration per SMBO iteration
    smac.solver.scenario.intensification_percentage = 1e-10
    smac.solver.intensifier.min_chall = 1
    # maximum number of function evaluations per configuration
    smac.solver.intensifier.maxR = max_feval

    smac.optimize()

    print('finish', trial)
    return trial, evaluator.losses


df = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)],
                  index=range(max_iter))

for trial, losses in Parallel(
        backend="multiprocessing", n_jobs=10)(
    delayed(evaluate)(trial) for trial in range(repetitions)
):
    df[f"trial-{trial}"] = losses
res = raw2min(df)
m = res.mean(1)
s = res.std(1)
final_result = {
    "mean": m.tolist(),
    "std": s.tolist(),
    "q10": res.quantile(0.10, 1).tolist(),
    "q25": res.quantile(0.25, 1).tolist(),
    "q75": res.quantile(0.75, 1).tolist(),
    "q90": res.quantile(0.90, 1).tolist()
}
os.system(f"mkdir -p results")
fname = f'results/SMAC-{benchmark}'
Path(f'{fname}.json').write_text(
    json.dumps(final_result)
)
print(m.to_list()[-1])
df.to_csv(f'{fname}.csv', index=False)
