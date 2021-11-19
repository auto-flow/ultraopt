#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-23
# @Contact    : qichun.tang@bupt.edu.cn
import json
import os
from copy import deepcopy
from pathlib import Path

from experiments.combination_benchmarks.common import *
from hpbandster.core.dispatcher import Job
from joblib import Parallel, delayed
from optuna.samplers import TPESampler
from tqdm import tqdm
from ultraopt.optimizer import ETPEOptimizer
from ultraopt import fmin
from ultraopt.utils.logging_ import setup_logger
from hpbandster.optimizers.config_generators.bohb import BOHB
import optuna
setup_logger()


def evaluate(trial):
    cur_evaluator=deepcopy(evaluator)

    tpe = TPESampler(
        n_startup_trials=n_startup_trials, multivariate=True,
        seed=trial * 10)
    study = optuna.create_study(sampler=tpe)
    study.optimize(cur_evaluator.run_optuna, n_trials=max_iter)
    # res[f"trial-{trial}"] =
    return trial, cur_evaluator.losses

df = pd.DataFrame(columns=[f"trial-{i}" for i in range(repetitions)],
                  index=range(max_iter))

for trial, losses in Parallel(
        backend="multiprocessing", n_jobs=-1)(
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
fname = f'results/Optuna-TPE'
fname+=f'-{benchmark}'
Path(f'{fname}.json').write_text(
    json.dumps(final_result)
)
print(m.to_list()[-1])
df.to_csv(f'{fname}.csv', index=False)
