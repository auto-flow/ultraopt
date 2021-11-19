#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-23
# @Contact    : qichun.tang@bupt.edu.cn
import json
import os
from pathlib import Path

from experiments.combination_benchmarks.common import *
from hpbandster.core.dispatcher import Job
from joblib import Parallel, delayed
from tqdm import tqdm
from ultraopt.optimizer import ETPEOptimizer
from ultraopt import fmin
from ultraopt.utils.logging_ import setup_logger
from hpbandster.optimizers.config_generators.bohb import BOHB

setup_logger()


def evaluate(trial):
    cs.seed(trial)
    losses = []

    bohb = BOHB(configspace=cs, min_points_in_model=n_startup_trials,
                random_fraction=0, bandwidth_factor=1, num_samples=24)
    for i in tqdm(range(max_iter)):
        config = bohb.get_config(1)[0]
        loss = evaluator(config)
        job = Job(id=i, budget=1, config=config)
        job.result = {'loss': loss}
        bohb.new_result(job)
        losses.append(loss)
    return trial, losses


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
fname = f'results/BOHB-KDE'
fname+=f'-{benchmark}'
Path(f'{fname}.json').write_text(
    json.dumps(final_result)
)
print(m.to_list()[-1])
df.to_csv(f'{fname}.csv', index=False)
