#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-23
# @Contact    : qichun.tang@bupt.edu.cn
import json
import os
from functools import partial
from pathlib import Path

from experiments.combination_benchmarks.common import *
from hyperopt import tpe, fmin, Trials
from joblib import Parallel, delayed


def evaluate(trial):
    trials = Trials()
    best = fmin(
        evaluator, space, algo=partial(tpe.suggest, n_startup_jobs=n_startup_trials),
        max_evals=max_iter,
        rstate=np.random.RandomState(trial * 10), trials=trials,
    )
    losses = trials.losses()
    return trial, losses


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
fname = f'results/hyperopt-TPE-{benchmark}'
Path(f'{fname}.json').write_text(
    json.dumps(final_result)
)
print(m.to_list()[-1])
df.to_csv(f'{fname}.csv', index=False)
