#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-23
# @Contact    : qichun.tang@bupt.edu.cn
import json
import os
from pathlib import Path

from experiments.combination_benchmarks.common import *
from joblib import Parallel, delayed
from ultraopt import fmin
from ultraopt.optimizer import ETPEOptimizer
from ultraopt.tpe import MutualInfomationSimilarity, SpearmanSimilarity, ConditionalMutualInfomationSimilarity
from ultraopt.utils.logging_ import setup_logger

setup_logger()

max_groups = int(sys.argv[5])
use_pretrain = (sys.argv[6]).lower() == 'true'
limit_max_groups = max_groups > 0

print(f"max_groups={max_groups}")
print(f"use_pretrain={use_pretrain}")

et_file = f'{benchmark}.txt'

pretrained_emb_expire_iter = int(os.getenv('pretrained_emb_expire_iter', '1000'))

sim = os.getenv('SIM', 'spearman')
category_encoder = os.getenv('category_encoder', 'embedding')
print(category_encoder)
if sim == 'spearman':
    sim='spearman3'
    similarity = SpearmanSimilarity(K=10)
elif sim == 'mi':
    similarity = MutualInfomationSimilarity()
elif sim=='cmi':
    similarity = ConditionalMutualInfomationSimilarity()
else:
    raise NotImplementedError
print(sim)

def evaluate(trial):
    optimizer = ETPEOptimizer(
        limit_max_groups=limit_max_groups,
        max_groups=max_groups,
        min_points_in_model=n_startup_trials,
        # optimize_each_varGroups=True,
        pretrained_emb=et_file if use_pretrain else None,
        pretrained_emb_expire_iter=pretrained_emb_expire_iter,
        similarity=similarity,
        category_encoder=category_encoder
    )
    ret = fmin(
        evaluator, cs, optimizer, random_state=trial * 10,
        n_iterations=max_iter,
    )
    losses = ret["budget2obvs"][1]["losses"]
    if not os.path.exists(et_file):
        ret.export_embedding_table(et_file)
        # 顺便把优化器存了，方便后期对数据进行分析
        # dump(ret,f'{benchmark}.bz2')
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
fname = f'results/ultraopt-ETPE-{sim}-{category_encoder}'
if limit_max_groups:
    fname += f"-g{max_groups}"
if use_pretrain:
    fname += f"-pretrain3"
else:
    fname += f"-noPretrain"
fname += f"-expIter{pretrained_emb_expire_iter}"
fname += f'-{benchmark}'
print(m.to_list()[-1])
if os.getenv('DEBUG') == 'true':
    exit(0)
Path(f'{fname}.json').write_text(
    json.dumps(final_result)
)
df.to_csv(f'{fname}.csv', index=False)
