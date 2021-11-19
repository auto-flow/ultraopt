#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-23
# @Contact    : qichun.tang@bupt.edu.cn
import json
from pathlib import Path

import numpy as np
import pylab as plt

info = {
    "hyperopt-TPE": ("hyperopt-TPE", "purple",),
    "ultraopt-ETPE": ("ultraopt-ETPE", "r",),
    "ultraopt-ETPE_g3": ("ultraopt-ETPE_g3", "r",),
    "ultraopt-ETPE_g4": ("ultraopt-ETPE_g4", "g",),
    "ultraopt-ETPE_g5": ("ultraopt-ETPE_g5", "brown",),
    "ultraopt-ETPE_g6": ("ultraopt-ETPE_g6", "b",),
    "ultraopt-ETPE_g8": ("ultraopt-ETPE_g8", "k",),
}

benchmarks = [
    ("GB1", (0, 10)),
    ("PhoQ", (9, 36)),
    ("RNA", (-30, 0)),
]

# 设置字体样式
plt.rcParams['font.family'] = 'YaHei Consolas Hybrid'
plt.rcParams['figure.figsize'] = (10, 8)

plt.close()
index = 1
iteration_truncate = 500
for benchmark, ylim in benchmarks:
    plt.subplot(2, 2, index)
    for fname, (name, color,) in info.items():
        if "_" in fname:
            fname, suffix = fname.split("_")
            path = f"results/{fname}-{benchmark}_{suffix}.json"
        else:
            path = f"results/{fname}-{benchmark}.json"
        mean_std = json.loads(Path(path).read_text())
        mean = np.array(mean_std["mean"])[:iteration_truncate]
        q1 = np.array(mean_std["q25"])[:iteration_truncate]
        q2 = np.array(mean_std["q75"])[:iteration_truncate]
        iters = range(len(mean))
        plt.ylim(*ylim)
        # if not log_scale:
        plt.fill_between(
            iters, q1, q2, alpha=0.1,
            color=color
        )
        plt.plot(
            iters, mean, color=color, label=name, alpha=0.9
        )
    index += 1
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.title(benchmark)
    plt.grid(alpha=0.4)
    plt.legend(loc="best")
# plt.yscale("log")
# plt.xscale("symlog")
# title = "Comparison between Optimizers"
task = 'combination_optimization'
plt.suptitle(f"{task}")
plt.tight_layout()
for suffix in ["pdf", "png"]:
    plt.savefig(f"{task}.{suffix}")
plt.show()
