#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn
import json
from pathlib import Path

import numpy as np
import pylab as plt

info = {
    "hyperopt": ("HyperOpt-TPE", "r",),
    "ultraopt_ETPE": ("UltraOpt-ETPE", "g",),
    "ultraopt_Random": ("Random", "b",),
}

benchs = list(json.loads(Path(f"{info.copy().popitem()[0]}.json").read_text()).keys())

cols = int(np.sqrt(len(benchs)))
rows = int(np.ceil(len(benchs) / cols))
# 设置字体样式
plt.rcParams['font.family'] = 'YaHei Consolas Hybrid'
plt.rcParams['figure.figsize'] = (15, 12)

for log_scale in [True, False]:
    plt.close()
    index = 1
    for bench in benchs:
        plt.subplot(rows, cols, index)
        for fname, (name, color,) in info.items():
            mean_std = json.loads(Path(f"{fname}.json").read_text())[bench]
            mean = np.array(mean_std["mean"])
            q1 = np.array(mean_std["q25"])
            if log_scale:
                q2 = np.array(mean_std["q90"])
            else:
                q2 = np.array(mean_std["q75"])
            iters = range(len(mean))
            # if not log_scale:
            plt.fill_between(
                iters, q1, q2, alpha=0.1,
                color=color
            )
            plt.plot(
                iters, mean, color=color, label=name, alpha=0.9
            )
        plt.title(bench)
        index += 1
        plt.xlabel("iterations")
        plt.ylabel("losses")
        if log_scale:
            plt.yscale("log")
        plt.grid(alpha=0.4)
        plt.legend(loc="best")
    title = "Comparison between Optimizers"
    if log_scale:
        title += "(log-scaled)"
    # plt.suptitle(title)
    plt.tight_layout()
    fname = "tpe_synthetic"
    if log_scale:
        fname += "(log-scaled)"
    for suffix in ["pdf", "png"]:
        plt.savefig(f"{fname}.{suffix}")
    plt.show()
