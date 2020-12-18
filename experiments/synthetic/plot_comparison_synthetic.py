#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json
from pathlib import Path

import numpy as np
import pylab as plt

info = {
    "hyperopt": ("r",),
    "ultraopt": ("g",)
}

benchs = list(json.loads(Path("ultraopt.json").read_text()).keys())

cols = int(np.sqrt(len(benchs)))
rows = int(np.ceil(len(benchs) / cols))
plt.rcParams['font.family'] = 'YaHei Consolas Hybrid'  # 设置字体样式
plt.rcParams['figure.figsize'] = (15, 12)
# plt.suptitle("对比")
# log_scale=True

for log_scale in [True, False]:
    plt.close()
    index = 1
    for bench in benchs:
        plt.subplot(rows, cols, index)
        for name, (color,) in info.items():
            mean_std = json.loads(Path(f"{name}.json").read_text())[bench]
            mean = np.array(mean_std["mean"])
            std = np.array(mean_std["std"])
            iters = range(len(mean))
            if not log_scale:
                plt.fill_between(
                    iters, mean - std, mean + std, alpha=0.1,
                    color=color
                )
            plt.plot(
                iters, mean, color=color, label=name, alpha=0.9
            )
        plt.title(bench)
        index += 1
        plt.xlabel("multi_fidelity")
        plt.ylabel("losses")
        if log_scale:
            plt.yscale("log")
        plt.grid(alpha=0.4)
        plt.legend(loc="best")
    title = "Comparison between UltraOpt and HyperOpt"
    if log_scale:
        title += "(log-scaled)"
    plt.suptitle(title)
    plt.tight_layout()
    fname = "tpe_synthetic"
    if log_scale:
        fname += "(log-scaled)"
    for suffix in ["pdf", "png"]:
        plt.savefig(f"{fname}.{suffix}")
    plt.show()
