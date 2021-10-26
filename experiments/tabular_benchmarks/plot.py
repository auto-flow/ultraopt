#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json
from pathlib import Path

import pandas as pd
import pylab as plt

info = {
    "tpe": ("HyperOpt-TPE", "r",),
    "ultraopt_ETPE": ("UltraOpt-ETPE", "g",),
    "ultraopt_ETPE_univar": ("UltraOpt-ETPE_univar", "k",),
    # "hpbandster": ("hpbandster", "k",),
    # "ultraopt_ETPE_univar_cat": ("UltraOpt-ETPE_univar_cat", "orange",),
    # "ultraopt_ETPE_2": ("UltraOpt-ETPE-2", "orange",),
    # "ultraopt_ETPE_3": ("UltraOpt-ETPE-3", "k",),
    "ultraopt_ETPE_7": ("UltraOpt-ETPE-7", "b",),
    # "smac": ("SMAC-0.10", "orange",),
    # "smac_0.12": ("SMAC-0.12", "purple",),
    # "ultraopt_Forest": ("UltraOpt-SMAC", "k",),
    "ultraopt_Forest_local_search": ("UltraOpt-SMAC-local_search", "brown",),
    "ultraopt_Random": ("Random", "b",),
}
plt.rcParams['font.family'] = 'YaHei Consolas Hybrid'
plt.rcParams['figure.figsize'] = (12, 10)

benchmarks = [
    "protein_structure",
    # "slice_localization",
    # "naval_propulsion",
    # "parkinsons_telemonitoring"
]
max_iter=500

print("| benchmark | method name | final loss |")
print("|-----------|-------------|------------|")
for idx, benchmark in enumerate(benchmarks):
    plt.subplot(2, 2, idx + 1)
    plt.title(benchmark)
    for fname, (name, color,) in info.items():
        regret_tests = []
        for file in Path(f"{benchmark}-{fname}").iterdir():
            data = json.loads(file.read_text())
            # regret_test = data["regret_validation"]
            regret_test = data["regret_test"]
            for i in range(1, len(regret_test)):
                regret_test[i] = min(regret_test[i - 1], regret_test[i])
            regret_tests.append(regret_test)

        df_m = pd.DataFrame(regret_tests).T

        df_m = df_m.iloc[:max_iter, :]
        plt.grid()
        mean = df_m.mean(axis=1)
        print("|", benchmark, "|", name, "|", mean.to_list()[-1], "|")
        q1 = df_m.quantile(q=.25, axis=1)
        q2 = df_m.quantile(q=.9, axis=1)
        iters = range(df_m.shape[0])
        plt.grid()
        plt.fill_between(
            iters, q1, q1 + q2, alpha=0.1,
            color=color
        )

        plt.plot(
            iters, mean, color=color, label=name, alpha=0.9
        )
        plt.grid(alpha=0.4)
        plt.legend(loc="best")
        plt.xlabel("iterations")
        plt.ylabel("immediate test regret")
        plt.yscale("log")
plt.tight_layout()
# from textwrap import wrap
# title="Comparison of Optimizers in Tabular-Benchmarks"
# plt.suptitle("\n".join(wrap(title, 30)))

plt.savefig("tabular_benchmarks.png")
plt.savefig("tabular_benchmarks.pdf")
plt.show()
