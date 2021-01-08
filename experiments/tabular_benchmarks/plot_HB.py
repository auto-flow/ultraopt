#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import pandas as pd
import pylab as plt
from joblib import load

m1 = "o"
m2 = "^"
info = {
    "tpe": ("HyperOpt-TPE", "r", "dashed", m1),
    "ultraopt_ETPE": ("UltraOpt-ETPE", "g", "solid", m1),
    "ultraopt_Random": ("Random", "b", "dotted", m1),
    "bohb": ("HpBandSter-BOHB", "purple", "dashed", m2),
    "ultraopt_BOHB": ("UltraOpt-BOHB", "olive", "solid", m2),
    "ultraopt_HyperBand": ("HyperBand", "dodgerblue", "dotted", m2),
}
plt.rcParams['font.family'] = 'YaHei Consolas Hybrid'
plt.rcParams['figure.figsize'] = (8, 6)

benchmarks = ["naval_propulsion"]
benchmarks = ["protein_structure", ]  # "slice_localization", "naval_propulsion", "parkinsons_telemonitoring"]
for idx, benchmark in enumerate(benchmarks):
    # plt.subplot(2, 2, idx + 1)
    plt.title(benchmark)
    for fname, (name, color, linestyle, marker) in info.items():
        df_t: pd.DataFrame = load(f"{benchmark}-{fname}.pkl")
        plt.grid()
        mean = df_t.mean(axis=1)
        q1 = df_t.quantile(q=.25, axis=1)
        q2 = df_t.quantile(q=.9, axis=1)
        iters = df_t.index
        plt.grid()
        plt.fill_between(
            iters, q1, q1 + q2, alpha=0.1,
            color=color
        )
        plt.plot(
            iters, mean, c=color, label=name, alpha=0.9,
            linestyle=linestyle,
            # marker=marker, markevery=2000
        )
        # todo: log scale 下marker好看
        plt.grid(alpha=0.4)
        plt.legend(loc="best")
        plt.xlabel("estimated wall-clock time (seconds)")
        plt.ylabel("immediate test regret")
        plt.yscale("log")
        plt.xscale("log")
        plt.xlim([10, 10e4])
plt.tight_layout()
# from textwrap import wrap
# title="Comparison of Optimizers in Tabular-Benchmarks"
# plt.suptitle("\n".join(wrap(title, 30)))

plt.savefig("tabular_benchmarks_HB.png")
plt.savefig("tabular_benchmarks_HB.pdf")
plt.show()
