#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json
from collections import defaultdict

import numpy as np
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

benchmarks = [
    "protein_structure",
    "slice_localization",
    "naval_propulsion",
    "parkinsons_telemonitoring"
]
xlim_list = [
    [10, 1e5],
    [80, 2 * 1e5],
    [10, 0.4 * 1e5],
    [10, 0.2 * 1e5],
]
ylim_list = [
    None,
    [0.5 * 1e-5, 0.99],
    None,
    None
]
performances = [
    0.001,
    0.00005,
    0.000001,
    0.02,
]
time_compare = defaultdict(dict)

for idx, (benchmark, xlim, ylim) in enumerate(zip(benchmarks, xlim_list, ylim_list)):
    # plt.subplot(2, 2, idx + 1)
    print(benchmark)
    plt.title(benchmark)
    for fname, (name, color, linestyle, marker) in info.items():
        df_t: pd.DataFrame = load(f"{benchmark}-{fname}.pkl")
        plt.grid()
        i = 0
        while True:
            sum_ = (~pd.isna(df_t.iloc[i, :])).sum()
            if sum_ >= 15:
                break
            i += 1
        df_t = df_t.iloc[i:, :]
        for col in df_t.columns:
            isna = pd.isna(df_t[col])
            df_t.loc[isna, col] = df_t.loc[~isna, col].values[0]
        mean = df_t.mean(axis=1).tolist()
        t_index = np.searchsorted(-np.array(mean), -performances[idx], 'left')
        if t_index >= len(df_t.index):
            t_index = t_index - 1
        time = df_t.index[t_index]
        print(name, ":", time)
        time_compare[benchmark][name] = time
        time_compare[benchmark]['full_time']=df_t.index[-1]
        # for i in range(1, len(mean)):
        #     mean[i] = min(mean[i], mean[i - 1])
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
        plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        # plt.ylim([0.001, 0.99])
    plt.tight_layout()
    # from textwrap import wrap
    # title="Comparison of Optimizers in Tabular-Benchmarks"
    # plt.suptitle("\n".join(wrap(title, 30)))

    plt.savefig(f"{benchmark}_HB.png")
    plt.savefig(f"{benchmark}_HB.pdf")
    plt.show()
json.dump(time_compare, open('time_compare.json', 'w'))
