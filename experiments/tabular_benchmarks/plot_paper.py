#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json
import os
from pathlib import Path

import pandas as pd
import pylab as plt

# plt.rcParams['font.family'] = 'YaHei Consolas Hybrid'
plt.rcParams['figure.figsize'] = (10, 8)

# plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]
})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "font.serif": ["Palatino"],
})
plt.rc('legend', fontsize=16)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ylims = [
    None,
    [3 * 10 ** (-6), 10 ** (-2)],
    [2 * 10 ** (-6), 10 ** (-2)],
    [10 ** (-3), 10 ** (-1)],
]
info = {
    "ultraopt_Random": ("Random", ["r", '.', 'solid']),
    "tpe": ("HyperOpt-TPE", ["purple", 'v', 'solid']),
    "tpe_init_pts(slice_localization,10)": ("HyperOpt-TPE", ["purple", 'v', 'dashed']),
    "optuna": ("Optuna-TPE", ["g", '^', 'solid']),
    "bohb-kde": ("BOHB-KDE", ["olive", 's', 'solid']),
    # "ultraopt_ETPE_18": ("ETPE", ["b", 's', 'dashed']),
    "ultraopt_ETPE_ord": ("ETPE", ["b", 's', 'dashed']),
    # "ultraopt_ETPE_univar": ("ETPE (univar)", ["brown", "x", 'dashed']),
    # "ultraopt_ETPE_ord": ("ETPE (ord)", ["g", "x", 'dashed']),
    # "ultraopt_ETPE_init_pts(slice_localization,20)": ("ETPE (init pts)", ["brown", "x", 'dashed']),
    "ultraopt_ETPE_init_pts(slice_localization,10)": ("ETPE (init pts)", ["k", "x", 'dashed']),
    # "ultraopt_ETPE_meta_learn(slice_localization,1.0,50,150)": ("ETPE (meta learn)", ["g", "x", 'dashed']),
    # "ultraopt_ETPE_meta_learn(slice_localization,0.5,50,150)": ("ETPE (meta learn)", ["g", "x", 'dashed']),
    #protein_structure-
    # "ultraopt_ETPE_meta_learn(slice_localization,0.51,50,100)": ("ETPE (meta learn)", ["g", "x", 'dashed']),
    "ultraopt_ETPE_meta_learn(parkinsons_telemonitoring,0.55,50,150)": ("ETPE (meta learn)", ["r", "x", 'dashed']),
    # "ultraopt_ETPE_meta_learn(slice_localization,0.3,30,100)": ("ETPE (meta learn)", ["g", "x", 'dashed']),
    # "ultraopt_ETPE_meta_learn(parkinsons_telemonitoring,0.5,50,150)": ("ETPE (meta learn)", ["g", "x", 'dashed']),
    # "ultraopt_ETPE_meta_learn(naval_propulsion,0.5,50,150)": ("ETPE (meta learn)", ["g", "x", 'dashed']),
    # "ultraopt_ETPE_meta_learn(parkinsons_telemonitoring,1.0,50,150)": ("ETPE (meta learn)", ["g", "x", 'dashed']),
}# protein_structure-ultraopt_ETPE_init_pts(slice_localization,20)
dir_name = 'tubular_benchmark_figures'
os.system(f'mkdir -p {dir_name}')
benchmarks = [
    "protein_structure",
    "slice_localization",
    "naval_propulsion",
    "parkinsons_telemonitoring"
]
max_iter = 300

print("| benchmark | method name | final loss |")
print("|-----------|-------------|------------|")
for idx, benchmark in enumerate(benchmarks):
    # plt.subplot(2, 2, idx + 1)
    # plt.title(benchmark)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ylim = ylims[idx]
    if ylim:
        plt.ylim(*ylim)
    plt.grid(alpha=0.4)

    for fname, (name, color,) in info.items():
        regret_tests = []
        for file in Path(f"results/{benchmark}-{fname}").iterdir():
            data = json.loads(file.read_text())
            # regret_test = data["regret_validation"]
            regret_test = data["regret_test"]
            for i in range(1, len(regret_test)):
                regret_test[i] = min(regret_test[i - 1], regret_test[i])
            regret_tests.append(regret_test)

        df_m = pd.DataFrame(regret_tests).T

        df_m = df_m.iloc[:max_iter, :]
        mean = df_m.mean(axis=1)
        print("|", benchmark, "|", name, "|", mean.to_list()[-1], "|")
        q1 = df_m.quantile(q=.25, axis=1)
        q2 = df_m.quantile(q=.9, axis=1)
        iters = range(df_m.shape[0])
        plt.fill_between(
            iters, q1, q2, alpha=0.1,
            color=color[0]
        )
        plt.plot(
            iters, mean, color=color[0], label=name, alpha=0.9,
            marker=color[1], linestyle=color[2], markevery=30
        )
    plt.legend(loc="best")
    plt.xlabel("iterations")
    plt.ylabel("immediate test regret")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"{dir_name}/{idx}_{benchmark}.png")
    plt.savefig(f"{dir_name}/{idx}_{benchmark}.pdf")
    plt.show()

target_dir = f'/data/Project/AutoML/ultraopt/paper_figures/{dir_name}'

os.system(f'rm -rf {target_dir}')
os.system(f'cp -r {dir_name} {target_dir}')
