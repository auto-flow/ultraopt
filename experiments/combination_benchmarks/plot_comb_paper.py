#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-23
# @Contact    : qichun.tang@bupt.edu.cn
import json
import os
from pathlib import Path

import numpy as np
import pylab as plt


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

info = {
    "Random": ("Random", ["r", '.', 'solid']),
    "hyperopt-TPE": ("HyperOpt-TPE", ["purple", 'v', 'solid']),
    "Optuna-TPE": ("Optuna-TPE",["g", '^', 'solid']),
    "BOHB-KDE": ("BOHB-KDE",["olive", 's', 'solid']),
    "ultraopt-ETPE": ("ETPE", ["b", 's', 'dashed']),
    # "ultraopt-ETPE-g3-noPretrain": ("ETPE($\max |g|=3$,ind)",["black", "x", 'dashed']),
    "ultraopt-ETPE-g3-noPretrain-expIter1000": ("ETPE($\max |g|=3$)",["brown", "x", 'dashed']),
    # "ultraopt-ETPE-g3-pretrain2-expIter200": ("ETPE($\max |g|=3$,200)",["r", "x", 'dashed']),
    # "ultraopt-ETPE-g3-pretrain2-expIter300": ("ETPE($\max |g|=3$,300)",["purple", "x", 'dashed']),
    # "ultraopt-ETPE-g3-pretrain2-expIter400": ("ETPE($\max |g|=3$,400)",["b", "x", 'dashed']),
    "ultraopt-ETPE-g3-pretrain2-expIter500": ("ETPE($\max |g|=3$,pretrain)",["k", "x", 'dashed']),
    # "ultraopt-ETPE-g3-pretrain": ("ETPE($\max |g|=3$,pretrain)",["orange", "x", 'dashed']),
    # "ultraopt-ETPE-g3-pretrain2": ("ETPE($\max |g|=3$,pretrain2)",["purple", "x", 'dashed']),
}

neg_RNA_global_min = 30

benchmarks = [
    ("PhoQ", (0, 36)),
    ("GB1", (0, 10)),
    ("RNA", (0, neg_RNA_global_min)),
]

# 设置字体样式
plt.rcParams['figure.figsize'] = (10, 8)

dir_name = 'comb_benchmark'
os.system(f'mkdir -p {dir_name}')

plt.close()
index = 1
iteration_truncate = 500
for benchmark, ylim in benchmarks:
    for fname, (name, color,) in info.items():
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        if "_g" in fname:
            fname, suffix = fname.split("_")
            path = f"results/{fname}-{benchmark}_{suffix}.json"
        else:
            path = f"results/{fname}-{benchmark}.json"
        mean_std = json.loads(Path(path).read_text())
        mean = np.array(mean_std["mean"])[:iteration_truncate]
        q1 = np.array(mean_std["q25"])[:iteration_truncate]
        q2 = np.array(mean_std["q75"])[:iteration_truncate]
        if benchmark == "RNA":
            mean += neg_RNA_global_min
            q1 += neg_RNA_global_min
            q2 += neg_RNA_global_min
        iters = range(len(mean))
        plt.ylim(*ylim)
        # if not log_scale:
        plt.fill_between(
            iters, q1, q2, alpha=0.1,
            color=color[0]
        )
        plt.plot(
            iters, mean, color=color[0], label=name, alpha=0.9,
            marker=color[1], linestyle=color[2], markevery=30

        )
    index += 1
    plt.xlabel("iterations")
    plt.ylabel("immediate regret")
    plt.legend(loc="best")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{dir_name}/{benchmark}.png")
    plt.savefig(f"{dir_name}/{benchmark}.pdf")
    plt.show()

target_dir = f'/data/Project/AutoML/ultraopt/paper_figures/{dir_name}'

os.system(f'rm -rf {target_dir}')
os.system(f'cp -r {dir_name} {target_dir}')
