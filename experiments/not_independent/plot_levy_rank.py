#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@bupt.edu.cn

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
    "font.size": 20,
    "font.serif": ["Palatino"],
})

# benchmark = sys.argv[1]
benchmark = 'rosenbrock'
#https://matplotlib.org/stable/api/markers_api.html
#https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
info = {
    f"random_{benchmark}": ("Random", ["r",'.', 'solid']),
    f"hyperopt_{benchmark}": ("HyperOpt-TPE", ["purple",'v', 'solid']),
    f"optuna_{benchmark}_multival": ("Optuna-TPE",[ "g",'^', 'solid']),
    f"bohb_{benchmark}": ("BOHB-KDE", ["olive",'s', 'solid']),
    f"ultraopt_{benchmark}_multival": ("ETPE", ["b",'s', 'dashed']),
    f"ultraopt_{benchmark}_g3_multival": ("ETPE($\max |g|=3$)",[ "brown","x", 'dashed']),
}

cls = benchmark[0].upper() + benchmark[1:]
benchs = [f"{cls}{x}D" for x in range(2, 21)]
from joblib import load

cols = int(np.sqrt(len(benchs)))
rows = int(np.ceil(len(benchs) / cols))
# 设置字体样式
# plt.rcParams['font.family'] = 'YaHei Consolas Hybrid'
plt.rcParams['figure.figsize'] = (10, 8)
plt.rc('legend',fontsize=16)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
T, D, F = 100, 19, 6

name2score_mat = {}
colors = []
for fname in info:
    label_name, color = info[fname]
    colors.append(color)
    name2df = load(fname + ".pkl")
    scores_list = []
    for bench in benchs:
        df = name2df[bench]
        scores = df.min(axis=0).tolist()
        scores_list.append(scores)
    score_mat = np.array(scores_list)  # [Td * Ttrial]
    name2score_mat[label_name] = score_mat

names = list(name2score_mat.keys())
mat = np.concatenate([name2score_mat[name] for name in names]).reshape([F, D, T])
ans = np.zeros([F, D, T])
for i in range(D):
    for j in range(T):
        idx = np.argsort(mat[:, i, j])
        top = [x for _, x in sorted(zip(idx, range(F)))]
        ans[:, i, j] = top

mean_arr = np.zeros([F, D])
median_arr = np.zeros([F, D])
q25_arr = np.zeros([F, D])
q75_arr = np.zeros([F, D])
std_arr = np.zeros([F, D])

for f in range(F):
    for d in range(D):
        mean_arr[f, d] = np.mean(ans[f, d, :])
        median_arr[f, d] = np.median(ans[f, d, :])
        q25_arr[f, d] = np.quantile(ans[f, d, :], q=0.25)
        q75_arr[f, d] = np.quantile(ans[f, d, :], q=0.75)
        std_arr[f, d] = np.std(ans[f, d, :])

def x_formatter(x,pos):
    print(x,pos)
    if x in (10,14):
        return ""
    if x ==12:
        return f"rosenbrock($d=12$)"
    return f"$d={x}$"

print(names[3])
print(median_arr[3, :])
print(names[4])
print(median_arr[4, :])

for i, name in enumerate(names):
    # todo: IO太多了
    m, s = mean_arr[i, :], std_arr[i, :]
    iters = range(2,D+2)
    # plt.fill_between(
    #     iters, m - s, m + s, alpha=0.1,
    #     color=colors[i]
    # )
    plt.plot(
        iters, m, color=colors[i][0], label=name, alpha=0.9, marker=colors[i][1], linestyle=colors[i][2]
    )
plt.legend(loc='best')
plt.ylabel('avg rank')
plt.grid(alpha=0.5)
# plt.xlabel('rosenbrock function with increased dimension ')
ax.set_xticks(range(2,D+2,2))
ax.xaxis.set_major_formatter(plt.FuncFormatter(x_formatter))
plt.tight_layout()
plt.savefig('rosenbrock_inc_dim.png')
plt.savefig('rosenbrock_inc_dim.pdf')

plt.show()
import os
os.system('mkdir -p ../../paper_figures')
os.system(f'cp rosenbrock_inc_dim.png ../../paper_figures')
os.system(f'cp rosenbrock_inc_dim.pdf ../../paper_figures')