#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json
import os
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from joblib import dump

info = {
    "bohb": ("HpBandSter-BOHB", "r",),
    "ultraopt_BOHB": ("UltraOpt-BOHB", "g",),
    "ultraopt_HyperBand": ("HyperBand", "b",),
    "tpe": ("HyperOpt-TPE", "r",),
    "ultraopt_ETPE": ("UltraOpt-ETPE", "g",),
    "ultraopt_Random": ("Random", "b",),
}

benchmarks = [
    "protein_structure",
    "slice_localization",
    "naval_propulsion",
    "parkinsons_telemonitoring"
]


def process(benchmark, fname):
    print(f"start, {benchmark}-{fname}")
    target_file = f"{benchmark}-{fname}.pkl"
    if os.path.exists(target_file):
        print(f"exist, {benchmark}-{fname}")
        return
    regret_tests = []
    runtimes = []
    ts = []
    df_t = pd.DataFrame()
    for file in Path(f"{benchmark}-{fname}").iterdir():
        if file.suffix != ".json":
            continue
        data = json.loads(file.read_text())
        col_name = file.name.split(".")[0]
        # regret_validation = data["regret_validation"]
        regret_test = data["regret_test"]
        for i in range(1, len(regret_test)):
            regret_test[i] = min(regret_test[i - 1], regret_test[i])
        regret_tests.append(regret_test)
        runtime = data["runtime"]
        runtimes.append(runtime)
        ts.extend(runtime)
        for timestamp, regret in zip(runtime, regret_test):
            df_t.loc[timestamp, col_name] = regret
    df_t.sort_index(inplace=True)
    n_rows = df_t.shape[0]
    for i, col in enumerate(df_t.columns):
        pre_max=None
        for j in range(n_rows):
            if pd.isna(df_t.iloc[j, i]):
                if pre_max is not None:
                    df_t.iloc[j, i] = pre_max
            else:
                pre_max = df_t.iloc[j, i]
    print(f"ok, {benchmark}-{fname}")
    dump(df_t, target_file)


args_list = []
for _, benchmark in enumerate(benchmarks):
    for fname in info.keys():
        args_list.append((benchmark, fname))

Parallel(n_jobs=10)(
    delayed(process)(*args) for args in args_list
)
