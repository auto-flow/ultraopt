#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : qichun.tang@bupt.edu.cn
import os
import shutil
from fractions import Fraction
from typing import Dict

import numpy as np
import pandas as pd
from joblib import dump

from ultraopt.constants import Configs
from ultraopt.utils.logging_ import get_logger

inc_logger = get_logger("incumbent_trajectory")


def pbudget(budget: float):
    if Configs.FractionalBudget:
        if budget - float(int(budget)) == 0:
            return str(int(budget))
        fraction = Fraction.from_float(budget)
        res = f"{fraction.numerator}/{fraction.denominator}"
        if Configs.AutoAdjustFractionalBudget and len(res) >= 8:
            Configs.FractionalBudget = False
            return pbudget(budget)
        else:
            return res
    return f"{budget:.2f}"


def print_incumbent_trajectory(chal_perf: float, inc_perf: float, challenger: dict, incumbent: dict, budget: float):
    inc_logger.info("Challenger (%.4f) is better than incumbent (%.4f) when budget is (%s)."
                    % (chal_perf, inc_perf, pbudget(budget)))
    # Show changes in the configuration
    params = sorted([(param, incumbent.get(param), challenger.get(param))
                     for param in challenger.keys()])
    inc_logger.info("Changes in incumbent:")
    for param in params:
        if param[1] != param[2]:
            inc_logger.info("  %s : %r -> %r" % (param))
        else:
            inc_logger.debug("  %s remains unchanged: %r" %
                             (param[0], param[1]))


def get_max_SH_iter(min_budget, max_budget, eta):
    return -int(np.log(min_budget / max_budget) / np.log(eta)) + 1


def get_import_error(pkg_name):
    raise ImportError(f"Cannot import {pkg_name}! Execute 'pip install {pkg_name}' in shell.")


def dump_checkpoint(optimizer, checkpoint_file):
    # todo: using thread
    checkpoint_file_bak = checkpoint_file + ".bak"
    if os.path.exists(checkpoint_file_bak):
        os.remove(checkpoint_file_bak)
    if os.path.exists(checkpoint_file):
        shutil.move(checkpoint_file, checkpoint_file_bak)
    dump(optimizer, checkpoint_file)


def dfMap_to_content(df_map: Dict[str, pd.DataFrame]):
    content = ""
    for key, df in df_map.items():
        content += f'{PREFIX}{key}\n'
        for name, row in df.iterrows():
            content += f"{name}\t"
            content += "\t".join([str(f) for f in row.tolist()])
            content += "\n"
    return content


def content_to_dfMap(content: str):
    cur_table = None
    df_map = {}
    for line in content.splitlines():
        if line.startswith(PREFIX):
            if cur_table is not None:
                df_map[cur_table] = pd.DataFrame(
                    df_map[cur_table]['data'], index=df_map[cur_table]['index'])
            cur_table = line[len(PREFIX):]
            df_map[cur_table] = {'index': [], 'data': []}
        else:
            words = line.split('\t')
            df_map[cur_table]['index'].append(words[0])
            df_map[cur_table]['data'].append([float(w) for w in words[1:]])
    if cur_table is not None:
        df_map[cur_table] = pd.DataFrame(
            df_map[cur_table]['data'], index=df_map[cur_table]['index'])
    return df_map


PREFIX = 'table | '