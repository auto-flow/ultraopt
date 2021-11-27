#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : qichun.tang@bupt.edu.cn
import io
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


def parse_eval_func_info(loss_info):
    if isinstance(loss_info, float):
        loss = loss_info
        nn_info = {}
    elif isinstance(loss_info, dict):
        loss = loss_info['loss']
        nn_info = loss_info.get('nn_info', {})
    elif loss_info is None:
        loss = 65535
        nn_info = {}
    else:
        raise NotImplementedError
    result = {
        "loss": loss,
        'nn_info': nn_info
    }
    return result


PREFIX = 'table | '


def dfMap_to_content(df_map: Dict[str, pd.DataFrame]):
    content = ""
    for key, df in df_map.items():
        content += f'{PREFIX}{key}\n'
        content += df.to_csv()
    return content


def content_to_dfMap(content: str):
    cur_table = None
    df_map = {}
    for line in content.splitlines():
        if line.startswith(PREFIX):
            cur_table = line[len(PREFIX):]
            df_map[cur_table] = []
        else:
            df_map[cur_table].append(line)
    for name, lines in df_map.items():
        df = pd.read_csv(io.StringIO("\n".join(lines)), index_col=0)
        df_map[name] = df
    return df_map

if __name__ == '__main__':
    df1=pd.DataFrame(
        [
            [1,2,3],
            [1,2,3],
            [1,2,3],
            [1,2,3],
        ],
        columns=[1,2,3],
        index=['a','b','c','d']
    )
    df2=pd.DataFrame(
        [
            [7,2,3],
            [8,2,3],
            [6,2,3],
            [5,2,3],
        ],
        columns=['col1','col2','col3'],
        index=['a', 'b', 'c', 'd']

    )
    df_map={
        'k1':df1,
        'k2':df2,
    }
    content=dfMap_to_content(df_map)
    df_map_get=content_to_dfMap(content)
    print(df_map_get)



