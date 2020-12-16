#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-16
# @Contact    : tqichun@gmail.com
import pandas as pd

from ultraopt.hdl import HDL2CS
from ultraopt.utils.config_space import get_array_from_configs
from ultraopt.utils.config_space import initial_design_2

cs = HDL2CS().recursion({
    "A(choice)": {
        "A1": {
            "A1_h1": {
                "_type": "choice",
                "_value": ["1", "2", "3"]
            },
            "A1_h2": {
                "_type": "uniform",
                "_value": [-999, 666]
            }
        },
        "A2": {
            "A2_h1": {
                "_type": "choice",
                "_value": ["1", "2", "3"]
            },
            "A2_h2": {
                "_type": "uniform",
                "_value": [-999, 666]
            }

        },
        "A3": {
            "A3_h1": {
                "_type": "choice",
                "_value": ["1", "2", "3"]
            },
            "A3_h2": {
                "_type": "uniform",
                "_value": [-999, 666]
            }
        },
    }
})
configs = initial_design_2(cs, 1, 1)
arr = get_array_from_configs(configs)
print(pd.Series(arr[:, 0]).nunique() == 3)
