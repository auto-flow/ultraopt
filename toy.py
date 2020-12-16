#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-16
# @Contact    : tqichun@gmail.com
from neon.hdl.hdl2cs import HDL2CS

hdl = {
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
}

cs=HDL2CS().recursion(hdl)
print(cs)
