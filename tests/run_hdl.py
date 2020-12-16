#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-16
# @Contact    : tqichun@gmail.com
from ultraopt.hdl import HDL2CS

# 测试嵌套依赖
cs = HDL2CS()({
    "A(choice)": {
        "A1(choice)": {
            "B1": {
                "B1_h": {
                    "_type": "choice",
                    "_value": ["1", "2", "3"]
                },
            },
            "B2": {
                "B2_h": {
                    "_type": "choice",
                    "_value": ["1", "2", "3"]
                },
            },
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
print(cs)

# 测试
