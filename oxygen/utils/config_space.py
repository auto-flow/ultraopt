#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : tqichun@gmail.com
from copy import deepcopy

import numpy as np
from ConfigSpace import Configuration
from typing import List


def is_top_level_activated(config_space, config, hp_name, hp_value=None):
    parent_conditions = config_space.get_parent_conditions_of(hp_name)
    if len(parent_conditions):
        parent_condition = parent_conditions[0]
        parent_value = parent_condition.value
        parent_name = parent_condition.parent.name
        return is_top_level_activated(config_space, config, parent_name, parent_value)
    # 没有条件依赖，就是parent
    if hp_value is None:
        return True
    return config[hp_name] == hp_value


def deactivate(config_space, vector):
    result = deepcopy(vector)
    config = Configuration(config_space, vector=vector)
    for i, hp in enumerate(config_space.get_hyperparameters()):
        name = hp.name
        if not is_top_level_activated(config_space, config, name, None):
            result[i] = np.nan
    result_config = Configuration(configuration_space=config_space, vector=result)
    return result_config

def add_configs_origin(configs: List[Configuration], origin):
    if isinstance(configs, Configuration):
        configs = [configs]
    for config in configs:
        config.origin = origin
