#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : tqichun@gmail.com
from collections import Counter
from copy import deepcopy
from typing import List

import numpy as np
from ConfigSpace import CategoricalHyperparameter
from ConfigSpace import Configuration


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



def initial_design(cs, n_configs):
    # todo: 将用户指定的 initial points 也纳入考虑中
    # todo: 更智能的方式
    # fixme: 完成HDL模块后， 添加单元测试。 目前的单元测试在autoflow代码中
    cs = deepcopy(cs)
    n_choices_list = []
    for hp in cs.get_hyperparameters():
        if isinstance(hp, CategoricalHyperparameter):
            n_choices_list.append(len(hp.choices))
        else:
            n_choices_list.append(0)
    n_choices_vec = np.array(n_choices_list)
    high_r_ix = np.arange(len(n_choices_list))[n_choices_vec >= 3]
    samples: list = cs.sample_configuration(n_configs)
    samples = [samples] if not isinstance(samples, list) else samples
    # rng = check_random_state(rng)
    while True:
        vectors = np.array([sample.get_array() for sample in samples])
        vectors[np.isnan(vectors)] = -1
        ok = True
        for ix in high_r_ix:
            col_vec = vectors[:, ix]
            col_vec = col_vec[col_vec != -1]
            counter = Counter(col_vec)
            k, cnt = counter.most_common()[-1]
            k = int(k)
            if len(counter) < n_choices_vec[ix]:
                ok = False
                # hp = cs.get_hyperparameter(cs.get_hyperparameter_by_idx(ix))
                # hp.default_value = hp.choices[k]
                break
        if ok:
            break
        samples.append(cs.get_default_configuration())
    return samples