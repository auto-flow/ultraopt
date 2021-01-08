#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : qichun.tang@bupt.edu.cn
from collections import Counter
from copy import deepcopy
from typing import List, Union

import numpy as np
from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace, UniformFloatHyperparameter
from ConfigSpace import Configuration
from sklearn.utils.validation import check_random_state


def CS2HyperoptSpace(cs: ConfigurationSpace):
    '''一个将configspace转hyperopt空间的函数'''
    from hyperopt import hp
    result = {}
    for hyperparameter in cs.get_hyperparameters():
        name = hyperparameter.name
        if isinstance(hyperparameter, CategoricalHyperparameter):
            result[name] = hp.choice(name, hyperparameter.choices)
        elif isinstance(hyperparameter, UniformFloatHyperparameter):
            lower = hyperparameter.lower
            upper = hyperparameter.upper
            result[name] = hp.uniform(name, lower, upper)
        else:
            raise ValueError
        # todo: 考虑更多情况
    return result


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
    samples: list = sample_configurations(cs, n_configs)
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
    vec = np.array([sample.get_array() for sample in samples])
    for i in range(4):
        print(len(Counter(vec[:, i])))
    return samples


def sample_vectors(cs, n_samples):
    return np.array([sample.get_array() for sample in sample_configurations(cs, n_samples)])


def sample_configuration_except_default(cs: ConfigurationSpace, idx2val: dict, is_child_list=None,
                                        sampled_vectors=None, rng=None):
    if is_child_list is None:
        is_child_list = [True] * len(idx2val)
    rng = check_random_state(rng)
    if sampled_vectors is None:
        sampled_vectors = sample_vectors(cs, 5000)
    refined_vectors = sampled_vectors.copy()
    while True:
        ok = True
        for i, (idx, val) in enumerate(idx2val.items()):
            if is_child_list[i]:
                refined_vectors[:, idx] = val
            else:
                refined_vectors = refined_vectors[refined_vectors[:, idx] == val, :]
                if refined_vectors.shape[0] == 0:
                    ok = False
                    sampled_vectors = np.vstack([sampled_vectors, sample_vectors(cs, 5000)])
                    break
        if ok:
            break
    L = refined_vectors.shape[0]
    which = rng.randint(0, L)
    vector = refined_vectors[which, :]
    return Configuration(cs, vector=vector), sampled_vectors


def sample_configurations(config_space, n_configs=1):
    if n_configs == 1:
        return [config_space.sample_configuration(1)]
    elif n_configs > 1:
        return config_space.sample_configuration(n_configs)
    else:
        raise ValueError(f"n_configs should >=1")


def get_array_from_configs(configs: List[Configuration]):
    return np.array([config.get_array() for config in configs])


def get_dict_from_config(config: Union[dict, Configuration]):
    if isinstance(config, dict):
        return config
    return config.get_dictionary()


def initial_design_2(cs, n_configs, rng):
    cs = deepcopy(cs)
    rng = check_random_state(rng)
    hp2n_choices = {}
    idx_list = []
    is_child_list = []
    for idx, hp in enumerate(cs.get_hyperparameters()):
        if isinstance(hp, CategoricalHyperparameter) \
                and len(cs.get_parents_of(hp.name)) == 0 \
                and len(hp.choices) >= 3:
            hp2n_choices[hp.name] = len(hp.choices)
            idx_list.append(idx)
            is_child_list.append(len(cs.get_child_conditions_of(hp)) == 0)
    # todo: 考虑没有高基离散变量的情况
    if hp2n_choices:
        n_configs = max(n_configs, max(list(hp2n_choices.values())))
    matrix = np.zeros([n_configs, len(hp2n_choices)], dtype="int32")
    for i, (hp, n_choices) in enumerate(hp2n_choices.items()):
        col_vec = []
        while len(col_vec) < n_configs:
            col_vec.extend(np.arange(n_choices).tolist())
        matrix[:, i] = rng.choice(col_vec[:n_configs], n_configs, replace=False)
    samples = []
    sampled_vectors_ = sample_vectors(cs, 5000)
    for i in range(matrix.shape[0]):
        # todo: 开发一个固定几个变量，其他随机的函数
        vec = matrix[i, :]
        idx2val = dict(zip(idx_list, vec.tolist()))
        sample, sampled_vectors_ = sample_configuration_except_default(
            cs, idx2val, is_child_list, sampled_vectors_, rng)
        samples.append(sample)
    # todo: 把这个注释整理为一个单元测试
    # vec = np.array([sample.get_array() for sample in samples])
    # for i in range(4):
    #     print(len(Counter(vec[:, i])))
    return samples


def initial_design_cat(cs, n_configs):
    n_choices_list = []
    for hp in cs.get_hyperparameters():
        if isinstance(hp, CategoricalHyperparameter):
            n_choices_list.append(len(hp.choices))
        else:
            n_choices_list.append(0)
    samples = sample_configurations(cs, n_configs)
    # rng = check_random_state(rng)
    vectors = np.array([sample.get_array() for sample in samples])
    for i, n_choices in enumerate(n_choices_list):
        if n_choices > 0:
            counter = Counter(vectors[:, i])
            most_common = counter.most_common()
            if len(most_common) < n_choices:
                instances = [item[0] for item in most_common]
                sub = np.setdiff1d(np.arange(n_choices), instances)
                idx = 0
                for j in range(vectors.shape[0]):
                    if idx >= len(sub):
                        break
                    obj = vectors[j, i]
                    if counter[obj] > 1:
                        vectors[j, i] = sub[idx]
                        idx += 1
                        counter[obj] -= 1
    results = []
    for i in range(vectors.shape[0]):
        results.append(Configuration(cs, vector=vectors[i, :]))
    vec = np.array([sample.get_array() for sample in samples])
    for i in range(4):
        print(len(Counter(vec[:, i])))
    return results
