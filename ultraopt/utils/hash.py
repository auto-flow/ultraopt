#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-15
# @Contact    : qichun.tang@bupt.edu.cn
import hashlib
from copy import deepcopy
from typing import Union, Dict, Any

import numpy as np
from ConfigSpace import Configuration
from scipy.sparse import issparse


def get_hash_of_array(X, m=None):
    if m is None:
        m = hashlib.md5()

    if issparse(X):
        m.update(X.indices)
        m.update(X.indptr)
        m.update(X.data)
        m.update(str(X.shape).encode('utf8'))
    else:
        if X.flags['C_CONTIGUOUS']:
            m.update(X.data)
            m.update(str(X.shape).encode('utf8'))
        else:
            X_tmp = np.ascontiguousarray(X.T)
            m.update(X_tmp.data)
            m.update(str(X_tmp.shape).encode('utf8'))

    hash = m.hexdigest()
    return hash


def sort_dict(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = sort_dict(v)
        return dict(sorted(obj.items(), key=lambda x: str(x[0])))
    elif isinstance(obj, list):
        for i, elem in enumerate(obj):
            obj[i] = sort_dict(elem)
        return list(sorted(obj, key=str))
    else:
        return obj


def get_hash_of_dict(dict_, m=None):
    if m is None:
        m = hashlib.md5()

    sorted_dict = sort_dict(deepcopy(dict_))
    # sorted_dict = deepcopy(dict_)
    m.update(str(sorted_dict).encode("utf-8"))
    return m.hexdigest()


def get_hash_of_config(config: Union[Configuration, Dict[str, Any]], m=None):
    if m is None:
        m = hashlib.md5()
    assert isinstance(config, (dict, Configuration))
    if isinstance(config, Configuration):
        config = config.get_dictionary()
    return get_hash_of_dict(config, m)
