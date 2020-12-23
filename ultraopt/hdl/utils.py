from copy import deepcopy
from typing import Dict, Union, Any

from ConfigSpace.configuration_space import Configuration

from ultraopt.hdl.hp_def import _decode


def is_hdl_bottom(key, value):
    if isinstance(key, str) and key.startswith("__"):
        return True
    if isinstance(value, dict) and "_type" in value:
        return True
    if not isinstance(value, dict):
        return True
    return False


def layering_config(c: Union[Configuration, Dict[str, Any]]):
    if isinstance(c, Configuration):
        dict_ = c.get_dictionary()
    elif isinstance(c, dict):
        dict_ = deepcopy(c)
    else:
        raise NotImplementedError
    result = {}
    for k, v in dict_.items():
        if isinstance(v, str):
            v = _decode(v)
        key_path = k.split(":")
        if key_path[-1] == "__choice__":
            key_path = key_path[:-1]
            if v is not None:
                key_path += [v]
                v = {}
        if "None" in key_path:
            continue
        __set_kv(result, key_path, v)
    return result


def __set_kv(dict_: dict, key_path: list, value):
    tmp = dict_
    for i, key in enumerate(key_path):
        if i != len(key_path) - 1:
            if key not in tmp:
                tmp[key] = {}
            tmp = tmp[key]
    key = key_path[-1]
    if (key == "placeholder" and value == "placeholder"):
        pass
    else:
        tmp[key] = value
