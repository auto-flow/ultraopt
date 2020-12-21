import re
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List

from ConfigSpace import CategoricalHyperparameter, Constant
from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace import ForbiddenInClause, ForbiddenEqualsClause, ForbiddenAndConjunction
from ConfigSpace import InCondition, EqualsCondition

from ultraopt.hdl import hp_def
from ultraopt.hdl.utils import is_hdl_bottom
from ultraopt.utils.logging_ import get_logger


class HDL2CS():
    def __init__(self):
        self.logger = get_logger(__name__)

    def __call__(self, hdl: Dict):
        cs = self.recursion(hdl)
        return cs

    def __condition(self, item: Dict, store: Dict):
        child = item["_child"]
        child = store[child]
        parent = item["_parent"]
        parent = store[parent]
        value = (item["_values"])
        if (isinstance(value, list) and len(value) == 1):
            value = value[0]
        if isinstance(value, list):
            cond = InCondition(child, parent, list(map(hp_def._encode, value)))
        else:
            cond = EqualsCondition(child, parent, hp_def._encode(value))
        return cond

    def __forbidden(self, value: List, store: Dict, cs: ConfigurationSpace):
        assert isinstance(value, list)
        for item in value:
            assert isinstance(item, dict)
            clauses = []
            for name, forbidden_values in item.items():
                if isinstance(forbidden_values, list) and len(forbidden_values) == 1:
                    forbidden_values = forbidden_values[0]
                if isinstance(forbidden_values, list):
                    clauses.append(ForbiddenInClause(store[name], list(map(hp_def._encode, forbidden_values))))
                else:
                    clauses.append(ForbiddenEqualsClause(store[name], hp_def._encode(forbidden_values)))
            cs.add_forbidden_clause(ForbiddenAndConjunction(*clauses))

    def reverse_dict(self, dict_: Dict):
        reversed_dict = defaultdict(list)
        for key, value in dict_.items():
            if isinstance(value, list):
                for v in value:
                    reversed_dict[v].append(key)
            else:
                reversed_dict[value].append(key)
        reversed_dict = dict(reversed_dict)
        for key, value in reversed_dict.items():
            reversed_dict[key] = list(set(value))
        return reversed_dict

    def pop_covered_item(self, dict_: Dict, length: int):
        dict_ = deepcopy(dict_)
        should_pop = []
        for key, value in dict_.items():
            assert isinstance(value, list)
            if len(value) > length:
                self.logger.warning("len(value) > length")
                should_pop.append(key)
            elif len(value) == length:
                should_pop.append(key)
        for key in should_pop:
            dict_.pop(key)
        return dict_

    def __activate(self, value: Dict, store: Dict, cs: ConfigurationSpace):
        assert isinstance(value, dict)
        for k, v in value.items():
            assert isinstance(v, dict)
            reversed_dict = self.reverse_dict(v)
            reversed_dict = self.pop_covered_item(reversed_dict, len(v))
            for sk, sv in reversed_dict.items():
                cond = self.__condition(
                    {
                        "_child": sk,
                        "_values": sv,
                        "_parent": k
                    },
                    store,
                )
                cs.add_condition(cond)

    def eliminate_suffix(self, key: str):
        s = "(choice)"
        if key.endswith(s):
            key = key[:-len(s)]
        return key

    def add_configuration_space(
            self, cs: ConfigurationSpace, cs_name: str, hdl_value: dict, is_choice: bool,
            option_hp: Configuration, children_is_choice=False):
        if is_choice:
            cs.add_configuration_space(cs_name, self.recursion(hdl_value, children_is_choice),
                                       parent_hyperparameter={"parent": option_hp, "value": cs_name})
        else:
            cs.add_configuration_space(cs_name, self.recursion(hdl_value, children_is_choice))

    def recursion(self, hdl, is_choice=False):
        ############ Declare ConfigurationSpace variables ###################
        cs = ConfigurationSpace()
        ####### Fill placeholder to empty ConfigurationSpace ################
        key_list = list(hdl.keys())
        if len(key_list) == 0:
            cs.add_hyperparameter(Constant("placeholder", "placeholder"))
            return cs
        ###################### Declare common variables #####################
        option_hp = None
        pattern = re.compile(r"(.*)\((.*)\)")
        store = {}
        conditions_dict = {}
        ########### If parent is choice configuration_space #################
        if is_choice:
            choices = []
            for k, v in hdl.items():
                if not is_hdl_bottom(k, v) and isinstance(v, dict):
                    k = self.eliminate_suffix(k)
                    choices.append(self.eliminate_suffix(k))
            option_hp = CategoricalHyperparameter('__choice__', choices)
            cs.add_hyperparameter(option_hp)
        #### Travel key,value in hdl items, if value is dict(hdl), do recursion ######
        # fixme: 'option_hp' maybe reference without define ?
        for hdl_key, hdl_value in hdl.items():
            mat = pattern.match(hdl_key)
            # add_configuration_space (choice)
            if mat and isinstance(hdl_value, dict):
                groups = mat.groups()
                assert len(groups) == 2, ValueError(f"Invalid hdl_key {hdl_key}")
                cs_name, method = groups
                assert method == "choice", ValueError(f"Invalid suffix {method}")
                self.add_configuration_space(cs, cs_name, hdl_value, is_choice, option_hp, True)
            elif is_hdl_bottom(hdl_key, hdl_value):
                if hdl_key.startswith("__"):
                    conditions_dict[hdl_key] = hdl_value
                else:
                    hp = self.__parse_dict_to_config(hdl_key, hdl_value)
                    cs.add_hyperparameter(hp)
                    store[hdl_key] = hp
            # add_configuration_space
            elif isinstance(hdl_value, dict):
                cs_name = hdl_key
                self.add_configuration_space(cs, cs_name, hdl_value, is_choice, option_hp)
            else:
                raise NotImplementedError
        ########### Processing conditional hyperparameters #################
        for key, value in conditions_dict.items():
            condition_indicator = key
            if condition_indicator == "__condition":
                assert isinstance(value, list)
                for item in value:
                    cond = self.__condition(item, store)
                    cs.add_condition(cond)
            elif condition_indicator == "__activate":
                self.__activate(value, store, cs)
            elif condition_indicator == "__forbidden":
                self.__forbidden(value, store, cs)
            else:
                self.logger.warning(f"Invalid condition_indicator: {condition_indicator}")
            # fixme: remove 'rely_model'
        return cs
        # add_hyperparameter

    def __parse_dict_to_config(self, key, value):
        if isinstance(value, dict):
            _type = value.get("_type")
            _value = value.get("_value")
            _default = value.get("_default")
            assert _value is not None
            if _type in ("choice", "ordinal"):
                return eval(f"hp_def.{_type}(key, _value, _default)")
            else:
                return eval(f'''hp_def.{_type}("{key}",*_value,default=_default)''')
        else:
            return Constant(key, hp_def._encode(value))


def hdl2cs(hdl: dict) -> ConfigurationSpace:
    return HDL2CS()(hdl)
