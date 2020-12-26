#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-23
# @Contact    : qichun.tang@bupt.edu.cn
from ultraopt.hdl.utils import is_hdl_bottom
from ultraopt.utils.misc import get_import_error

SUFFIX = "(choice)"


def plot_hdl_recurse(hdl: dict, g, parent=None):
    for key, value in hdl.items():
        if key.endswith(SUFFIX):
            key = key[:-len(SUFFIX)]
            g.node(key, shape="hexagon")
            plot_hdl_recurse(value, g, key)
        elif is_hdl_bottom(key, value):
            if (key.startswith("__")):
                key = None
            else:
                g.node(key, shape="ellipse")
        else:
            g.node(key, shape="box")
            plot_hdl_recurse(value, g, key)
        if parent and key:
            g.edge(parent, key)


def plot_hdl(hdl: dict, title="Config Space"):
    try:
        from graphviz import Digraph
    except Exception as e:
        raise get_import_error("graphviz")
    g = Digraph(title)
    plot_hdl_recurse(hdl, g)
    return g

def plot_layered_dict_recurse(layered_dict: dict, g, parent=None):
    for key,value in layered_dict.items():
        g.node(key, shape="ellipse")
        if isinstance(value,dict):
            plot_layered_dict_recurse(value,g,parent)
        if parent and key:
            g.edge(parent, key)


def plot_layered_dict(layered_dict: dict, title="Layered Dict"):
    try:
        from graphviz import Digraph
    except Exception as e:
        raise get_import_error("graphviz")
    g = Digraph(title)
    plot_hdl_recurse(layered_dict, g)
    return g
