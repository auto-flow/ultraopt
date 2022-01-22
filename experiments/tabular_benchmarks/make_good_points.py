#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-12-03
# @Contact    : qichun.tang@bupt.edu.cn
import ConfigSpace
from joblib import dump
cs = ConfigSpace.ConfigurationSpace()

cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("n_units_1", [16, 32, 64, 128, 256, 512]))
cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("n_units_2", [16, 32, 64, 128, 256, 512]))
cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("dropout_1", [0.0, 0.3, 0.6]))
cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("dropout_2", [0.0, 0.3, 0.6]))
cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_fn_1", ["tanh", "relu"]))
cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("activation_fn_2", ["tanh", "relu"]))
cs.add_hyperparameter(
    ConfigSpace.OrdinalHyperparameter("init_lr", [5 * 1e-4, 1e-3, 5 * 1e-3, 1e-2, 5 * 1e-2, 1e-1]))
cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("lr_schedule", ["cosine", "const"]))
cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter("batch_size", [8, 16, 32, 64]))

slice = ConfigSpace.Configuration(
    cs,
    values={
        'init_lr': 0.0005,
        'batch_size': 32,
        'lr_schedule': 'cosine',
        'activation_fn_1': 'relu',
        'activation_fn_2': 'tanh',
        'n_units_1': 512,
        'n_units_2': 512,
        'dropout_1': 0,
        'dropout_2': 0,
    }
)
naval = ConfigSpace.Configuration(
    cs,
    values={
        'init_lr': 0.0005,
        'batch_size': 8,
        'lr_schedule': 'cosine',
        'activation_fn_1': 'tanh',
        'activation_fn_2': 'relu',
        'n_units_1': 128,
        'n_units_2': 512,
        'dropout_1': 0,
        'dropout_2': 0,
    }
)
parkinson = ConfigSpace.Configuration(
    cs,
    values={
        'init_lr': 0.0005,
        'batch_size': 8,
        'lr_schedule': 'cosine',
        'activation_fn_1': 'tanh',
        'activation_fn_2': 'relu',
        'n_units_1': 128,
        'n_units_2': 512,
        'dropout_1': 0,
        'dropout_2': 0,
    }
)

good_configs=[
    slice,
    slice,
    naval,parkinson]
dump(good_configs, 'store/good_configs.pkl')
