#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : qichun.tang@bupt.edu.cn
from copy import copy
from typing import Optional, Union

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Constant, CategoricalHyperparameter, Configuration, OrdinalHyperparameter
from ConfigSpace.util import deactivate_inactive_hyperparameters
from sklearn.preprocessing import LabelEncoder
from tabular_nn.base_tnn import get_embed_dims

from ultraopt.utils.config_space import deactivate


class ConfigTransformer():
    def __init__(self, impute: Optional[float] = -1, encoder=None):
        self.impute = impute
        self.encoder = encoder

    def fit(self, config_space: ConfigurationSpace):
        mask = []
        n_choices_list = []
        is_ordinal_list = []
        sequence_mapper = {}
        n_constants = 0
        n_variables = 0
        n_variables_embedded = 0
        n_top_levels = 0
        parents = []
        parent_values = []
        is_child = []
        # todo: 划分parents与groups
        for hp in config_space.get_hyperparameters():
            if isinstance(hp, Constant) or \
                    (isinstance(hp, CategoricalHyperparameter) and len(hp.choices) == 1) or \
                    (isinstance(hp, OrdinalHyperparameter) and len(hp.sequence) == 1):
                # ignore
                mask.append(False)
                n_constants += 1
            else:
                mask.append(True)
                n_variables += 1
                if isinstance(hp, CategoricalHyperparameter):
                    n_choices = len(hp.choices)
                    n_choices_list.append(n_choices)
                    n_variables_embedded += int(get_embed_dims(n_choices)) # avoid bug
                else:
                    n_choices_list.append(0)
                    n_variables_embedded += 1
                if isinstance(hp, OrdinalHyperparameter):
                    is_ordinal_list.append(True)
                    sequence_mapper[len(is_ordinal_list) - 1] = hp.sequence
                else:
                    is_ordinal_list.append(False)
                cur_parents = config_space.get_parents_of(hp.name)
                if len(cur_parents) == 0:
                    n_top_levels += 1
                    parents.append(None)
                    parent_values.append(None)
                    is_child.append(False)
                else:
                    is_child.append(True)
                    parents.append(cur_parents[0])
                    parent_conditions = config_space.get_parent_conditions_of(hp.name)
                    parent_condition = parent_conditions[0]
                    parent_values.append(parent_condition.value)
        groups_str = [f"{parent}-{parent_value}" for parent, parent_value in zip(parents, parent_values)]
        group_encoder = LabelEncoder()
        groups = group_encoder.fit_transform(groups_str)
        self.is_child = is_child
        self.sequence_mapper = sequence_mapper
        self.is_ordinal_list = is_ordinal_list
        self.config_space = config_space
        self.groups_str = groups_str
        self.group_encoder = group_encoder
        self.groups = groups
        self.n_groups = np.max(groups) + 1
        self.mask = np.array(mask, dtype="bool")
        self.n_choices_list = n_choices_list
        self.n_constants = n_constants
        self.n_variables = n_variables
        self.n_variables_embedded = n_variables_embedded
        self.n_top_levels = n_top_levels
        self.hp_names = pd.Series([hp.name for hp in config_space.get_hyperparameters()])[self.mask]
        high_r_mask = np.array(self.n_choices_list) > 2
        self.high_r_cols = self.hp_names[high_r_mask].to_list()
        self.high_r_cats = []
        for ix in np.arange(n_variables)[high_r_mask]:
            n_choices = n_choices_list[ix]
            cat = list(range(n_choices))
            if is_child[ix]:
                cat.insert(0, -1)
            self.high_r_cats.append(cat)
        if self.encoder is not None:
            self.encoder.cols = copy(self.high_r_cols)
            self.encoder.categories = copy(self.high_r_cats)
        return self

    def fit_encoder(self, vectors, losses=None):
        vectors = vectors[:, self.mask]
        df = pd.DataFrame(vectors, columns=self.hp_names)
        if self.encoder is not None:
            self.encoder.fit(df, losses)

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.array(vectors)
        vectors = vectors[:, self.mask]
        if self.encoder is not None:
            df = pd.DataFrame(vectors, columns=self.hp_names)
            vectors = self.encoder.transform(df)
            if not isinstance(vectors, np.ndarray):
                vectors = np.array(vectors)
        if self.impute is not None:
            if self.impute == "random_choice":
                vectors = self.impute_conditional_data(vectors)
            else:  # is numeric
                vectors[np.isnan(vectors)] = float(self.impute)
        return vectors

    def inverse_transform(self, array: np.ndarray, return_vector=False) -> Union[np.ndarray, None, Configuration]:
        if self.encoder is not None:
            array = self.encoder.inverse_transform(array)
        array = np.array(array)
        for i, n_choices in enumerate(self.n_choices_list):
            if n_choices == 2:
                array[:, i] = (array[:, i] > 0.5).astype("float64")
            is_ordinal = self.is_ordinal_list[i]
            if is_ordinal:
                sequence = self.sequence_mapper[i]
                array[:, i] = np.clip(np.round(array[:, i]), 0, len(sequence) - 1)
        N, M = array.shape
        result = np.zeros([N, len(self.mask)])
        result[:, self.mask] = array
        if return_vector:
            return result
        configs = []
        for i in range(N):
            try:
                config = deactivate(self.config_space, result[i, :])
                config = deactivate_inactive_hyperparameters(
                    configuration_space=self.config_space,
                    configuration=config
                )
                configs.append(config)
            except Exception as e:
                pass
        return configs

    def impute_conditional_data(self, array):
        # copy from HpBandSter
        return_array = np.empty_like(array)

        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).flatten()

            while (np.any(nan_indices)):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(array[:, nan_idx])).flatten()

                if len(valid_indices) > 0:
                    # pick one of them at random and overwrite all NaN values
                    row_idx = np.random.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]

                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = self.n_choices_list[nan_idx]
                    if t == 0:
                        datum[nan_idx] = np.random.rand()
                    else:
                        datum[nan_idx] = np.random.randint(t)
                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i, :] = datum
        return (return_array)
