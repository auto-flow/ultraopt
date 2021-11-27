#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : qichun.tang@bupt.edu.cn
from copy import copy
from typing import Optional, Union, List, Dict

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace, Constant, CategoricalHyperparameter, \
    Configuration, OrdinalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.util import deactivate_inactive_hyperparameters
from sklearn.preprocessing import LabelEncoder
from tabular_nn import EmbeddingEncoder
from tabular_nn.base_tnn import get_embed_dims
from ultraopt.utils.config_space import deactivate

_pow_scale = lambda x: np.float_power(x, 0.5)


class ConfigTransformer():
    def __init__(
            self, impute: Optional[float] = -1, encoder=None, pretrained_emb=None,
            consider_ord_as_cont=True, scale_cont_var=True
    ):
        self.scale_cont_var = scale_cont_var
        self.consider_ord_as_cont = consider_ord_as_cont
        self.pretrained_emb: Dict[str, pd.DataFrame] = pretrained_emb if pretrained_emb else {}
        self.impute = impute
        self.encoder: Union[EmbeddingEncoder, None] = encoder

    def fit(self, config_space: ConfigurationSpace):
        mask = []
        n_choices_list = []
        is_ordinal_list = []
        sequence_mapper = {}
        n_constants = 0
        n_variables = 0
        n_categorical = 0
        n_numerical = 0
        n_ordinal = 0
        n_variables_embedded = 0  # 用于统计Embedding后变量总数（离散转连续）
        n_top_levels = 0
        parents = []
        parent_values = []
        is_child = []
        n_variables_embedded_list = []
        bounds_list = []  # 用于对边界效应纠偏
        cont_cols = []
        ord_cols = []
        low_r_cols = []
        # todo: 划分parents与groups
        for hp in config_space.get_hyperparameters():
            hp_name = hp.name
            if isinstance(hp, Constant) or \
                    (isinstance(hp, CategoricalHyperparameter) and len(hp.choices) == 1) or \
                    (isinstance(hp, OrdinalHyperparameter) and len(hp.sequence) == 1):
                # ignore
                mask.append(False)
                n_constants += 1
                n_variables_embedded_list.append(0)
            else:
                mask.append(True)
                n_variables += 1
                if isinstance(hp, CategoricalHyperparameter):
                    n_choices = len(hp.choices)
                    n_choices_list.append(n_choices)
                    # 确定n_embeds
                    # 对于预训练的dfmap，顺便将其转化为array
                    # 根据对类别变量的处理方式，处理bounds_list
                    if n_choices == 2:
                        n_embeds = 2
                        bounds_list.append([-0.5, 1.5])  # 留出-q/2的缓冲
                        cont_cols.append(hp.name)
                        low_r_cols.append(hp.name)
                    else:
                        if hp_name in self.pretrained_emb:
                            df: pd.DataFrame = self.pretrained_emb[hp_name]
                            assert set(df.index) == set(hp.choices), ValueError
                            n_embeds = df.shape[1]
                            df = df.loc[pd.Series(hp.choices), :]
                            self.pretrained_emb[hp_name] = df.values
                        else:
                            n_embeds = int(get_embed_dims(n_choices))
                        # 离散转连续空间后，无法确定其边界，故不设置
                        for i in range(n_embeds):
                            bounds_list.append(None)
                    n_variables_embedded += n_embeds
                    n_variables_embedded_list.append(n_embeds)
                    n_categorical += 1
                else:
                    n_choices_list.append(0)
                    n_variables_embedded += 1
                    n_variables_embedded_list.append(1)
                if isinstance(hp, OrdinalHyperparameter):
                    if len(hp.sequence) == 2:
                        bounds_list.append([-0.5, 1.5])
                        cont_cols.append(hp.name)
                        low_r_cols.append(hp.name)
                    else:
                        n_ordinal += 1
                        is_ordinal_list.append(True)
                        sequence_mapper[len(is_ordinal_list) - 1] = hp.sequence
                        # todo: 用神经网络来学习边距
                        bounds_list.append(None)
                        ord_cols.append(hp.name)
                else:
                    is_ordinal_list.append(False)
                if not isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
                    n_numerical += 1
                # fixme: 对log的q进行测试
                if isinstance(hp, UniformFloatHyperparameter):
                    cont_cols.append(hp.name)
                    q = hp.q
                    if q is None:
                        bounds_list.append([0, 1])
                    else:
                        q = 1 / ((hp.upper - hp.lower) / q)
                        bounds_list.append([-q / 2, 1 + q / 2])
                if isinstance(hp, UniformIntegerHyperparameter):
                    cont_cols.append(hp.name)
                    q = hp.q
                    if q is None:
                        q = 1
                    q = 1 / ((hp.upper - hp.lower) / q)
                    bounds_list.append([-q / 2, 1 + q / 2])
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
        if self.consider_ord_as_cont:
            cont_cols += ord_cols
            ord_cols = []
        # assert len(bounds_list)==n_variables_embedded
        groups_str = [f"{parent}-{parent_value}" for parent, parent_value in zip(parents, parent_values)]
        group_encoder = LabelEncoder()
        groups = group_encoder.fit_transform(groups_str)
        self.n_categorical = n_categorical
        self.n_numerical = n_numerical
        self.n_ordinal = n_ordinal
        self.is_child = is_child
        self.n_variables_embedded_list = n_variables_embedded_list
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
        self.bounds_list = bounds_list
        self.n_top_levels = n_top_levels
        self.ord_cols = ord_cols
        self.cont_cols = cont_cols
        self.low_r_cols = low_r_cols
        self.hp_names = pd.Series([hp.name for hp in config_space.get_hyperparameters()])[self.mask]
        self.cont_mask = self.hp_names.isin(self.cont_cols)
        high_r_mask = np.array(self.n_choices_list) > 2
        self.high_r_cols = self.hp_names[high_r_mask].to_list()
        self.high_r_cats = []
        for i, ix in enumerate(np.arange(n_variables)[high_r_mask]):
            n_choices = n_choices_list[ix]
            cat = list(range(n_choices))
            if is_child[ix]:  # 处理缺失值
                cat.insert(0, -1)
            self.high_r_cats.append(cat)
            if self.high_r_cols[i] in self.pretrained_emb:
                continue
        # fixme: 可能用不到NN
        if len(self.high_r_cols) == 0 and (not self.scale_cont_var):
            self.encoder = None

        # fixme: 这里开始设置encoder的信息
        if self.encoder is not None:
            self.encoder: EmbeddingEncoder
            self.encoder.cat_cols = copy(self.high_r_cols)
            self.encoder.cont_cols = copy(self.cont_cols)
            self.encoder.categories = copy(self.high_r_cats)
            self.encoder.pretrained_emb = self.pretrained_emb
        self.embedding_encoder_history = []
        return self

    def fit_encoder(self, vectors, label=None):
        vectors = vectors[:, self.mask]
        df = pd.DataFrame(vectors, columns=self.hp_names)
        if self.encoder is not None:
            self.encoder.fit(df, label)
            if not isinstance(self.encoder, EmbeddingEncoder):
                return
            if not self.embedding_encoder_history or \
                    self.encoder.stage != self.embedding_encoder_history[-1][0]:
                if self.encoder.transform_matrix is not None:
                    df_map = {}
                    for hp_name, matrix in zip(self.encoder.cat_cols, self.encoder.transform_matrix):
                        choices = self.config_space.get_hyperparameter(hp_name).choices
                        df = pd.DataFrame(matrix, index=choices)
                        df_map[hp_name] = df
                    self.embedding_encoder_history.append([
                        self.encoder.stage,
                        df_map
                    ])

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        # 1. 根据掩码删除常数项（保留变量项）
        vectors = np.array(vectors)
        vectors = vectors[:, self.mask]
        # 2. 归一化ordinal变量
        # for idx, seq in self.sequence_mapper.items():
        #     L = len(seq)
        #     vectors[:, idx] = (vectors[:, idx] / (L - 1)) * (_pow_scale(L - 1))
        if self.encoder is not None and self.scale_cont_var:
            mean = self.encoder.continuous_variables_mean
            weight = self.encoder.continuous_variables_weight
            N = vectors.shape[0]
            mean = np.tile(mean[np.newaxis, :], [N, 1])
            weight = np.tile(weight[np.newaxis, :], [N, 1])
            vectors[:, self.cont_mask] = (vectors[:, self.cont_mask] - mean) * weight
        # 3. 编码离散变量
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

    def inverse_transform(self, array: np.ndarray, return_vector=False) -> \
            Union[List[np.ndarray], List[Configuration]]:
        # 3.1 处理离散变量
        if self.encoder is not None:
            array = self.encoder.inverse_transform(array)
        array = np.array(array)

        N, M = array.shape

        if self.encoder is not None and self.scale_cont_var:
            mean = self.encoder.continuous_variables_mean
            weight = self.encoder.continuous_variables_weight
            mean = np.tile(mean[np.newaxis, :], [N, 1])
            weight = np.tile(weight[np.newaxis, :], [N, 1])
            array[:, self.cont_mask] = (array[:, self.cont_mask] / weight) + mean

        for i, hp_name in enumerate(self.hp_names):
            # 3.2 对于只有2个项的离散变量，需要做特殊处理
            if hp_name in self.low_r_cols:
                array[:, i] = (array[:, i] > 0.5).astype("float64")
            is_ordinal = self.is_ordinal_list[i]
            if hp_name in self.cont_cols and (not is_ordinal):
                array[:, i] = np.clip(array[:, i], 0, 1)
            # # 2 反归一化ordinal变量
            if is_ordinal and self.consider_ord_as_cont:
                seq = self.sequence_mapper[i]
                L = len(seq)
                array[:, i] = np.clip(
                    # np.round((array[:, i]) * (L - 1)),
                    np.round(array[:, i]),
                    # np.round(array[:, i]),
                    0, L - 1)

        result = np.zeros([N, len(self.mask)])
        result[:, self.mask] = array
        if return_vector:
            return [result[i, :] for i in range(result.shape[0])]
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
