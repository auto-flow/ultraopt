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

    @property
    def n_categorical(self):
        return len(self.cat_cols)

    @property
    def n_numerical(self):
        return len(self.cont_cols)

    @property
    def n_ordinal(self):
        return len(self.ord_cols)

    def fit(self, config_space: ConfigurationSpace):
        mask = []
        n_constants = 0
        n_variables = 0
        n_top_levels = 0
        parents = []
        parent_values = []
        is_child = []
        n_variables_embedded_list = []
        cont_cols = []
        ord_cols = []
        cat_cols = []
        n_sequences_list = []
        n_choices_list = []
        low_r_cols = []
        hp_names = []
        # todo: 划分parents与groups
        for hp in config_space.get_hyperparameters():
            hp_name = hp.name
            # ====常量处理逻辑====
            if isinstance(hp, Constant) or \
                    (isinstance(hp, CategoricalHyperparameter) and len(hp.choices) == 1) or \
                    (isinstance(hp, OrdinalHyperparameter) and len(hp.sequence) == 1):
                mask.append(False)
                n_constants += 1
                n_variables_embedded_list.append(0)
                continue
            # ====变量处理逻辑=====
            hp_names.append(hp_name)
            mask.append(True)
            n_variables += 1
            # 处理 n_variables_embedded
            if isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
                choices = hp.choices if isinstance(hp, CategoricalHyperparameter) else hp.sequence
                n_choices = len(choices)
                # 确定n_embeds
                # 对于预训练的dfmap，顺便将其转化为array
                # 根据对类别变量的处理方式，处理bounds_list
                if n_choices == 2:
                    n_embeds = 2
                    cont_cols.append(hp_name)
                    low_r_cols.append(hp_name)
                else:
                    if hp_name in self.pretrained_emb:
                        df: pd.DataFrame = self.pretrained_emb[hp_name]
                        assert set(df.index) == set(choices), ValueError
                        n_embeds = df.shape[1]
                        df = df.loc[pd.Series(choices), :]
                        self.pretrained_emb[hp_name] = df.values
                    else:
                        n_embeds = int(get_embed_dims(n_choices))
                n_variables_embedded_list.append(n_embeds)
            else:
                n_variables_embedded_list.append(1)
            # 连续变量
            if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
                cont_cols.append(hp_name)
            # 类别变量
            if isinstance(hp, CategoricalHyperparameter):
                n_choices_list.append(len(hp.choices))
                if n_choices_list[-1] > 2:
                    cat_cols.append(hp_name)
            else:
                n_choices_list.append(0)
            # 有序变量
            if isinstance(hp, OrdinalHyperparameter):
                n_sequences_list.append(len(hp.sequence))
                if n_sequences_list[-1] > 2:
                    ord_cols.append(hp_name)
            else:
                n_sequences_list.append(0)
            # 条件变量
            cur_parents = config_space.get_parents_of(hp_name)
            if len(cur_parents) == 0:
                n_top_levels += 1
                parents.append(None)
                parent_values.append(None)
                is_child.append(False)
            else:
                is_child.append(True)
                parents.append(cur_parents[0])
                parent_conditions = config_space.get_parent_conditions_of(hp_name)
                parent_condition = parent_conditions[0]
                parent_values.append(parent_condition.value)
        self.origin_ord_cols = ord_cols[:]
        if self.consider_ord_as_cont:
            cont_cols += ord_cols
            ord_cols = []
        # assert len(bounds_list)==n_variables_embedded
        groups_str = [f"{parent}-{parent_value}" for parent, parent_value in zip(parents, parent_values)]
        group_encoder = LabelEncoder()
        groups = group_encoder.fit_transform(groups_str)
        self.is_child = is_child
        self.n_variables_embedded_list = n_variables_embedded_list
        self.n_variables_embedded = sum(n_variables_embedded_list)
        self.config_space = config_space
        self.groups_str = groups_str
        self.group_encoder = group_encoder
        self.groups = groups
        self.n_groups = np.max(groups) + 1
        self.mask = np.array(mask, dtype="bool")
        self.n_constants = n_constants
        self.n_variables = n_variables
        self.n_top_levels = n_top_levels
        self.ord_cols = ord_cols
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.low_r_cols = low_r_cols
        self.n_choices_list = n_choices_list
        self.n_sequences_list = n_sequences_list
        self.hp_names = pd.Series(hp_names)
        self.cont_mask = self.hp_names.isin(self.cont_cols)
        # fixme: 可能用不到NN
        if len(self.cat_cols) == 0 and len(self.ord_cols) == 0 and (not self.scale_cont_var):
            self.encoder = None

        # fixme: 这里开始设置encoder的信息
        if self.encoder is not None:
            self.encoder: EmbeddingEncoder
            self.encoder.cat_cols = copy(self.cat_cols)
            self.encoder.ord_cols = copy(self.ord_cols)
            self.encoder.cont_cols = copy(self.cont_cols)
            self.encoder.n_choices_list = [x for x in self.n_choices_list if x > 2]
            self.encoder.n_sequences_list = [x for x in self.n_sequences_list if x > 2]
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
                df_map = {}
                for hp_name, matrix in zip(self.encoder.cat_cols, self.encoder.cat_matrix_list):
                    choices = self.config_space.get_hyperparameter(hp_name).choices
                    df = pd.DataFrame(matrix, index=choices)
                    df_map[hp_name] = df
                for hp_name, matrix in zip(self.encoder.ord_cols, self.encoder.ord_matrix_list):
                    sequence = self.config_space.get_hyperparameter(hp_name).sequence
                    df = pd.DataFrame(matrix, index=sequence)
                    df_map[hp_name] = df
                self.embedding_encoder_history.append([
                    self.encoder.stage,
                    df_map
                ])

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        # 1. 根据掩码删除常数项（保留变量项）
        vectors = np.array(vectors)
        vectors = vectors[:, self.mask]
        # 2.
        if self.encoder is not None and self.scale_cont_var:
            mean = self.encoder.continuous_variables_mean
            weight = self.encoder.continuous_variables_weight
            N = vectors.shape[0]
            mean = np.tile(mean[np.newaxis, :], [N, 1])
            weight = np.tile(weight[np.newaxis, :], [N, 1])
            vectors[:, self.cont_mask] = (vectors[:, self.cont_mask] - mean) * weight
        #
        if self.consider_ord_as_cont:
            for i in range(vectors.shape[1]):
                if self.n_sequences_list[i] >= 2:
                    L = self.n_sequences_list[i]
                    vectors[:, i] = (vectors[:, i] / (L - 1)) * (_pow_scale(L - 1))
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
            is_ordinal = hp_name in self.origin_ord_cols
            if hp_name in self.cont_cols and (not is_ordinal):
                array[:, i] = np.clip(array[:, i], 0, 1)
            # # 2 反归一化ordinal变量
            if is_ordinal and self.consider_ord_as_cont:
                L = self.n_sequences_list[i]
                if L >= 2:
                    array[:, i] = np.clip(
                        np.round((array[:, i] / _pow_scale(L - 1)) * (L - 1)),
                        # np.round(array[:, i]),
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
