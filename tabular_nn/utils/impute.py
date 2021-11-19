#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : qichun.tang@bupt.edu.cn
from collections import Counter

import category_encoders.utils as util
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.core.common import SettingWithCopyWarning
from sklearn.utils._testing import ignore_warnings


class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            strategy="most_frequent",
            fill_value="<NULL>",
            numeric_fill_value=-1
    ):
        self.numeric_fill_value = numeric_fill_value
        self.strategy = strategy
        self.fill_value = fill_value
        assert strategy in ("most_frequent", "constant"), ValueError(f"Invalid strategy {strategy}")

    def fit(self, X, y=None):
        X = util.convert_input(X)
        self.columns = X.columns
        self.statistics_ = np.array([self.fill_value] * len(self.columns), dtype='object')
        numeric_mask = np.array([is_numeric_dtype(dtype) for dtype in X.dtypes])
        self.statistics_[numeric_mask] = self.numeric_fill_value
        if self.strategy == "most_frequent":
            for i, column in enumerate(X.columns):
                for value, counts in Counter(X[column]).most_common():
                    if not pd.isna(value):
                        self.statistics_[i] = value
                        break
        return self

    @ignore_warnings(category=SettingWithCopyWarning)
    def transform(self, X):
        # note: change inplace
        for i, (column, dtype) in enumerate(zip(X.columns, X.dtypes)):
            value = self.statistics_[i]
            mask = pd.isna(X[column]).values
            if np.count_nonzero(mask) == 0:
                continue
            if dtype.name == "category" and value not in X[column].cat.categories:
                X[column].cat.add_categories(value, inplace=True)
            X.loc[mask, column] = value
        return X