#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2020-12-14
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np
from sklearn.utils import check_array

ERR_LOSS = 1 << 31


class LossTransformer():
    def fit_transform(self, y, *args):
        y: np.ndarray = check_array(y, ensure_2d=False)
        # cutoff
        # fixme: all of y < ERR_LOSS
        y[y >= ERR_LOSS] = y[y < ERR_LOSS].max() + 0.1
        self.y_max = y.max()
        self.y_min = y.min()
        self.y_mean = y.mean()
        self.y_std = y.std()
        self.perc = np.percentile(y, 5)
        return y


class ScaledLossTransformer(LossTransformer):
    def fit_transform(self, y, *args):
        y = super(ScaledLossTransformer, self).fit_transform(y)
        # Subtract the difference between the percentile and the minimum
        y_min = self.y_min - (self.perc - self.y_min)
        # linear scaling
        if self.y_min == self.y_max:
            # prevent diving by zero
            y_min *= 1 - 10 ** -101
        y = (y - y_min) / (self.y_max - self.y_min)
        return y


class LogScaledLossTransformer(LossTransformer):
    def fit_transform(self, y, *args):
        y = super(LogScaledLossTransformer, self).fit_transform(y)
        # Subtract the difference between the percentile and the minimum
        y_min = self.y_min - (self.perc - self.y_min)
        y_min -= 1e-10
        # linear scaling
        if y_min == self.y_max:
            # prevent diving by zero
            y_min *= 1 - (1e-10)
        y = (y - y_min) / (self.y_max - y_min)
        y = np.log(y)
        f_max = y[np.isfinite(y)].max()
        f_min = y[np.isfinite(y)].min()
        y[np.isnan(y)] = f_max
        y[y == -np.inf] = f_min
        y[y == np.inf] = f_max
        return y
