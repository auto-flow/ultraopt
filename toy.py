#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-11-12
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from category_encoders import CatBoostEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

MLPClassifier()

from wrap_lightgbm import LGBMClassifier

df = pd.read_csv('titanic_train.csv')
df.pop('Name')
df.pop('Ticket')
df.pop('PassengerId')
label = df.pop('Survived')
print(df)
obj_col = df.select_dtypes(include='object').columns
num_col = df.select_dtypes(exclude='object').columns
df[num_col] = SimpleImputer(strategy='median').fit_transform(df[num_col])
df[obj_col] = SimpleImputer(strategy='constant', fill_value="<NULL>").fit_transform(df[obj_col])
df[obj_col] = OrdinalEncoder().fit_transform(df[obj_col])
ohe_col = ['Sex', 'Embarked']
hr_col = ['Cabin']
num_df = df.copy()
cat_encoder = CatBoostEncoder(cols=hr_col)
ohe_encoder = OneHotEncoder(cols=ohe_col)
df = cat_encoder.fit_transform(df, label)
df = ohe_encoder.fit_transform(df, label)

C = {"_type": "loguniform", "_value": [1, 10]}
lgbm_kwargs = {
    "num_leaves": {"_type": "int_quniform", "_value": [10, 150], "_default": 31},
    "max_depth": {"_type": "int_quniform", "_value": [1, 100], "_default": 31},
    "learning_rate": 0.1,  # {"_type": "loguniform", "_value": [1e-2, 0.2], "_default": 0.1},
    "subsample_for_bin": {"_type": "int_quniform", "_value": [2e4, 3e5, 2e4], "_default": 40000},
    "feature_fraction": {"_type": "quniform", "_value": [0.5, 0.95, 0.05], "_default": 0.6},
    "bagging_fraction": {"_type": "quniform", "_value": [0.5, 0.95, 0.05], "_default": 0.6},
    # alias "subsample"
    "lambda_l1": {"_type": "loguniform", "_value": [1e-7, 10], "_default": 0},  # reg_alpha
    "lambda_l2": {"_type": "loguniform", "_value": [1e-7, 10], "_default": 0},  # reg_lambda
    "min_child_weight": {"_type": "loguniform", "_value": [1e-7, 10], "_default": 1e-3},
    # aliases to min_sum_hessian
}
lgbm_rf_kwargs = lgbm_kwargs.copy()
lgbm_rf_kwargs.pop('learning_rate')
xgb_kwargs = {
    "learning_rate": 0.1,  # {"_type": "loguniform", "_value": [1e-2, 0.1], "_default": 0.1},
    "max_depth": {"_type": "int_uniform", "_value": [10, 90]},
    "colsample_bytree": {"_type": "quniform", "_value": [0.5, 0.95, 0.05]},
    "reg_lambda": {"_type": "loguniform", "_value": [1e-3, 1]}
}

catboost_kwargs = {
    "n_estimators": {"_type": "int_uniform", "_value": [10, 200]},
}

rf_kwargs = {
    "criterion": {"_type": "choice", "_value": ["gini", "entropy"], "_default": "gini"},
    "max_features": {"_type": "choice", "_value": ["sqrt", "log2"], "_default": "sqrt"},
    # //      "max_features": {"_type": "uniform", "_value": [0.01, 1.0],"_default": 1},
    "max_depth": {"_type": "int_uniform", "_value": [5, 100], "_default": 30},
    "min_samples_split": {"_type": "int_uniform", "_value": [2, 20], "_default": 2},
    "min_samples_leaf": {"_type": "int_uniform", "_value": [1, 20], "_default": 1},
}

mlp_kwargs = {
    "hidden_layer_size": {"_type": "int_quniform", "_value": [50, 150, 10], "_default": 100},
}


def softmax(df):
    if len(df.shape) == 1:
        df[df > 20] = 20
        df[df < -20] = -20
        ppositive = 1 / (1 + np.exp(-df))
        ppositive[ppositive > 0.999999] = 1
        ppositive[ppositive < 0.0000001] = 0
        return np.transpose(np.array((1 - ppositive, ppositive)))
    else:
        # Compute the Softmax like it is described here:
        # http://www.iro.umontreal.ca/~bengioy/dlbook/numerical.html
        tmp = df - np.max(df, axis=1).reshape((-1, 1))
        tmp = np.exp(tmp)
        return tmp / np.sum(tmp, axis=1).reshape((-1, 1))


HDL = {
    "model(choice)": {
        # "LR-std-l1-liblinear": {"C": C},
        # "LR-std-l2-lbfgs": {"C": C},
        # "LR-std-l2-liblinear": {"C": C},
        # "LR-std-l2-saga": {"C": C},
        # "LR-std-l2": {"C": C},
        # "LR-minmax-l2-lbfgs": {"C": C},
        # "LR-minmax-l2-liblinear": {"C": C},
        # "LR-minmax-l2-saga": {"C": C},
        # "LR-minmax-l2": {"C": C},
        "LR-minmax-l1": {"C": C},
        "LR-std-l1": {"C": C},
        # "LSVM-std-l1": {"C": C},
        # "LSVM-std-l2": {"C": C},
        # "LR-minmax-l2": {"C": C},
        # "LSVM-minmax-l1": {"C": C},
        # "LSVM-minmax-l2": {"C": C},
        # "MLP-adam": dict(**mlp_kwargs),
        # "MLP-sgd": dict(**mlp_kwargs),
        # "KNN-l1": {"n_neighbors": {"_type": "int_quniform", "_value": [3, 7, 1]}},
        "KNN-l2": {"n_neighbors": {"_type": "int_quniform", "_value": [3, 7, 1]}},
        "LGBM-gbdt": dict(**lgbm_kwargs),
        "LGBM-dart": dict(**lgbm_kwargs),
        "LGBM-rf": dict(**lgbm_rf_kwargs),
        "XGBoost": dict(**xgb_kwargs),
        "CatBoost": dict(**catboost_kwargs),
        "RF-bag": dict(**rf_kwargs),
        "RF-nobag": dict(**rf_kwargs),
        "ET-bag": dict(**rf_kwargs),
        "ET-nobag": dict(**rf_kwargs),
    }
}

from ultraopt.hdl import layering_config


class Evaluator():
    def __init__(self):
        pass

    def __call__(self, config):
        config = layering_config(config)
        AS_HP = config['model']
        AS, HP = AS_HP.popitem()
        HP['random_state'] = 0
        scaler = None
        linear_model = False
        model_name = None
        if AS.startswith('LR') or AS.startswith('LSVM'):
            # model_name, scale_name, reg_method, solver = AS.split('-')
            # HP['solver'] = solver
            model_name, scale_name, reg_method = AS.split('-')
            HP['penalty'] = reg_method
            cls = LinearSVC if model_name == "LSVM" else LogisticRegression
            scaler = MinMaxScaler() if scaler == "minmax" else StandardScaler()
            if model_name == "LR":
                if reg_method == "l1":
                    HP.update(dict(
                        solver='liblinear',
                    ))
            if model_name == "LSVM":
                HP.update(dict(
                    dual=False,
                ))
            linear_model = True
        elif AS.startswith('MLP'):
            # active_func = AS.split('-')[-1]
            cls = MLPClassifier
            scaler = StandardScaler()
            linear_model = True
            HP['hidden_layer_sizes'] = (HP.pop('hidden_layer_size'),)
            HP['solver'] = AS.split('-')[-1]
            HP['max_iter'] = 100
        elif AS.startswith("KNN"):
            p = AS.split('-')[-1]
            cls = KNeighborsClassifier
            scaler = MinMaxScaler()
            linear_model = True
            HP.pop('random_state')
            HP['p'] = int(p[-1])
        elif AS.startswith('LGBM'):
            boosting_type = AS.split('-')[-1]
            HP.update(boosting_type=boosting_type, n_estimators=100)
            cls = LGBMClassifier
        elif AS == "XGBoost":
            cls = XGBClassifier
            HP.update(n_estimators=100)
        elif AS == "CatBoost":
            cls = CatBoostClassifier
        elif AS.startswith('RF') or AS.startswith('ET'):
            model_name, bag_type = AS.split('-')
            cls = RandomForestClassifier if model_name == "RF" else ExtraTreesClassifier
            if bag_type == 'bag':
                HP['bootstrap'] = True
            else:
                HP['bootstrap'] = False
            HP['n_estimators'] = 100
        else:
            raise NotImplementedError
        model = cls(**HP)
        if linear_model:
            X = num_df.copy().values
        else:
            X = df.copy().values
        if scaler:
            X = scaler.fit_transform(X)
        y = label.values
        splitter = StratifiedShuffleSplit(n_splits=1, random_state=0)
        train_ix, test_ix = next(splitter.split(X, y))
        X_train = X[train_ix, :]
        y_train = y[train_ix]
        X_test = X[test_ix, :]
        y_test = y[test_ix]
        model.fit(X_train, y_train)
        if model_name == "LSVM":
            y_prob = softmax(model.decision_function(X_test))
        else:
            y_prob = model.predict_proba(X_test)
        log_loss_score = log_loss(y_test, y_prob)
        y_pred = np.argmax(y_prob, axis=1)
        f1 = f1_score(y_test, y_pred)
        return 1 - f1


evaluator = Evaluator()
from ultraopt import fmin
from ultraopt.optimizer import ETPEOptimizer
import os

os.environ['MAX_DIM'] = '3'
opt = ETPEOptimizer(min_points_in_model=20)

ret = fmin(evaluator, HDL, opt, n_iterations=100)
df_pair = ret.optimizer.config_transformer.embedding_encoder_history[-1][1]
df_emb = df_pair[0]
names = list(df_emb.index)
from sklearn.manifold import TSNE
import pylab as plt
plt.rcParams['figure.figsize'] = (8, 6)
from sklearn.decomposition import PCA
if df_emb.shape[1] > 2:
    X_emb = PCA(n_components=2,random_state=0,whiten=True).fit_transform(df_emb)

    # X_emb = TSNE().fit_transform(df_emb)
else:
    X_emb = df_emb.values
for i, name in enumerate(names):
    x, y = X_emb[i, :]
    plt.scatter(x, y)
    plt.annotate(name, (x, y))
plt.xlim(-2.5,3)
plt.ylim(-2.5,3)
plt.show()
df_emb.to_csv('emb_ans.csv')
# plt.legend()
print(ret)
