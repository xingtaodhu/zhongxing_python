# -*- coding: utf-8 -*-
import random

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

def lightgbm():
    print("start")

    dataset = np.loadtxt("/Users/xingtao/PycharmProjects/zhongxing/train1.csv", delimiter=',')
    # df = pd.DataFrame(dataset)
    # df[df.columns[-1]] = df[df.columns[-1]].shift(-1)
    # dataset = df.values

    label = dataset[:, -1]

    # print(label.shape)
    feature = dataset[:, :-1]
    # print(feature.shape)

    # label[len(label)-1] = 1

    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=0)
    return lightgbm_c(X_train, X_test, y_train, y_test)

def lightgbm_c(X_train, X_test, y_train, y_test):
    param_grid = {
        'learning_rate': [0.001],
        'n_estimators': [30,40]
    }
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'regression', # 目标函数
        'metric': {'l2', 'auc'},  # 评估函数
        'num_leaves': 31,   # 叶子节点数
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.9, # 建树的特征选择比例
        'bagging_fraction': 0.8, # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    # gbm = lgb.LGBMRegressor(objective='regression',num_leaves=31,learning_rate=0.05,n_estimators=20)
    # gbm.fit(X_train, y_train,eval_set=[(X_test, y_test)],eval_metric='l1',early_stopping_rounds=5)
    #
    # print('Start predicting...')
    # # 测试机预测
    # y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    # # 模型评估
    # print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    #
    # # feature importances
    # print('Feature importances:', list(gbm.feature_importances_))
    #
    # # 网格搜索，参数优化
    # estimator = lgb.LGBMRegressor(num_leaves=31)
    #
    # param_grid = {
    #     'learning_rate': [0.01, 0.1, 1],
    #     'n_estimators': [20, 40]
    # }
    #
    # gbm = GridSearchCV(estimator, param_grid)
    #
    # gbm.fit(X_train, y_train)
    #
    # print('Best parameters found by grid search are:', gbm.best_params_)


    lightgbm  = lgb.sklearn.LGBMClassifier()

    lightgbm = GridSearchCV(lightgbm, param_grid)
    lightgbm.fit(X_train, y_train)
    y_lgb = lightgbm.predict(X_test)

    #print(accuracy_score(y_test, y_lgb))
    print(precision_score(y_test, y_lgb,average='weighted',zero_division=0))
    print(recall_score(y_test, y_lgb,average='weighted'))
    return precision_score(y_test, y_lgb,average='weighted',zero_division=0),recall_score(y_test, y_lgb,average='weighted')


if __name__ == "__main__":
   lightgbm()