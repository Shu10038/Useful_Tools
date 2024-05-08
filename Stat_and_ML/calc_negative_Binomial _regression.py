# -*- coding: utf-8 -*-
"""
calc_negative_Binomial _regression.py

Created on Dec 12 2023

@author: S.O

"""
import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class calcPoissonReg:
    
    def __init__(self):
        # あとで拡張子しやすいようにあえて残しておく
        a = 1

    def _standard_scaler(self, data):
        """標準化の計算をする
        Args:
            data：入力データ

        Returns:
            標準化された値
        """

        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def show_coef_val(self,
                train_x,
                train_y,
                standard_scaler_flag=True
                ):

        feature_names = list(train_x.columns.values)

        if standard_scaler_flag:
            train_x = self._standard_scaler(train_x)
        train_y = train_y.values.ravel()
        
        self.model = PoissonRegressor(alpha=1.0, max_iter=1000)
        self.model.fit(train_x, train_y)

        coef_dict = {'weights':self.model.coef_,
                'abs_weights':[abs(i) for i in self.model.coef_],
                }
        # ソートが面倒なのでデータフレームにしてから全部を並べ替える。他に良い方法を思いついたら後で修正する
        dff = pd.DataFrame(coef_dict).T.set_axis(feature_names, axis='columns').sort_values('abs_weights', axis=1, ascending=True) 

        plt.barh(range(len(dff.columns.values)),dff.loc['abs_weights'], tick_label=dff.columns.values)
        plt.show()

        # 棒グラフにするとなぜか昇順・降順が逆転するので、一度昇順にしてから降順でRetrun
        return dff.sort_values('abs_weights', axis=1, ascending=False)
    
    def eval_model(self, train_x, train_y):
        pred = self.model.predict(train_x).astype(float)
        #print(train_y,pred)
        y = train_y.to_numpy()
        # 対数尤度の計算
        log_factorial_y = np.array([np.sum(np.log(np.arange(1, i+1))) for i in y])  # log(y_i!) を計算
        log_likelihood = np.sum(y * np.log(pred) - pred - log_factorial_y)

        # パラメータ数（係数 + 切片）
        p = train_x.shape[1] + 1

        # AICの計算
        val_AIC = 2 * p - 2 * log_likelihood

        print("Log-Likelihood:", log_likelihood)
        print("AIC:", val_AIC)

        return log_likelihood, val_AIC
        

    def cross_validation(self,
                train_x,
                train_y,
                kernel='linear',
                standard_scaler_flag=True
                ):

        feature_names = list(train_x.columns.values)

        if standard_scaler_flag:
            train_x = self._standard_scaler(train_x)
        train_y = train_y.values.ravel()
        
        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, random_state=0)

        svm_clf = svm.SVC(kernel=kernel)
        svm_clf.fit(X_train, y_train)
        # test set を用いて評価
        score = svm_clf.score(X_test, y_test)
        print('score: {}'.format(score))

