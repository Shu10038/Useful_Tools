# -*- coding: utf-8 -*-
"""
calc_logistic_regression.py
正則化付きロジスティクス回帰

Created on Dec 12 2023

@author: S.O
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



class Calc_LogisticRegression:
    
    def __init__(self):
        
        self.c_values_list = np.logspace(-2, 4, 50) # 正則化パラメータCの範囲を設定する

    def _standard_scaler(self, data):
        """標準化の計算をする
        Args:
            data：入力データ

        Returns:
            標準化された値
        """

        scaler = StandardScaler()
        return scaler.fit_transform(data)
    

    def show_solution_path(self, train_x, train_y, standard_scaler_flag=True):

        feature_names = list(train_x.columns.values)

        if standard_scaler_flag:
            train_x = self._standard_scaler(train_x)
            
        train_y = train_y.values.ravel()  # 列ベクトル形式のため警告が出るのでyを1次元配列に変換

        coefficients_list = []

        for C in self.c_values_list:
            lr = LogisticRegression(penalty='l1', 
                                    C=C, 
                                    solver='saga', 
                                    max_iter=10000)
            lr.fit(train_x, train_y)
            coefficients_list.append(lr.coef_[0])

        self.coefficients = np.array(coefficients_list) # とりあえず、計算結果をSelfで持たせておく
        plt.figure(figsize=(10, 6))

        
        for i in range(self.coefficients.shape[1]):
            plt.plot(self.c_values_list, self.coefficients[:, i], label=feature_names[i])

        plt.xscale('log')
        plt.xlabel('C (Inverse of regularization strength)')
        plt.ylabel('Coefficients')
        plt.title('Path of the L1-regularized coefficients')
        plt.legend()
        plt.grid()
        plt.show()
