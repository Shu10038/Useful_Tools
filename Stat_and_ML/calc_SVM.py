# -*- coding: utf-8 -*-
"""

Created on Dec 12 2023
@author: S.O

"""
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

from sklearn.model_selection import train_test_split

class CalcSVM:
    
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
                kernel='linear',
                standard_scaler_flag=True
                ):

        feature_names = list(train_x.columns.values)

        if standard_scaler_flag:
            train_x = self._standard_scaler(train_x)
        train_y = train_y.values.ravel()
        
        self.svm_clf = svm.SVC(kernel=kernel)
        self.svm_clf.fit(train_x, train_y)

        coef_dict = {'weights':self.svm_clf.coef_[0],
                'abs_weights':[abs(i) for i in self.svm_clf.coef_[0]],
                'intercept':self.svm_clf.intercept_[0]}
        # ソートが面倒なのでデータフレームにしてから全部を並べ替える。他に良い方法を思いついたら後で修正する
        dff = pd.DataFrame(coef_dict).T.set_axis(feature_names, axis='columns').sort_values('abs_weights', axis=1, ascending=True) 

        plt.barh(range(len(dff.columns.values)),dff.loc['abs_weights'], tick_label=dff.columns.values)
        plt.show()

        # 棒グラフにするとなぜか昇順・降順が逆転するので、一度昇順にしてから降順でRetrun
        return dff.sort_values('abs_weights', axis=1, ascending=False)

        
    def plot_decision_boundary(self, train_x):
        # 境界線の表示
        X_embedded = TSNE(n_components=2, random_state=42).fit_transform(train_x)
        # 削減されたデータと元のデータの対応付け
        X_original_indices = np.arange(train_x.shape[0])
        df_X = pd.DataFrame(train_x)
        h = 0.02  # メッシュのステップサイズ

        x_min, x_max = X_embedded[:, 0].min() - 1, X_embedded[:, 0].max() + 1
        y_min, y_max = X_embedded[:, 1].min() - 1, X_embedded[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        # メッシュ上の各点に対する予測
        Z = self.svm_clf.predict(xx, yy)

        # プロット
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
        plt.title('SVM Decision Boundary')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.show()

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

