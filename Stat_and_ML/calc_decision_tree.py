# -*- coding: utf-8 -*-
"""
決定木で重要特徴量を調べる
Created on Dec 12 2023

@author: S.O
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn import tree

class Calc_DecisionTree:
    
    def __init__(self):
        a = 1

    def fit(self, features, Y,max_depth=5):
        self.Y = Y

        self.features = features
        self.model = tree.DecisionTreeClassifier(max_depth=max_depth)
        self.result = self.model.fit(self.features, self.Y)
        print(self.result.score(self.features, self.Y))

    def show_tree(self):    
        plt.figure(figsize=(100,100))
        plot_tree(self.result, feature_names=list(self.features.columns), filled=True)
        #plot_tree(result, filled=True)
        plt.show()

    def show_feature_importances(self):

        feature_imp = self.model.feature_importances_
        label = self.features.columns
        indices = np.argsort(feature_imp)

        # 特徴量の重要度の棒グラフ
        fig =plt.figure (figsize = (10,len(label)//2))

        plt.barh(range(len(feature_imp)), feature_imp[indices])

        plt.yticks(range(len(feature_imp)), label[indices], fontsize=14)
        plt.xticks(fontsize=14)
        plt.ylabel("Feature", fontsize=18)
        plt.xlabel("Feature Importance", fontsize=18)
        plt.grid()
        plt.show()

def main():
    data = {
    'A': [1, 1, 0, 0, 1],
    'B': [1, 1, 0, 1, 0],
    'C': [1, 0, 0, 1, 0],
    'Y': [1, 0, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    # 特徴量とラベルに分割
    X = df[['A', 'B','C']]
    Y = df[['Y']]
    dt = Calc_DecisionTree()
    dt.fit(X, Y)
    dt.show_feature_importances()


if __name__ == '__main__':
    main()