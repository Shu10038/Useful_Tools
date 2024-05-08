# -*- coding: utf-8 -*-
"""

Created on Dec 12 2023

@author: S.O
"""
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt

class CalcDecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, n_trials=1000):
        self.n_trials = n_trials
        self.best_params = None
        self.clf = None

    def fit(self, x_train, y_train):
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 1, 32),
                'min_samples_split': trial.suggest_float('min_samples_split', 0.01, 1.0),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
            clf = DecisionTreeClassifier(**params)
            return cross_val_score(clf, x_train, y_train, n_jobs=-1, cv=3).mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.opt_params = study.best_trial.params
        self.clf = DecisionTreeClassifier(**self.opt_params)
        self.clf.fit(x_train, y_train)

    def predict(self, x_test):
        if self.clf is not None:
            return self.clf.predict(x_test)
        else:
            raise NotFittedError("This OptunaOptimizedDecisionTree instance is not fitted yet.")

    def show_score(self, x_train, y_train):
        print(self.clf.score(x_train, y_train))

    def show_feature_importances(self, x_train):

        feature_imp = self.clf.feature_importances_
        label = x_train.columns
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

