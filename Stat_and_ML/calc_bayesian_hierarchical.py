# -*- coding: utf-8 -*-
"""
calc_bayesian_hierarchical.py
Created on Dec 12 2023

@author: S.O
"""

import pymc as pm
import numpy as np
import pandas as pd

import arviz as az
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class BayesianHierarchical:
    
    def __init__(self,
                 draws = 1000,
                 tune = 500,
                 ):
        self.mu_a_param = 0
        self.sigma_a_param = 10 
        self.ex_param = 1.0
        self.draws = draws
        self.tune = tune


    def _standard_scaler(self, data ,feature_names):

        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(data), columns=feature_names)

    def run_MCMC(self,
            data,
            category_col_name,
            label_col_name ,
            standard_scaler_flag=True
        ):

        category_codes, industries = pd.factorize(data[category_col_name])
        data['category_code'] = category_codes

        n_category = data[category_col_name].nunique()
        train_x =data.drop([category_col_name, label_col_name], axis=1)
        feature_names = list(train_x.columns.values)
        n_factors = len(feature_names)

        if standard_scaler_flag:
            train_x = self._standard_scaler(train_x, feature_names)

        with pm.Model() as model:
            # 階層事前分布
            mu_a = pm.Normal('mu', mu=self.mu_a_param, sigma=self.sigma_a_param, shape=(n_category, n_factors))
            sigma_a = pm.Exponential('sigma', self.ex_param, shape=(n_category, n_factors))
            effects = pm.Normal('effects', mu=mu_a, sigma=sigma_a, shape=(n_category, n_factors))
            
            # ポアソン分布の平均
            factors = train_x.values
            category_effects = effects[data['category_code']]
            lambda_ = pm.math.exp(pm.math.sum(category_effects * factors, axis=1))
            
            # 観測データ
            y_obj = pm.Poisson('y_obj', mu=lambda_, observed=data[label_col_name])

            # MCMCによるサンプリング
            trace = pm.sample(draws=self.draws , tune=self.tune, return_inferencedata=True, target_accept=0.99)

        with model:
            az.plot_trace(trace)
            summary = az.summary(trace, var_names=['mu', 'effects'])

            print(summary)


        effects_means = summary.loc[[idx for idx in summary.index if 'effects' in idx], 'mean']

        # パラメータ名と値をDataFrameに変換
        effects_df = pd.DataFrame({
            'parameter': effects_means.index,
            'mean_effect': effects_means.values
        })

        effects_df['industry'] = effects_df['parameter'].apply(lambda x: industries[int(x.split('[')[1].split(',')[0])])
        effects_df['factor'] = effects_df['parameter'].apply(lambda x: feature_names[int(x.split('[')[1].split(',')[1].replace(']', ''))])

        effects_df['label'] = effects_df['industry'] + ' - ' + effects_df['factor']

        # 平均効果で並び替え
        effects_df.sort_values(by='mean_effect', ascending=False, inplace=True)

        print(len(effects_df['label']))
        # 棒グラフの描画
        plt.figure(figsize=(10, int(len(effects_df['label'])/6)))
        plt.barh(effects_df['label'], effects_df['mean_effect'])
        plt.xlabel('Mean Effect')
        plt.ylabel('Parameter')
        plt.title('Effects of Factors by Industry')
        plt.gca().invert_yaxis()  # Y軸を逆順にして大きいものが上にくるようにする
        plt.show()

        #pm.model_to_graphviz(model)
        graph = pm.model_to_graphviz(model)

        try:
            display(graph)
        except:
            graph.view()


if __name__ == '__main__':
    # 動作確認用サンプルデータの生成
    np.random.seed(42)
    n_factors = 5  # 要因の数
    n_industries = 3  # 業界の数
    n_samples = 100  # サンプル数
    # 仮想データを生成
    data = pd.DataFrame({
        f'factor{i+1}': np.random.poisson(5, size=n_samples) for i in range(n_factors)
    })
    data['industry_name'] = np.random.choice(['Seizou', 'Kensetu', 'Kouri'], size=n_samples)
    data['sales'] = np.random.poisson(2, size=n_samples)

    bh = BayesianHierarchical(draws = 60,tune = 20)
    bh.run_MCMC(data,'industry_name','sales')