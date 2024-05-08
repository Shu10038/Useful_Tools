# -*- coding: utf-8 -*-
"""
主成分分析
Created on Dec 12 2023

@author: S.O
"""
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Calc_PCA:
    
    def __init__(self, features, Y, n_components=3):
        self.Y = Y
        self.n_components = n_components

        self.features_0 = features[Y[Y.columns.values[0]]==0.0]
        self.features_1 = features[Y[Y.columns.values[0]]==1.0]

    def fit(self):
        self.pca = PCA(n_components=self.n_components, svd_solver='randomized')
        #print(np.vstack((self.features_0, self.features_1)))
        self.pca.fit(pd.concat([self.features_0 , self.features_1]))
        self.pca_arr0 = self.pca.transform(self.features_0)
        self.pca_arr1 = self.pca.transform(self.features_1)

        self.contribution_rate_list = ['{:.3f}'.format(n) for n in self.pca.explained_variance_ratio_]


    def plot_by_axis(self):
        plt.figure(figsize=(10, 6))
        
        for i in range(self.n_components):

            plt.subplot(3, 1, i+1)
            plt.title(f'component {i})\n contribution ratio: {self.contribution_rate_list[i]}', fontsize=10)
            plt.grid()
            plt.scatter(range(len(self.pca.components_[i])), self.pca.components_[i], c='blue', lw=1)

        plt.tight_layout()
        plt.show()
        plt.close()

    def show_variance_ratio(self):
        plt.bar(self.contribution_rate_list , self.pca.explained_variance_ratio_)

    def show_pca_3D_plot(self):
        if self.n_components!=3:
            return 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.scatter(self.pca_arr0[:, 0], self.pca_arr0[:, 1],
                    self.pca_arr0[:, 2], marker='o', s=3, c='blue', label='label_0')
        ax.scatter(self.pca_arr1[:, 0],self.pca_arr1[:, 1],
                    self.pca_arr1[:, 2], marker='o', s=3, c='red', label='label_1')
        plt.legend()
        plt.show()
        plt.close()

    def show_pca_2D_plot(self):
        if self.n_components!=2:
            print("Component count is not 2 !!")
            return
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.scatter(self.pca_arr0[:, 0], self.pca_arr0[:, 1],marker='o', s=3, c='blue', label='label_0')
        ax.scatter(self.pca_arr1[:, 0], self.pca_arr1[:, 1] ,marker='o', s=3, c='red', label='label_1')
        
        plt.xlabel(f'1st main component : variance ratio {self.contribution_rate_list[0]}')
        plt.ylabel(f'2nd main component : variance ratio {self.contribution_rate_list[1]}')
        plt.title('Principal component score')
        plt.legend()
        plt.grid()
        plt.show()

    def show_loading_amoun(self):
        if self.n_components!=2:
            print("Component count is not 2 !!")
            return
        plt.figure(figsize=(8, 8))
        for x, y, name in zip(self.pca.components_[0], self.pca.components_[1], self.features_0.columns[0:]):
            plt.text(x, y, name)
        plt.scatter(self.pca.components_[0], self.pca.components_[1])
        plt.grid()
        plt.xlabel(f'Principal component loadings of the 1st principal component : variance ratio {self.contribution_rate_list[0]}')
        plt.ylabel(f'Principal component loadings of the 2nd principal component : variance ratio {self.contribution_rate_list[1]}')
        plt.title('Principal component loading')
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
    pca = Calc_PCA(X, Y)
    pca.fit()
    pca.show_variance_ratio()
    pca.plot_by_axis()
    pca.show_loading_amoun()

    pca2 = Calc_PCA(X, Y,n_components=2)
    pca2.fit()
    pca2.show_variance_ratio()
    pca2.plot_by_axis()
    pca2.show_loading_amoun()

if __name__ == '__main__':
    main()