"""
データの描画
Created on Dec 12 2023

@author: S.O
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



def violinplot1(df):
    plt.style.use('default')
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('gray')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.violinplot(df)
    ax.set_xticklabels(df.columns.values)
    ax.set_xlabel('items')
    ax.set_ylabel('val')

    data_max = df.max().max()
    data_min = df.min().min()

    ax.set_ylim(data_min-(data_max-data_min)*0.05, data_max+(data_max-data_min)*0.05)
    plt.show()


def violinplot2(df):
    plt.style.use('default')
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('gray')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    sns.violinplot(data=df, color='gray', ax=ax)

    ax.set_xticklabels(df.columns.values)
    ax.set_xlabel('items')
    ax.set_ylabel('val')
    
    data_max = df.max().max()
    data_min = df.min().min()

    ax.set_ylim(data_min-(data_max-data_min)*0.05, data_max+(data_max-data_min)*0.05)
    plt.show()


def show_describe(df):
    print(df.describe())
    try:
        display(df.describe()) # displayがJupyterNotebookでしか使えないので、こういう処理を入れた
    except:
        print(df.describe())

def show_corr(df):
    sns.heatmap(df.corr(), cmap= sns.color_palette('coolwarm', 10), annot=True,fmt='.2f', vmin = -1, vmax = 1)
    plt.show()


