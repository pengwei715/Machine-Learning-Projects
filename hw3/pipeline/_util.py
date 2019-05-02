'''
Plotting functions
    -- plot hist of given column
    -- plot hist of all columns
    -- plot correlation table
    -- plot pair wise scatter, hue as is_out
    -- plot heat map of correlation table
    -- plot plot_precision_recall
    -- plot 
'''
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import *
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
from sklearn.externals import joblib
from datetime import datetime

def hist_one(df, col_name):
    '''
    Given dataframe and the interest column name
    Plot the distribution of that column

    Input:
        df: dataframe
        col_name: string
    Return:
        plot object
    '''
    ax = sns.distplot(df[df[col_name].notnull()][col_name])
    ax.set_title('Distribution of {}'.format(col_name))
    return ax

def hist_some(df, col_lst = []):
    '''
    Plot some the attributes of the dataframe
    plot all by default

    Input:
        df: dataframe
    Return:
        plt of all distributions of the dataframe
    '''
    if col_lst:
        df = df[col_lst]
    plt.rcParams['figure.figsize'] = 14, 12
    df.hist()
    return plt

def plot_corr(corrs_df, y):
    '''
    Generate a bar plot 
    Inputs:
        corrs_df  dataframe get from explorer.corr_df
        y string: dependent variable 
    Returns:
        bar plot showing correlations 
    '''
    plt.rcParams['figure.figsize'] = 8, 6
    corrs_df.drop(y,axis = 0, inplace =True)
    y_corr = corrs_df[y]
    y_corr.plot.barh(title = 'Correlation plot')
    return plt

def plot_outlier(df, col_lst):
    '''
    Generate pair-wise statter plot with hue of "is_out"

    Inputs:
        df dataframe with a column of tag
        col_lst string list columns of interests
    Return:
        pair wise  scatter plot 
    '''
    g = sns.PairGrid(df, vars=col_lst, hue="is_out")
    return g.map(plt.scatter)

def heat_corr(corr):
    '''
    Plot the correlation heatmap

    Input: 
        corr: dataframe of the correlation
    Return:
        Heatmap of the correlation pair-wise
    '''
    return sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

def plot_precision_recall_n(y_true, y_prob, model_name, output_type):
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0, 1])
    ax1.set_ylim([0, 1])
    ax2.set_xlim([0, 1])

    name = model_name
    plt.title(name)
    if (output_type == 'save'):
        plt.savefig(name, close=True)
        plt.close()
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()

def plot_roc(name, probs, y_true, output_type):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")
    if (output_type == 'save'):
        plt.savefig(name, close=True)
        plt.close()
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()