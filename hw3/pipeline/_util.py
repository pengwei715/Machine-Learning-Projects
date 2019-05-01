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
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier,BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
from sklearn.externals import joblib
from datetime import datetime
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *

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

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(
        np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall

def baseline(X_train, X_test, y_train, y_test):
    clf = DummyClassifier(strategy='most_frequent', random_state=0)
    clf.fit(X_train, y_train)
    return clf

def accuracy_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(
        np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    acc = accuracy_score(y_true_sorted, preds_at_k)
    return acc

def f1_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(
        np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    f1 = f1_score(y_true_sorted, preds_at_k)
    return f1

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
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()