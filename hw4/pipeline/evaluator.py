'''
Evaluate the model with different matics
    -- get the accuracy
    -- get report
    -- get roc
'''
from __future__ import division
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import *
import random
from scipy import optimize
import time
import seaborn as sns
from sklearn.externals import joblib
from datetime import datetime


def pre(model, x_test, threshold):
    '''
    Use model to do predictions

    Input:
        model: log model
        x_test: dataframe of independent variable
        threshod: float that used to do the prediction
    Return:
        A array of predictions
    '''
    pred_scores = model.predict_proba(x_test)
    pred_label = [1 if x[1]>threshold else 0 for x in pred_scores]
    return pred_label


def get_accu(y_test, pres):
    '''
    Use model to do predictions

    Input:
        model: log model
        x_test: dataframe of independent variable
    Return:
        A array of predictions
    '''
    return metrics.accuracy_score(y_test, pres)

def get_report(y_test, pres):
    '''
    Get the report of the model

    Input:
        y_test: dependent variable's dataframe
        pres: np array of predictions
    Return:
        report withprecision, recall, f1-score, support
    '''
    return classification_report(y_test, pres)

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
    clf = DummyClassifier(strategy='stratified', random_state=0)
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
