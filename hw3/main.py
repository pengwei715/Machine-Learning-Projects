#!/usr/bin/env python3
'''
main function of the machinelearning pipeline

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
from pipeline import loader as lo
from pipeline import classifier as clas
from pipeline import explorer as ex
from pipeline import processor as pro
from pipeline import _util as ut
from pipeline import evaluator as ev
from pipeline import features_generator as fe
from datetime import timedelta
from pipeline import time_validate as tv
import pdb

def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test, grid_size,
    train_start, train_end, test_start, test_end ):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    results_df =  pd.DataFrame(columns=('train_start','train_end', 
                  'test_start',
                  'test_end',
                  'model_type','clf', 
                  'parameters', 'auc-roc',
                  'baseline_at_1', 'baseline_at_2', 'baseline_at_5',
                  'baseline_at_10', 'baseline_at_20', 'baseline_at_30',
                  'baseline_at_50',
                  'accuracy_at_1', 'accuracy_at_2', 'accuracy_at_5',
                  'accuracy_at_10', 'accuracy_at_20', 'accuracy_at_30',
                  'accuracy_at_50','precision_at_1', 'precision_at_2', 'precision_at_5',
                  'precision_at_10', 'precision_at_20', 'precision_at_30',
                  'precision_at_50', 'recall_at_1', 'recall_at_2', 'recall_at_5',
                  'recall_at_10', 'recall_at_20', 'recall_at_30',
                  'recall_at_50'
                  ))
    
    baseline_clf = ut.baseline(X_train, X_test, y_train, y_test)

    baseline_y_pred_probs= baseline_clf.predict_proba(X_test)[:,1]
    baseline_y_pred_probs_sorted, base_y_test_sorted = zip(*sorted(zip(baseline_y_pred_probs, y_test), reverse=True))
    for index,clf in enumerate([clfs[x] for x in models_to_run]):
        #pdb.set_trace()
        print(models_to_run[index])
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            try:
                clf.set_params(**p)
                y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                # store the model in moldes dictionary
                #joblib.dump(clf, './models/'+ models_to_run[index] + str(p))
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                results_df.loc[len(results_df)] = [train_start, train_end, test_start, test_end,
                    models_to_run[index],clf, p,
                    ut.roc_auc_score(y_test, y_pred_probs),
                    ut.precision_at_k(base_y_test_sorted,baseline_y_pred_probs_sorted,1.0),
                    ut.precision_at_k(base_y_test_sorted,baseline_y_pred_probs_sorted,2.0),
                    ut.precision_at_k(base_y_test_sorted,baseline_y_pred_probs_sorted,5.0),
                    ut.precision_at_k(base_y_test_sorted,baseline_y_pred_probs_sorted,10.0),
                    ut.precision_at_k(base_y_test_sorted,baseline_y_pred_probs_sorted,20.0),
                    ut.precision_at_k(base_y_test_sorted,baseline_y_pred_probs_sorted,30.0),
                    ut.precision_at_k(base_y_test_sorted,baseline_y_pred_probs_sorted,50.0),
                    ut.accuracy_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                    ut.accuracy_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                    ut.accuracy_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                    ut.accuracy_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                    ut.accuracy_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                    ut.accuracy_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                    ut.accuracy_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                    ut.precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                    ut.precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                    ut.precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                    ut.precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                    ut.precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                    ut.precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                    ut.precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                    ut.recall_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                    ut.recall_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                    ut.recall_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                    ut.recall_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                    ut.recall_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                    ut.recall_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                    ut.recall_at_k(y_test_sorted,y_pred_probs_sorted,50.0)
                   ]
                name = './graphs/p_r_graph'+ models_to_run[index] + '-{}-'.format(str(train_end)) + datetime.now().strftime("%m-%d-%Y, %H:%M:%S")
                ut.plot_precision_recall_n(y_test, y_pred_probs, name, 'save')
                roc_name = './graphs/roc_graph'+ models_to_run[index] + '-{}-'.format(str(train_end)) + datetime.now().strftime("%m-%d-%Y, %H:%M:%S")
                ut.plot_roc(roc_name, y_pred_probs, y_test, 'save')
            except IndexError as e:
                print('Error:',e)
                continue
    return results_df


def run_time_validation(models_to_run, clfs, grid, grid_size,
    df, start_time, end_time, prediction_window, update_window, time_col, x_cols, y_col):
    res_lst = []
    for item in tv.generate_temporal_tain_test(df, start_time, end_time, 
        prediction_window, update_window, time_col, x_cols, y_col):
        x_train, x_test, y_train, y_test, train_start, train_end, test_start, test_end = item
        y_train, y_test = y_train.values.ravel(), y_test.values.ravel()
        temp = clf_loop(models_to_run, clfs, grid, x_train, x_test, y_train, y_test, grid_size,
            train_start, train_end, test_start, test_end )
        res_lst.append(temp)
    res = pd.concat(res_lst)
    res.to_csv('./results/'+datetime.now().strftime("%m-%d-%Y, %H:%M:%S"), index=False)
    return res


