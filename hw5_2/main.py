#!/usr/bin/env python3
'''
main function of running the donor's problem

produce a dataframe of the result and two graphs for each model
one is the AUC of ROC, the other one is the precision and recall curve.
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
from pipeline import preprocessor as pro
from pipeline import _util as ut
from pipeline import evaluator as ev
from pipeline import features_generator as fe
from datetime import timedelta
from pipeline import time_validate as tv
import pdb



def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test, grid_size,
    train_start, train_end, test_start, test_end ):
    '''
    Runs the loop using models_to_run, clfs, grid and the data

    Input:
        models_to_run: a list of models
        clfs: the classifiers built from the classifer.py
        X_train: train data of the independent variable.
        y_train: outcome of the train data
        X_test: test data fo the independent variable
        y_test: outcome of the testing data
        grid_size: choose one from test, small, large
        train_start: start date of training data
        train_end: end data of the training data
        test_start: start date of the testing data
        test_end: end date of the testing data
    return:
        save the graphs
        reutrn the dataframe of the model     
    '''
    results_df =  pd.DataFrame(columns=(
                  'train_start',
                  'train_end', 
                  'test_start',
                  'test_end',
                  'model_type',
                  'clf', 
                  'parameters',
                  'auc-roc',
                  'baseline_at_1',
                  'baseline_at_2',
                  'baseline_at_5',
                  'baseline_at_10',
                  'baseline_at_20',
                  'baseline_at_30',
                  'baseline_at_50',
                  'accuracy_at_1',
                  'accuracy_at_2',
                  'accuracy_at_5',
                  'accuracy_at_10',
                  'accuracy_at_20',
                  'accuracy_at_30',
                  'accuracy_at_50',
                  'precision_at_1',
                  'precision_at_2',
                  'precision_at_5',
                  'precision_at_10',
                  'precision_at_20',
                  'precision_at_30',
                  'precision_at_50',
                  'recall_at_1',
                  'recall_at_2',
                  'recall_at_5',
                  'recall_at_10',
                  'recall_at_20',
                  'recall_at_30',
                  'recall_at_50'
                  ))
    
    baseline_clf = ev.baseline(X_train, X_test, y_train, y_test)
    baseline_y_pred_probs= baseline_clf.predict_proba(X_test)[:,1]
    baseline_y_pred_probs_sorted, base_y_test_sorted = zip(*sorted(zip(baseline_y_pred_probs, y_test), reverse=True))
    for index, clf in enumerate([clfs[x] for x in models_to_run]):
        print(models_to_run[index])
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            try:
                clf.set_params(**p)
                y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                row = [train_start, train_end,
                       test_start, test_end,
                       models_to_run[index],
                       clf, p,
                       ev.roc_auc_score(y_test, y_pred_probs)]
                precision_at_k_baseline = []
                accuracy_at_k_lst = []
                precision_at_k_lst = []
                recall_at_k_lst = []
                for i in [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0]:
                    base_pre = ev.precision_at_k(base_y_test_sorted,
                        baseline_y_pred_probs_sorted, i)
                    precision_at_k_baseline.append(base_pre)
                    accuracy = ev.accuracy_at_k(y_test_sorted, y_pred_probs_sorted, i)
                    accuracy_at_k_lst.append(accuracy)
                    precision =  ev.precision_at_k(y_test_sorted,y_pred_probs_sorted,i)
                    precision_at_k_lst.append(precision)
                    recall = ev.recall_at_k(y_test_sorted,y_pred_probs_sorted,i)
                    recall_at_k_lst.append(recall)
                for item in [precision_at_k_baseline, accuracy_at_k_lst, precision_at_k_lst, recall_at_k_lst]:
                    row.extend(item)
                results_df.loc[len(results_df)] = row   
                name = './graphs/p_r_graph'+ \
                       models_to_run[index] + \
                       '-{}-'.format(str(train_end)) +\
                       datetime.now().strftime("%m-%d-%Y, %H:%M:%S")
                ut.plot_precision_recall_n(y_test, y_pred_probs, name, 'save')
                roc_name = './graphs/roc_graph'+\
                           models_to_run[index] + \
                           '-{}-'.format(str(train_end))+\
                           datetime.now().strftime("%m-%d-%Y, %H:%M:%S")
                ut.plot_roc(roc_name, y_pred_probs, y_test, 'save')
            except IndexError as e:
                print('Error:',e)
                continue
    return results_df


def run_time_validation(models_to_run, clfs, grid, grid_size,
    df, start_time, end_time, prediction_window, update_window, time_col, x_cols, y_col):
    '''
    Run all the models on diffent training and testing datasets
    save the result into a csv file
    '''
    res_lst = []
    for item in tv.generate_temporal_train_test(df, start_time, end_time, 
        prediction_window, update_window, time_col, x_cols, y_col):
        x_train, x_test, y_train, y_test, train_start, train_end, test_start, test_end = item
        y_train, y_test = y_train.values.ravel(), y_test.values.ravel()
        x_train = transform(x_train)
        x_test = transform(x_test)
        cols = list(set(x_train.columns).intersection(set(x_test.columns)))
        x_train = x_train[cols]
        x_test = x_test[cols]
        temp = clf_loop(models_to_run, clfs, grid, x_train, x_test, y_train, y_test, grid_size,
            train_start, train_end, test_start, test_end )
        res_lst.append(temp)
    res = pd.concat(res_lst)
    res.to_csv('./results/'+datetime.now().strftime("%m-%d-%Y, %H:%M:%S"), index=False)
    return res


def transform(df):
    '''
    perform the clean data, imputation to the dataframe in the main loop
    '''
    cat_cols =  ['school_city', 'school_district', 'school_county',
                 'school_state', 'school_metro',
                'teacher_prefix', 'resource_type',
                'primary_focus_subject', 'primary_focus_area',
                'secondary_focus_subject', 'secondary_focus_area',
                'poverty_level', 'grade_level']
    df = fe.dummize(df, cat_cols)
    tf_cols = ['school_charter', 'school_magnet',
               'eligible_double_your_impact_match']
    df = ex.replace_tfs(df, tf_cols)
    df['students_reached'] = df['students_reached'].fillna(0)
    df['total_price_norm'] = preprocessing.scale(df['total_price_including_optional_support'].astype('float64'))
    df['students_reached_norm'] = preprocessing.scale(df['students_reached'].astype('float64'))
    #pdb.set_trace()
    return df


if __name__ == '__main__':
	df = lo.load('projects_2012_2013.csv')
	df['date_posted'] = pd.to_datetime(df['date_posted'])
    df['datefullyfunded'] = pd.to_datetime(df['datefullyfunded'])
    df['not_funded_in_60_days'] = \
        (df['datefullyfunded'] - df['date_posted'] >= pd.to_timedelta(60, unit='days')).astype('int')
    
    xs_lst = [item for item in df.columns if item not in {'date_posted',
                                                      'datefullyfunded',
                                                      'not_funded_in_60_days',
                                                      'projectid',
                                                      'teacher_acctid',
                                                      'schoolid',
                                                      'school_ncesid'}]
    grid_size = 'test'
    clfs, grid = clas.define_clfs_params(grid_size)
    models_to_run=['DT','LR','RF','AB', 'GB','ET','BG']
    res = main.run_time_validation(models_to_run, clfs, grid, grid_size,
        df, '2012-01-01' ,'2014-1-1' , 6, 6, 'date_posted', xs_lst, ['not_funded_in_60_days'])

    