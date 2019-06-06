'''
Main function for the pipeline
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

import yaml
from collections import OrderedDict
from itertools import product
import logging
import sys
import numpy as np
import argparse
import os
from pipeline import model_factory
from pipeline import evaluator

from pipeline.get_dummy import *
from pipeline.imputer import *
from pipeline.minmax_scaler import *

import transformer
import pandas as pd
import gc
#from memory_profiler import profile
#from contextlib import contextmanager


logger = logging.getLogger('main function')
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

def run(config):
    logger.info("starting to run the pipeline")
    #pdb.set_trace()
    #config = args.config
    with open (config) as config_file:
        configs = yaml.safe_load(config_file)
    #'input_file', 'roc_path', 'pr_path','out_path'

    df = pd.read_csv(configs['io']['input_path'])
    cols_config = configs['cols']
    time_config = configs['time']
    trans_configs = configs['transform']
    model_configs = configs['models']
    matrix_configs = configs['matrix']
    count = 1
    for data in split(cols_config, time_config, df):
        X_train, X_test, y_train, y_test = data
        X_train, X_test = transformer.transform(trans_configs, X_train, X_test)
        results_df = pd.DataFrame(columns=matrix_configs['col_list'])
        for name, model in model_factory.get_models(model_configs):
            logger.info('start to run the model {}'.format(model))
            model.fit(X_train, y_train)
            print(sys.getsizeof(model))
            if name == 'LinearSVC':
               y_pred_probs = model.decision_function(X_test)
            else:
               y_pred_probs = model.predict_proba(X_test)[:, 1]
            index = len(results_df)
            results_df.loc[index] = get_matrix(results_df, y_pred_probs, y_test, name, model, count,index, matrix_configs)
            del model
            gc.collect()
        results_df.to_csv(matrix_configs['out_path'] + str(count) + ".csv")
        count += 1

def split(cols_config, time_config, df):
    logger.info('starging to split the dataframe')
    X = df[cols_config['x_cols']]
    y = df[cols_config['y_col'][0]]
    min_year = time_config['start_year']
    max_year = time_config['end_year']
    for year in range(min_year + 1, max_year - 3, 2):
        X_train = X[X['year'] <= year]
        X_test = X[(X['year'] == year + 3) | (X['year'] == year + 4)]
        y_train = y[X['year'] <= year].ravel()
        y_test = y[(X['year'] == year + 3) | (X['year'] == year + 4)].ravel()
        logger.info('delivering data to pipeline')
        yield X_train, X_test, y_train, y_test


def get_matrix(results_df, y_pred_probs, y_test, name, model, count, index, matrix_configs):
    # Sort true y labels and predicted scores at the same time
    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
    # Write the evaluation results into data frame
    threshold = matrix_configs['percentage']
    record = [name, str(model),
              evaluator.precision_at_k(y_test_sorted, y_pred_probs_sorted, 100),
              evaluator.compute_acc(y_test_sorted, y_pred_probs_sorted, threshold),
              evaluator.compute_f1(y_test_sorted, y_pred_probs_sorted, threshold),
              evaluator.compute_auc_roc(y_test_sorted, y_pred_probs_sorted, threshold)]

    threshold_list = [1, 2, 5, 10, 20, 30, 50]
    for t in threshold_list:
    	record.append(evaluator.precision_at_k(y_test_sorted, y_pred_probs_sorted, t))
    	record.append(evaluator.recall_at_k(y_test_sorted, y_pred_probs_sorted, t))

    graph_name_pr = matrix_configs['pr_path'] + r'''precision_recall_curve_{}_{}_{}'''.format(name,count,index)
    #pdb.set_trace()
    evaluator.plot_precision_recall_n(y_test, y_pred_probs, str(model), graph_name_pr, 'save')
    graph_name_roc = matrix_configs['roc_path'] + r'''roc_curve__{}_{}_{}'''.format(name,count,index)
    evaluator.plot_roc(str(model), graph_name_roc, y_pred_probs, y_test, 'save')

    return record


def transform(config,X_train,X_test):
        '''
    perform all the tranform ops on the data   
    Input: 
        config: OrdedDict with the key as the name of op, value as params
    Return:
        dataframe
    '''
    logger.info('begin to transform')
    #pdb.set_trace()
    categorical_col = config['imputation']['cols']
    time_column = config['imputation']['time_col'][0]
    loc_column = config['imputation']['loc_col'][0]
    
    logger.info('start to imputation')
    imputer = community_mean_imputer()
    X_train, X_test = imputer.filled_categorical(X_train, X_test, categorical_col)
    X_train = imputer.train_regional_mean(X_train, loc_column, time_column)
    X_test = imputer.transform_test(X_test, loc_column, time_column)
    
    dummies_cols  = config['dummy']['cols']
    k = config['dummy']['k'][0]

    # Drop year column
    X_train = X_train.drop(columns=[time_column])
    X_test = X_test.drop(columns=[time_column])
    
    #Scaling
    continuous_columns = list(set(X_train.columns) - set(categorical_col))
    logger.info('start to scaling')
    X_train, X_test = min_max_transformation(X_train, X_test, continuous_columns)
    
    logger.info('start to get dummies')
    #get dummies
    for col in dummies_cols:
        X_train, X_test = get_dummies(X_train, X_test, col, k)
    gc.collect()
    return X_train, X_test












if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Do a simple machine learning pipeline, load data, split the data, transform data, build models, run models, get the performace matix results')
    parser.add_argument('--config', dest='config', help='config file for this run', default ='./test_simple.yml')
    args = parser.parse_args()
    run(args)
