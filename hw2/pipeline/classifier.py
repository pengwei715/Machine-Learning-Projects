'''
Build Logistic regression models
    --split the data into four subsets
    --Using RFECV to get the best features
    --Build the model using the set of features
'''
import sys
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import statsmodels.api as sm

def split_data(xs_df, y_df, size_test, seed):
    '''
    Split the data into 4 sub dataframe

    Input:
        xs_df: dataframe of independent variables
        y_df: dataframe of dependent variable
    Return:
        Four sub dataframes for training and testing
    '''
    x_train, x_test, y_train, y_test = train_test_split(xs_df,
        y_df, test_size= size_test, random_state = seed)
    return x_train, x_test, y_train, y_test

def select_feature(data_x, data_y):
    '''
    Get the best set of features that used to build the model

    Input:
        data_x: ataframe of independent variables from trainning data
        data_y: dataframe of dependent variable from training data
    Return:
        rank dict of all features
    '''
    logreg = LogisticRegression()
    rfe = RFECV(logreg)
    rfe = rfe.fit(data_x, data_y.values.ravel())
    columns =  pd.DataFrame(data = data_x.columns)
    return columns[rfe.support_], rfe.support_, rfe.ranking_

def build_model(x_train, y_train):
    '''
    Using the opimized set of x columns data and dependent data
    to build a logit regression model

    Input:
        data_x: ataframe of independent variables from trainning data
        data_y: dataframe of dependent variable from training data
    Return:
        summary of statistic of the model
    '''
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    return logreg