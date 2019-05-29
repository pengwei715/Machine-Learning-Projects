'''
Explore the data set
    -- give basic summary statistic
    -- detect outlier using Z score approch
    -- get pair-wise correlations
    -- tag the data as outlier or common
    -- drop some irrelevent columns
'''

import pandas as pd
import numpy as np
from pipeline import _util as ut
from scipy import stats
import pdb
from pipeline import preprocessor as pro


def basic_sum(df):
    '''
    Get the basic summary of stats of the data

    Input:
        df: dataframe 
    Return:
        basic summary stats
    '''
    return df.describe()

def detect_outlier(df, col_lst=[]):
    '''
    Detect outlier of certain column

    Input:
        df: dataframe
        col_lst: list of string column name
    Return:
        All out lier
    Reference:
       https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
    '''
    np.warnings.filterwarnings('ignore')
    if col_lst:
        df = df[col_lst]

    return df[(np.abs(stats.zscore(df)) > 3).any(axis=1)]


def rm_outlier(df, col_lst=[]):
    '''
    Remove all the outlier of the certain columns(in col_lst)

    Input:
        df: dataframe
        col_lst: list of string columns that needs to remove
    Return:
        dataframe without outlier
    '''
    np.warnings.filterwarnings('ignore')
    if col_lst:
        df = df[col_lst]

    return df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

def corr_df(df, col_lst=[]):
    '''
    Generate coorelation table
    Input:
        df dataframe
    Returns:
        dataframe of pairwise variable correlations 
    '''
    if col_lst:
        df = df[col_lst]
    return df.corr(method = 'pearson')


def tag_out(df, col_lst=[]):
    '''
    tage the data as outlier

    Input:
        df: dataframe
        col_lst: list of string columns that needs to remove
    Return:
        dataframe added one column named 'is_out'
    '''
    if col_lst:
        df = df[col_lst]
    temp = detect_outlier(df)
    size, _ = temp.shape
    temp['is_out'] = [True for i in range(size)]
    rows, _ = df.shape
    df['is_out'] = [False for i in range(rows)]
    df.update(temp)
    
    return df

def drop_cols(df, col_lst):
    '''
    drop some irrelavite columns

    Input:
        df: dataframe
        col_lst: the columns that need to drop

    Return:
        dataframe that without col_lst
    '''
    return df.drop(columns=col_lst, axis = 1)


def replace_tfs(df, cols):
    for col in cols:
        arr = df[col]
        df[col] = np.where(arr == 't', 1, 0)
    return df