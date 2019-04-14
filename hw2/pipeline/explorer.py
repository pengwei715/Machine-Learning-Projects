import pandas as pd
import numpy as np
from pipeline import _util as ut
from scipy import stats

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
    if col_lst:
        df = df[col_lst]

    return df[(np.abs(stats.zscore(df)) < 3).any(axis=1)]

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

    df['is_out'] = df.index
    common = df[(np.abs(stats.zscore(df)) < 3).any(axis=1)]
    extreme = df[(np.abs(stats.zscore(df)) > 3).any(axis=1)]
    common.is_out = [False for i in range(len(common))]
    extreme.is_out = [True for i in range(len(extreme))]
    return df