'''
Get some features from the original col
    -- binize some continuous variable
    -- change categorical columns to dummy
'''
import pandas as pd 
import numpy as np
from sklearn import preprocessing

def binize(df, col, bins=None, labels=None):
    '''
    Cut the col into categories

    Input:
        df: dataframe 
        col: continous column that needs to be cut
        col_bins: list of numbers, boundaries of each bin
        col_labels: list of string to fill the new col
        new_name: the categories col 's new name
    Return:
        dataframe with a new categorical column
    '''
    if not bins and not labels:
        series = pd.qcut(df[col], q=4)
    elif not labels and bins != None:
        series = pd.qcut(df[col], q=bins)
    elif not bins and labels != None:
        series = pd.qcut(df[col], q=len(labels), labels=labels)
    else:
        series = pd.qcut(df[col], q=bins, labels=labels)
    return series

def dummize(df, cols):
    '''
    Make one categorical data dummy

    Input:
        df: dataframe
        col: the col name of the categorical data
    Return:
        dataframe with a new dummy column
    '''
    return pd.get_dummies(data=df, columns=cols, dummy_na=True)

def dummize_top_k(df, cols, k, count_col):
    '''
    make dummies from the cols that have too many discrete values
    only take the top k categories
    Input:
        df: the dataframe
        cols: the list of the discrete columns
        k: number of top k
        count-col: the column that used to count
    Return:
        the dataframe with new k+1 columns one is other
    '''
    for i, item in enumerate(cols):
        lst = list(df.groupby(item).count().reset_index().sort_values(
            by = count_col, ascending =False).head(k)[item])
        need_to_replace = set(df[item].unique()) - set(lst)
        df[item].fillna('unknown',inplace =True)
        df[item]= df[item].replace(list(need_to_replace), 'other')
    return dummize(df, cols)
