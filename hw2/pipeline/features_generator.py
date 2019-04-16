'''
Get some features from the original col
    -- binize some continuous variable
    -- change categorical columns to dummy
'''
import pandas as pd 
import numpy as np 

AGE_BINS = [0, 21, 36, 50, 65, 100]
AGE_LABELS = ['juvenile', 'young_adult', 'adult', 'middle_aged', 'senior']
AGE_CAT = 'age_category'


def binize(df, col, col_bins, col_lables, new_name):
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
    df[new_name] = pd.cut(df[col],
        bins=col_bins,labels=col_lables,
        include_lowest=True, right=True)
    return df

def dummize(df, col):
    '''
    Make one categorical data dummy

    Input:
        df: dataframe
        col: the col name of the categorical data
    Return:
        dataframe with a new dummy column
    '''
    dum_df = pd.get_dummies(df[col], prefix=col)
    return  pd.concat([df, dum_df], axis = 1)

