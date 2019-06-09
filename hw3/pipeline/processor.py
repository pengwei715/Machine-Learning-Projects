'''
Process the data
    -- fill null with meadian
    -- fill null with mean
    -- drop null
'''
import pandas as pd
import numpy as np


def fill_median(df, col_lst = []):
    '''
    Fills all missing values with the median value
    Inputs:
        df dataframe
    Returns:
        chagne in place
    '''
    if col_lst:
    	for item in col_lst:
    		df[item].fillna(df[item].median(), inplace=True)
    df.fillna(df.median(), inplace=True)

def fill_mean(df, col_lst = []):
    '''
    Fills all missing values with the mean value 
    Inputs:
        df dataframe
    Returns:
        change in place
    '''
    if col_lst:
    	for item in col_lst:
    		df[item].fillna(df[item].mean(), inplace=True)
    df.fillna(df.mean(), inplace=True)

def drop_na(df, col_lst):
	'''
	Drops NAs from the columns
	Inputs:
		df dataframe
		col_lst:list of columns that needs drop
	Returns:
		dataframe
	'''
	return df.drop(col_lst, axis=1)