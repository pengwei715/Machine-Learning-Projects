import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

def hist_one(df, col_name):
    '''
    Given dataframe and the interest column name
    Plot the distribution of that column

    Input:
        df: dataframe
        col_name: string
    Return:
        plot object
    '''
    ax = sns.distplot(df[df[col_name].notnull()][col_name])
    ax.set_title('Distribution of {}'.format(col_name))
    return ax

def hist_some(df, col_lst = []):
    '''
    Plot some the attributes of the dataframe
    plot all by default

    Input:
        df: dataframe
    Return:
        plt of all distributions of the dataframe
    '''
    if col_lst:
        df = df[col_lst]
    plt.rcParams['figure.figsize'] = 14, 12
    df.hist()
    return plt

def plot_corr(corrs_df, y):
    '''
    Generate a bar plot 
    Inputs:
        corrs_df  dataframe get from explorer.corr_df
        y string: dependent variable 
    Returns:
        bar plot showing correlations 
    '''
    y_corr = corrs_df[y]
    title = 'Correlation with {}'.format(y)
    y_corr.plot.bar(title = title)
    plt.xlabel('Variable')
    return plt

def plot_outlier(df, col_lst):
    '''
    Generate pair-wise statter plot with hue of "is_out"

    Inputs:
        df dataframe with a column of tag
        col_lst string list columns of interests
    Return:
        pair wise  scatter plot 
    '''
    g = sns.PairGrid(df, vars=col_lst, hue="is_out")
    return g.map(plt.scatter)












