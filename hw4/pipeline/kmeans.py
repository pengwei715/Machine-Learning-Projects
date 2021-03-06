from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
import numpy as np
import graphviz
import pdb

def label(k, Xs):
    '''
    This function uses kmeans to cluster data points.
    k: number of cluster
    Xs: independent dataframe
    
    returns: Kmeans labels
    '''
    clf = KMeans(n_clusters = k)
    clf.fit(Xs)
    return list(clf.labels_)
def summarize(df_pred, pred_col, method = 'mean'):
    '''
    This function summarizes the features, grouped by the label
    
    df_pred: df with predictions
    pred_col: column name of predictions
    method: method to aggregate, default is mean
    
    returns: two tables one contains the average of each featurs,
             the other contains the count of each cluster
    '''
    return df_pred.groupby(pred_col).agg([method]).astype(str).T,df_pred.groupby(pred_col)[pred_col].count()

def merge_clusters(df, new_cluster, clusters):
    '''
    Given a dataframe, then merge several clusters into one.
    Inputs:
        df: a dataframe
        clusters: (list) of clusters that are gonna be merged
        new_cluster: (str) label of merged cluster
    Returns:
        a dataframe with merged clusters
    '''
    df.loc[df['label'].isin(clusters), 'label'] = new_cluster
    return df


def split_cluster(df, features, cluster, number):
    '''
    split one cluster into many
    '''
    data = df[df['label'] == cluster]
    data_2 = df[df['label']!= cluster]
    data['label']= label(number, data[features])
    temp = len(df['label'].unique())
    data['label'] = data['label']+ temp
    return pd.concat([data,data_2])                                                    
    
def viz(df, features, target, cluster, depth):
    '''
    Plot decision tree.
    Inputs:
        df: a dataframe
        features: (list) of features
        target: (str) target variable
        cluster: (int) number of clusters
    Returns:
        a decision tree graph
    '''
    dt = DecisionTreeClassifier(max_depth = depth)
    dt.fit(df[features], df[target])
    dot_data = tree.export_graphviz(dt, feature_names=features,\
        class_names=True, filled=True, rounded=True, out_file=None)
    file = 'DT_{}'.format(cluster)
    graph = graphviz.Source(dot_data, filename=file, format='png')
    graph.view()
    return graph
