#!/usr/bin/env python3
'''
The code is from magic loop https://github.com/rayidghani/magicloops
Split the data for training and testing use
Build the following models
  -- logistic regression
  -- k nearest neighbor
  -- decision tree
  -- SVM
  -- Random Forests
  -- Boosting
  -- Bagging
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
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from datetime import datetime
from sklearn.feature_selection import RFE

# for jupyter notebooks
#%matplotlib inline

# if you're running this in a jupyter notebook, print out the graphs
NOTEBOOK = 0

def define_clfs_params(grid_size):
    """Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3), 
        'BG': BaggingClassifier(n_estimators=10)
            }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 
          'max_depth': [1,5,10,20,50,100], 
          'max_features': ['sqrt','log2'],
          'min_samples_split': [2,5,10], 
          'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 
            'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 
             'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 
            'criterion' : ['gini', 'entropy'] ,
            'max_depth': [1,5,10,20,50,100], 
            'max_features': ['sqrt','log2'],
            'min_samples_split': [2,5,10], 
            'n_jobs': [-1]},
    'AB': {'algorithm': ['SAMME', 'SAMME.R'], 
           'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 
           'learning_rate' : [0.001,0.01,0.05,0.1,0.5],
           'subsample' : [0.1,0.5,1.0], 
           'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': 
           ['gini', 'entropy'], 
           'max_depth': [1,5,10,20,50,100],
           'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],
            'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],
            'weights': ['uniform','distance'],
            'algorithm': ['auto','ball_tree','kd_tree']},
    'BG' : {'n_estimators': [1,10,100,1000,10000], 
            'max_samples': [5,10,20,50,100], 
            'max_features':[1,5,10,20,50,100]}      
    }

    
    small_grid = { 
    'RF':{'n_estimators': [10,100], 
          'max_depth': [5,50], 
          'max_features': ['sqrt','log2'],
          'min_samples_split': [2,10], 
          'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 
            'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 
             'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 
            'criterion' : ['gini', 'entropy'] ,
            'max_depth': [5,50], 
            'max_features': ['sqrt','log2'],
            'min_samples_split': [2,10], 
            'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 
            'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 
           'learning_rate' : [0.001,0.1,0.5],
           'subsample' : [0.1,0.5,1.0], 
           'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 
           'max_depth': [1,5,10,20,50,100],
           'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],
            'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],
            'weights': ['uniform','distance'],
            'algorithm': ['auto','ball_tree','kd_tree']},
    'BG' : {'n_estimators': [10,100], 
            'max_samples': [5,50], 
            'max_features': [20,100]}      
    }
    
    test_grid = { 
    'RF':{'n_estimators': [100], 
          'max_depth': [10], 
          'max_features': ['sqrt'],
          'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 
            'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 
             'penalty': ['l2']},
    'ET': { 'n_estimators': [100], 
            'criterion' : ['gini'] ,
            'max_depth': [10], 
            'max_features': ['sqrt'],
            'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 
            'n_estimators': [100]},
    'GB': {'n_estimators': [100], 
           'learning_rate' : [0.1],
           'subsample' : [0.5], 
           'max_depth': [10]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 
           'max_depth': [10],
           'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [3],
            'weights': ['uniform'],
            'algorithm': ['auto']},
    'BG' : {'n_estimators': [10], 
            'max_samples': [50], 
            'max_features': [50]}  
           }
    
    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0
