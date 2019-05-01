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
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
from sklearn.externals import joblib

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
    'RF':{'n_estimators': [1], 
          'max_depth': [1], 
          'max_features': ['sqrt'],
          'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 
            'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 
             'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 
            'criterion' : ['gini'] ,
            'max_depth': [1], 
            'max_features': ['sqrt'],
            'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 
            'n_estimators': [1]},
    'GB': {'n_estimators': [1], 
           'learning_rate' : [0.1],
           'subsample' : [0.5], 
           'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 
           'max_depth': [1],
           'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],
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

# a set of helper function to do machine learning evalaution

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(
        np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall

def baseline(X_train, X_test, y_train, y_test):
    clf = DummyClassifier(strategy='most_frequent', random_state=0)
    clf.fit(X_train, y_train)
    baseline = clf.score(X_test, y_test)
    return baseline

def accuracy_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(
        np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    acc = accuracy_score(y_true_sorted, preds_at_k)
    return acc

def f1_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(
        np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    f1 = f1_score(y_true_sorted, preds_at_k)
    return f1

def clf_loop(models_to_run, clfs, grid, X, y):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    results_df =  pd.DataFrame(columns=('model_type','clf', 
                  'parameters', 'auc-roc',
                  'p_at_1', 'p_at_2', 'p_at_20'))
    for n in range(1, 2):
        # create training and valdation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    # you can also store the model, feature importances, and prediction scores
                    # we're only storing the metrics for now
                    joblib.dump(clf, './models/'+ models_to_run[index] + str(p))
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0)]
                    if NOTEBOOK == 1:
                        plot_precision_recall_n(y_test,y_pred_probs,clf)
                except IndexError as e:
                    print('Error:',e)
                    continue
    return results_df

'''

def main():

    # define grid to use: test, small, large
    grid_size = 'test'
    clfs, grid = define_clfs_params(grid_size)

    # define models to run
    models_to_run=['RF','DT','KNN', 'ET', 'AB', 'GB', 'LR', 'NB']

    # load data from csv
    df = pd.read_csv("/Users/rayid/Projects/uchicago/Teaching/MLPP-2017/Homeworks/Assignment 2/credit-data.csv")

    # select features to use
    features  =  ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'age', 'NumberOfTimes90DaysLate']
    X = df[features]
    
    # define label
    y = df.SeriousDlqin2yrs

    # call clf_loop and store results in results_df
    results_df = clf_loop(models_to_run, clfs,grid, X,y)
    if NOTEBOOK == 1:
        results_df

    # save to csv
    results_df.to_csv('results.csv', index=False)


if __name__ == '__main__':
    main()
'''