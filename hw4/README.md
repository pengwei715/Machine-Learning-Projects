# Unsupervised machine learning

## Using k-means to exam the result of hw5


### Prerequisites

All the packages' requirement is in the enviorment.yml

To clone the enviorment, simply run the following:

```
conda env create -f environment.yml
```

To activate the enviorment, simply run the following:

```
conda activate hw4_env
```

### Files

#### pipeline: a python package that contains the following module

1. loader: Read/Load Data

2. explorer: give summary statistic, detect outlier, enerate distributions of variables, correlations between them.

3. preprocesser: fill NA with median, fill NA with mean, drop NAs. 

4. features_generator: discretize a continuous variable, create binary/dummy variables
from categorical data.

5. classifier: generate the following classifiers

6. evaluater: Accuracy, Clasification report, ROC, presicion_at_k, accuracy_at_k, recall_at_k

7. \_util: all the help functions to help to do plotting

8. time_validation: split the temporal data with rolling windows

#### Report and write-up:

- Report: report.pdf

- write-up: hw4.ipynb


## Authors

* **Rayid Ghani** - *Initial work* - [Design](https://github.com/dssg/MLforPublicPolicy/tree/master/Assignments)
* **Peng Wei** - *Main implementation* [Portfolio and cv](https://pengwei715.github.io/)