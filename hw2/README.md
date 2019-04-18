# Financial Distress Prediction 

## Skeleton of a machine learning pipe line

### Data Sources
Data in this project is draw from https://www.kaggle.com/c/GiveMeSomeCredit

### Prerequisites

All the packages' requirement is in the enviorment.yml

To clone the enviorment, simply run the following:

```
conda env create -f environment.yml
```

To activate the enviorment, simply run the following:

```
conda activate hw2_env
```

### Files

#### pipeline: a python package that contains the following module

1. loader: Read/Load Data

2. explorer: give summary statistic, detect outlier, enerate distributions of variables, correlations between them.

3. processer: fill NA with median, fill NA with mean, drop NAs. 

4. features_generator: discretize a continuous variable, create binary/dummy variables
from categorical data.

5. classifier: Logistic Regression 

6. evaluater: Accuracy, Clasification report, ROC

7. \_util: all the help functions to help to do plotting

#### Report and write-up:

- Report: homework_2.ipynb

- write-up: homework_2_writeup.html


## Authors

* **Rayid Ghani** - *Initial work* - [Design](https://github.com/dssg/MLforPublicPolicy/tree/master/Assignments)
* **Peng Wei** - *Main implementation* [Portfolio and cv](https://pengwei715.github.io/)