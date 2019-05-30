# Models evaluation

## Improve the skeleton of a machine learning pipe line


### Prerequisites

All the packages' requirement is in the enviorment.yml

To clone the enviorment, simply run the following:

```
conda env create -f env.yml
```

To activate the enviorment, simply run the following:

```
conda activate hw3_env
```

To run the pipeline, simply run the following:

```
python3 main.py

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

- Report: [report.pdf](./report.pdf)

#### Graphs

All the ROC curve graph, and precision-recall graph

#### Results

The result of all models, contains information of which trainging data used, which testing data used, every score of different metrics at K.

#### main.py

Go through every model, every parament, every training data to build the model, and generate the results and graphs with the testing data.

#### [CHANGES.md](./CHANGES.md)

The changes that I made based on the feedback of hw3

## Authors

* **Rayid Ghani** - *Initial work* - [Design](https://github.com/dssg/MLforPublicPolicy/tree/master/Assignments)
* **Peng Wei** - *Main implementation* [Portfolio and cv](https://pengwei715.github.io/)
