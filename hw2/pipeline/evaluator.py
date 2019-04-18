'''
Evaluate the model with different matics
    -- get the accuracy
    -- get report
    -- get roc
'''
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def pre(model, x_test, threshold):
    '''
    Use model to do predictions

    Input:
        model: log model
        x_test: dataframe of independent variable
        threshod: float that used to do the prediction
    Return:
        A array of predictions
    '''
    pred_scores = model.predict_proba(x_test)
    pred_label = [1 if x[1]>threshold else 0 for x in pred_scores]
    return pred_label


def get_accu(y_test, pres):
    '''
    Use model to do predictions

    Input:
        model: log model
        x_test: dataframe of independent variable
    Return:
        A array of predictions
    '''
    return metrics.accuracy_score(y_test, pres)

def get_report(y_test, pres):
    '''
    Get the report of the model

    Input:
        y_test: dependent variable's dataframe
        pres: np array of predictions
    Return:
        report withprecision, recall, f1-score, support
    '''
    return classification_report(y_test, pres)


def get_roc_auc( y_test, x_test, model, pres):
    '''
    Plot the ROC curve

    Input:
        y_test: dependent variable's dataframe
        pres: np array of predictions
    Return:
        ROC curve
    '''
    logit_roc_auc = roc_auc_score(y_test, pres)
    fpr, tpr, thresholds = roc_curve(y_test,model.predict_proba(x_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    return plt