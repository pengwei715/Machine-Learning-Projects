'''
Generate the temporal validation training and testing data set
for cleaning and imputation.
'''

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

def generate_temporal_train_test(df, start_time, end_time, prediction_window, 
    update_window, time_col, x_cols, y_col):
    '''
    Generate all the training and testing dataframe based on the time
    
    Input:
        df: data frame with one time column
        start_time: the time of the start date
        end_time: the time of the end date
        prediction_window, the time period that need to wait to get the outcome
        update_window: the time period depends on user's pick
        time-col: the column name of the time information
        x_cols: list of all independent variables
        y_col: the dependent variable
    Return:
        A updated parameter for the next iteration and the training and testing 
        data set.
        
    '''
    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')

    test_end_time = end_time_date
    train_start_time = start_time_date
    while (test_end_time >= start_time_date + 2 * relativedelta(months=+prediction_window)):
        test_start_time = test_end_time - relativedelta(months=+prediction_window)
        train_end_time = test_start_time  - relativedelta(days=+60) # minus 1 day
        print(train_start_time, train_end_time, test_start_time, test_end_time)
        yield extract_train_test_sets (df, time_col, train_start_time, 
            train_end_time, test_start_time, test_end_time, x_cols, y_col)
        test_end_time -= relativedelta(months=+update_window)

def extract_train_test_sets(df, time_col,train_start, 
    train_end, test_start, test_end, x_cols, y_col):
    '''
    Generate one training and testing dataframe based on the time
    
    Input:
        df: the untouched whole dataframe
        time_col: the column name of the time information
        train_start: the start date of the single training dataset
        train_end: the end data of the single training dataset
        test_start: the start date of the signle testing data
        test_end: the end date of the single testing data
        X_cols: the independent variables.
        y_col: the dependent variable
    return:
        the training dataset and the testing dataset
    '''
    df_train = df.loc[df[time_col].between(
                    train_start, train_end, inclusive=True)]
    df_test = df.loc[df[time_col].between(
                    test_start, test_end, inclusive=True)]
    return df_train[x_cols],df_test[x_cols], df_train[y_col], df_test[y_col], train_start, train_end, test_start, test_end


