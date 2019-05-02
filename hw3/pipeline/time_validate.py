'''
Gnerate the temporal validation training and testing data set

'''
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

def generate_temporal_tain_test(df, start_time, end_time, prediction_window, 
    update_window, time_col, x_cols, y_col):

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

    df_train = df.loc[df[time_col].between(
                    train_start, train_end, inclusive=True)]
    df_test = df.loc[df[time_col].between(
                    test_start, test_end, inclusive=True)]
    return df_train[x_cols],df_test[x_cols], df_train[y_col], df_test[y_col], train_start, train_end, test_start, test_end


