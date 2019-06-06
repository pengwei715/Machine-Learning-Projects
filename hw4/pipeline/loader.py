import pandas as pd

def load (csv_file='credit-data.csv', pre='./data/'):
    '''
    Read the csv file into pd dataframe
    Input:
      csv_file: (string) csv file name 
    Return:
      df: pandas data frame 
    '''
    return pd.read_csv(pre + csv_file, parse_dates=True)


def save (df, csv_file, pre='./data/'):
    '''
    Keep the result of to csv file
    Input:
        csv_file: (string) csv file name 
        df: pandas dataframe
    Return:
        nothing, keep the record to csv file
    '''
    df.to_csv(pre + csv_file, index=False)
