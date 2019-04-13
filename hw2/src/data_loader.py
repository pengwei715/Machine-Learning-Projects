import pandas as pd
import os

DATA_DIR = os.getcwd() + 'data/'

def load (csv_file, data_dic):
    '''
    Read the csv file into pd dataframe
    Input:
       csv_file: (string) csv file name
       data_dic: (string) excel file name 
    Return:
       df: pandas data frame 
    '''
    df = pd.read_csv(DATA_DIR + filename)
    return df

    
