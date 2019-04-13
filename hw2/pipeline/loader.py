import pandas as pd
import os
import numpy as np

DATA_DIR = os.getcwd() + 'data/'

def load (csv_file):
    '''
    Read the csv file into pd dataframe
    Input:
      csv_file: (string) csv file name 
    Return:
      df: pandas data frame 
    '''
    df = pd.read_csv(DATA_DIR + filename)
    return df
  

    
