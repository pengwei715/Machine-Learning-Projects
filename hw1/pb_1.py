from sodapy import Socrata
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt


MAX_ROWS = 6839451

COL_TYPES = {'block': str, 
             'case_number': str,
             'community_area': str,
             'primary_type': 'category',
             'date': str,
             'latitude': float,
             'longitude': float,
             'year': int,
             'ward': int}

def get_crime():
	'''
	Get 2017 and 2018 crime data from chicago data portal

	Return:
	    pandas dataframe with the columns and dtypes as COL_TYPES
	'''
	cols = [item for item in COL_TYPES.keys()]
	client = Socrata('data.cityofchicago.org',
	                 'Lfkp6VmeW3p5ePTv0GhNSmrWh',
	                 username="pengwei@uchciago.edu",
	                 password="2h1m@k@1men")                     
	res = client.get("6zsd-86xi", select=",".join(cols),
	                 where="year = 2017 or year = 2018",
	                 limit = MAX_ROWS)

	df = pd.DataFrame.from_records(res).astype(COL_TYPES)
	df['date'] = pd.to_datetime(df['date'])
	return df

if __name__ == '__main__':
	go()