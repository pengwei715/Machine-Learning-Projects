from sodapy import Socrata
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
import json
from census import Census
from us import states


NAMES_DIC = {'B01003_001E': "population", 
             'B02001_002E': "population_white_alone", 
             'B02001_003E': "population_black_alone", 
             'B15003_021E': "education_associates", 
             'B19001_001E': "total_income"}

def get_acs_data():
    '''
    Get the information from census data
    Total population, white population, black population,
    associates degree population, household income

    Return:
        pandas dataframe
    '''  
    c = Census("3eb1575454b4de2cf12e0072bd946ecb852579d2")
    res = c.acs5.get(('NAME', 
    	       'B01003_001E',
    	       'B02001_002E',
    	       'B02001_003E',
    	       'B15003_021E',
    	       'B19001_001E'),
               {'for': 'block group',
               'in': 'state: {} county: {}'.format('17','031')},
               year = 2017)
    df = pd.DataFrame.from_records(res)
    df.rename(columns=NAMES_DIC,inplace=True)
    df.drop(columns =['NAME'], axis=1, inplace=True)
    df['geoid'] = df["state"] + df["county"] + df["tract"]
    return df

def link_block(acs_df):
	'''
    Get blocks using API from chicago data portal

    Return:
        geo dataframe of chicago
    '''
	client = Socrata('data.cityofchicago.org',
                     'Lfkp6VmeW3p5ePTv0GhNSmrWh',
                     username="pengwei@uchciago.edu",
                     password="2h1m@k@1men")
    res = client.get("74p9-q2aq")
    df = pd.DataFrame.from_records(res)
    df.rename(index = str, 
              columns = {"geoid10": "geoid"}, 
              inplace = True)
    client.close()
	return pd.merge(df, acs_df, on='geoid')

def link_crime_acs(geo_crime, geo_acs):
	'''
	Link two geo data frame based on the location
	'''
	return gpd.sjoin(geo_crime, geo_acs, how="left")


	


