from sodapy import Socrata
import pandas as pd
import numpy as np
import datetime as dt
import geopandas as gpd
import shapely
import json
from census import Census
from us import states
import pdb


NAMES_DIC = {'B01003_001E': "population", 
             'B02001_002E': "population_white_alone", 
             'B02001_003E': "population_black_alone", 
             'B15003_017E': "education_high_school", 
             'B19001_001E': "total_income"}

def get_acs_data():
    '''
    Get the information from census data
    Total population, white population, black population,
    high school degree population, household income

    Return:
        pandas dataframe
    '''  
    c = Census("3eb1575454b4de2cf12e0072bd946ecb852579d2")
    res = c.acs5.get(('NAME', 
               'B01003_001E',
               'B02001_002E',
               'B02001_003E',
               'B15003_017E',
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

def link_crime_acs(geo_crime, acs):
    '''
    Link two geo data frame based on the location
    '''
    proj = {'init': 'epsg:4269'}
    acs['the_geom'] = acs.the_geom.apply(shapely.geometry.shape)
    geo_acs = gpd.GeoDataFrame(acs,crs=proj,geometry='the_geom')
    geo_acs = geo_acs.to_crs({'init': 'epsg:4326'})
    total = gpd.sjoin(geo_crime, geo_acs,op='within',how="left")
    res  =  total.drop_duplicates(['case_number'])
    return res
