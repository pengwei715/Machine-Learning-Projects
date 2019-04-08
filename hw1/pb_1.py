from sodapy import Socrata
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
import neighborhoods as nbhds
import json

COL_TYPES = {'block': str, 
             'case_number': str,
             'community_area': str,
             'primary_type': 'category',
             'date': str,
             'latitude': float,
             'longitude': float,
             'year': int,
             'ward': int}
MAX_ROWS = 6839451 # the total rows of the original data

CRIME_DATA_ID = "6zsd-86xi"

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
    conds = "year = 2017 or year = 2018"
    res = client.get(CRIME_DATA_ID, 
                     select=",".join(cols),
                     where= conds,
                     limit = MAX_ROWS)
    client.close()
    df = pd.DataFrame.from_records(res)
    df = df[df['ward'].notna()].astype(COL_TYPES)
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_community():
    '''
    Get 77 community nums's community name from chicago data portal

    Return:
        dataframe with community num and community name
    '''
    client = Socrata('data.cityofchicago.org',
                     'Lfkp6VmeW3p5ePTv0GhNSmrWh',
                     username="pengwei@uchciago.edu",
                     password="2h1m@k@1men")
    res = client.get("igwz-8jzy", select = 'area_numbe, community')
    df = pd.DataFrame.from_records(res)
    df.rename(index = str, 
              columns = {"area_numbe": "community_area"}, 
              inplace = True)
    client.close()
    return df

def link_with_neighborhoods(dataframe, lng_col, lat_col):
    '''
    Helper function to get a geocoded dataframe

    Input:
        df: dataframe with latitude and longtitude columns
    Return:
        A dataframe that is geocoded
    '''
    nbhd = nbhds.import_geometries(nbhds.NEIGHS_ID)
    geodf = nbhds.convert_to_geodf(dataframe, lng_col, lat_col)
    return nbhds.find_neighborhoods(geodf, nbhd)

def map_city(geodf):
    '''
    map the a chicago heatmap

    Input:
        geodf: geom dataframe after link with the neighborhoods
        nbhd: geom dataframe of 

    return plot of heatmap
    '''
    nbhd = nbhds.import_geometries(nbhds.NEIGHS_ID)
    first_col = geodf.columns[0]
    fig, ax = plt.subplots(1)
    fig.set_size_inches(20, 13)
    heat = geodf.dissolve(by='pri_neigh', aggfunc='count')
    heat = nbhd.merge(heat, on='pri_neigh', how='left').fillna(0)
    heat.plot(ax=ax, cmap='coolwarm', column=first_col, linewidth=0.8,
              linestyle='-')
    ax.axis('off')
    ax.set_title('Chicago ' + 'crime' + ' Heat Map')
    n_min = min(heat[first_col])
    n_max = max(heat[first_col])
    leg = mpl.cm.ScalarMappable(cmap='coolwarm', norm=mpl.colors.Normalize(
        vmin=n_min, vmax=n_max))
    leg._A = []
    colorbar = fig.colorbar(leg)
    return plt
