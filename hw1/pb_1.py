from sodapy import Socrata
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt


MAX_ROWS = 6839451

def get_crime():
	client = Socrata('data.cityofchicago.org',
	                 'Lfkp6VmeW3p5ePTv0GhNSmrWh',
	                 username="pengwei@uchciago.edu",
	                 password="2h1m@k@1men")                     
	res = client.get("6zsd-86xi", select="*",
	                 where="year = 2017 or year = 2018",
	                 limit = MAX_ROWS)
	df = pd.DataFrame.from_records(res)