# Anlysis crime data of Chicago in 2017 and 2018

Using the crime data and augmented with the ACS data to identify some charecteristics in level of blocks and neighborhoods. 

### Data Sources
Data in this project comes from four sources:

City of Chicago Data Portal's Crime Dataset (API): https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2

City of Chicago Data Portal's Neighborhood Boundaries (API): https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Neighborhoods/9wp7-iasj

City of Chicago Data Portal's Block Boundaries Code Boundaries (API):https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Census-Blocks-2010/mfzt-js4n

ACS website's geocoder API and 2017's one year data (API) https://www.census.gov/programs-surveys/acs/technical-documentation/table-and-geography-changes/2017/1-year.html

### Prerequisites

The following Python packages are needed to run this software:

| Package | Version |
|  ---- |  ---- |
| descartes | 1.1.0 |
| geocoder | 1.38.1 |
| geopandas | 0.4.1 |
| matplotlib | 3.0.2 |
| numpy | 1.16.2 |
| pandas | 0.23.4 |
| shapely | 1.6.4.post2 |
| sodapy | 1.5.2 |
| seaborn | 0.9.0 |
|census | 0.8.13|
|us |1.0.0|
|censusgeocode|0.4.3|

### Files

#### py files

- pb_1.py: problem 1 module 

    Get crime, neighborhood data. 
    Link these two data set together. 
    Generate the heatmap of the crime.

- pb_2.py: problem 2 module
    
    Get ACS data using ACS API
    Get 2010 block data from chicago data portal.
    Link ACS data with block data based on Tract id
    Link the result of above with Crime data based on location

- neighborhoods.py: helper module

    Get the neighborhood data
    Geocoding data with longitude and latitude columns
    Link data with zipcode

#### Images file

- Index.png: heap map of the crime of the whole data
- theft.png: heap map of the theft type of crime
- desp.png: heap map of the Deception type of crime.

#### Report and write-up:

- Report: homework_1.ipynb
- write-up: homework_1_writeup.html


## Authors

* **Rayid Ghani** - *Initial work* - [Design](https://github.com/dssg/MLforPublicPolicy/tree/master/Assignments)
* **Peng Wei** - *Main implementation* [Portfolio and cv](https://pengwei715.github.io/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
