''' Mining the source data from the internet.'''
# Web crawler mining
import requests
from bs4 import BeautifulSoup

# Data analysis
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt


def set_font():
    """Settings of font."""

    # Set Chinese characters.
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # Cope with the excepted display of the negative sign.
    plt.rcParams['axes.unicode_minus'] = False 
    # Set font.
    font = FontProperties(fname = r'C:\Windows\Fonts\STKAITI.TTF', size = 19)
    return font


def get_url(city, year, month):
    """Concatenate url."""

    raw_url = 'http://www.tianqihoubao.com/aqi/{}-{}{}.html'
    url = raw_url.format(city, year, month)
    return url


def get_data(url):
    """Common web crawler mining methods."""

    # Enhance retry.
    requests.DEFAULT_RETRIES = 5
    s = requests.session()
    # Close redundant connections.
    s.keep_alive = False

    response = s.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    # Find the table, sort it out and read.
    read_table = soup.find('table')
    df = pd.read_html(read_table.prettify(), header = 0)[0]
    return df


def save_df(dfs, filename):
    """Save the .csv document."""

    for df in dfs:
        df.to_csv(filename, mode = 'a', header = None)
        

'''Main function'''
# ['Date', 'Level', 'AQI', 'AQI_Ranking', 'PM2.5', 'PM10', 'So2', 'No2', 'Co', 'O3']
city = input('Please enter the city: ')
year = input('Please enter the year: ')
filename = city + year

for month in range(1,13):
    print('Mining data in the %dth month...' %month)
    dfs = []

    # Months processing.
    month = str(month)
    if len(month) == 1:
        month = '0' + month

    font = set_font()
    url = get_url(city, year, month)
    df = get_data(url)
    dfs.append(df)
    save_df(dfs, (filename + '.csv'))

print('Successfully!')
