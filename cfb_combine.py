# -*- coding: utf-8 -*-
"""
NFL Combine Data Scraper

Description:
This script scrapes NFL draft combine data from Pro Football Reference and saves it as CSV files.
It uses BeautifulSoup to parse HTML and collects key statistics from each season.

Author: Sambi
Created: June 12, 2025
"""

# Import necessary libraries
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import time
import random
import warnings
import os
import re

# Headers to mimic a Google Chrome request
headers = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/98.0.4758.102 Safari/537.36'
    )
}

# Suppress deprecation warnings


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# Function to introduce randomized sleep delays


def sleep(min_sec: float, max_sec: float):
    """
    Introduces a randomized sleep delay to mimic human browsing behavior.

    A log-normal distribution is used to create a variable wait time, reducing the chances
    of triggering anti-scraping protections.

    Args:
        min_sec (float): Minimum sleep duration (seconds).
        max_sec (float): Maximum sleep duration (seconds).
    """
    median = (min_sec + max_sec) / 2
    mu = np.log(median)  # log-normal mean
    sigma = 0.75  # Determines variance in sleep times

    sleep_time = random.lognormvariate(mu, sigma)

    print(f'Sleeping for {round(sleep_time,2)} seconds\n')

    # Ensure sleep time is within the defined range
    sleep_time = max(min_sec, min(sleep_time, max_sec))
    time.sleep(sleep_time)


# Define the range of seasons to scrape
SEASON_DATES = list(range(2025, 2026))  # NFL combine data from 2003 to 2024
# SEASON_DATES = 2025
# URLs for scraping
url_parent = "https://www.sports-reference.com"
url_boxscore = 'https://www.pro-football-reference.com/draft/{season}-combine.htm'

# Storage containers for scraped data
data_storage = []
game_df = pd.DataFrame()

# Loop through each season and scrape data
for season in SEASON_DATES:
    print(f"\n\nScraping boxscores from the {season} regular season")
    sleep(0., 20.)  # Add delay to prevent detection

    # Ensure output directory exists
    if not os.path.exists('cfb_combine/'):
        os.makedirs('cfb_combine/')

    # Construct the URL for the given season
    url_summaries = url_boxscore.format(season=season)
    print(f'Fetching data from: {url_summaries}')

    # Send a request to the website and parse the HTML content
    response = requests.get(url_summaries, headers=headers)
    response.raise_for_status()  # Raise error if request fails
    soup_summaries = BeautifulSoup(response.text, 'lxml')

    # Extract tables from the webpage
    tables = soup_summaries.find_all('div')[0:-50]
    tables = [x for x in tables if len(x) > 0]  # Ensure non-empty tables

    # Debugging output (printing extracted tables)
    y = 0
    for table in tables:
        print(
            f'\n\n====== Table {y} ========\n{table}\n========================\n')
        y += 1

    # Extract data rows from the first table
    data = [table.find_all('tr') for table in tables][0]

    # Extract column headers
    cols = data[0].find_all('th')
    cols = [x.get_text() for x in cols]

    # Process last column (splitting multiple values)
    cols_tmp = cols[-1].split()[-1].replace('(', '').replace(')',
                                                             '').split('/')
    cols.pop(-1)
    cols += cols_tmp

    # Standardized column headers
    cols = ['player', 'position', 'school', 'college', 'Ht', 'Wt', '40yd', 'Vertical',
            'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'tm', 'rnd', 'pick', 'yr']

    # Extract all player rows
    rows = data[1:]

    # Function to extract relevant player data
    def extract_data(tag):
        """
        Extracts relevant player data from a BeautifulSoup tag.

        Args:
            tag (BeautifulSoup element): The tag containing player data.

        Returns:
            list: Extracted data as a list of strings.
        """
        if tag is None:
            return []  # Return empty list if tag is None

        data_list = []
        for cell in tag.find_all(['th', 'td']):
            full_text = cell.get_text(separator=" ", strip=True)
            # Avoid None values
            data_list.append(full_text if full_text else "")

        return data_list

    # Function to clean numeric values
    def extract_numbers(data):
        """
        Extracts only numeric values from the player's draft information.

        Args:
            data (list): Raw extracted values containing text and numbers.

        Returns:
            list: Cleaned list with numbers extracted and spaces stripped.
        """
        cleaned_data = [data[0].strip()]  # Preserve first item (player name)
        cleaned_data += [re.sub(r'\D', '', item).strip()
                         for item in data[1:]]  # Remove non-numeric characters
        return cleaned_data

    # Collect player data from rows
    combine_data = []
    for i in rows:
        print(f'\nDATA ROW: {i}')
        row_info = extract_data(i)
        print(f'Extracted Data: {row_info}')

        try:
            # Process draft round and pick
            row_info_tmp = row_info[-1].split('/')
            row_info_tmp = extract_numbers(row_info_tmp)
            row_info.pop(-1)  # Remove old value
            row_info += row_info_tmp
        except IndexError:
            continue  # Skip invalid entries

        combine_data.append(row_info)

    # Convert scraped data into a Pandas DataFrame
    combine_df = pd.DataFrame(combine_data, columns=cols)

    # Drop unnecessary columns
    combine_df.drop(['college'], axis=1, inplace=True)

    # Drop rows where 'player' column is 'Player'
    combine_df = combine_df[combine_df["player"] != "Player"]

    # Save data to a CSV file
    combine_df.to_csv(f'cfb_combine/combine_{season}.csv', index=False)
    print(f"Data saved: cfb_combine/combine_{season}.csv")
