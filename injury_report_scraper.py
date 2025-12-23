# -*- coding: utf-8 -*-
"""
Created on Tue May 20 19:31:27 2025

@author: sambi
"""
from bs4 import BeautifulSoup, Comment
import requests
import pandas as pd
import numpy as np
import time
import random
import warnings
import os
import glob


# Headers to mimic a Google Chrome request
headers = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/98.0.4758.102 Safari/537.36'
    )
}


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


def sleep(min_sec: float, max_sec: float):
    """
    Sleeps asynchronously for a random duration that is generated from a log-normal distribution.

    Using a log-normal distribution provides delays that are not strictly uniform but instead 
    mimic human browsing behavior with occasional longer delays (a long tail effect). This increased 
    variability makes it less likely for anti-scraping systems to detect a regular pattern.

    The median value of the distribution is set to the average of min_sec and max_sec, but the generated 
    value is bounded within [min_sec, max_sec].

    Args:
        min_sec (float): Minimum number of seconds to sleep.
        max_sec (float): Maximum number of seconds to sleep.
    """
    # Compute the median of the distribution
    median = (min_sec + max_sec) / 2
    # For a log-normal, the median is exp(mu). We set mu = log(median)
    mu = np.log(median)
    # Sigma determines the variability. Adjust sigma (recommended between 0.5 and 1.0) for more or less variance.
    sigma = 0.75

    # Generate a sleep time from the log-normal distribution.
    sleep_time = random.lognormvariate(mu, sigma)

    print(f'Sleeping for {round(sleep_time,2)}')
    # Bound the sleep_time within [min_sec, max_sec]
    sleep_time = max(min_sec, min(sleep_time, max_sec))
    time.sleep(sleep_time)


save_path = './injury_report'


# Scraping
SECONDS_SLEEP = random.uniform(2, 30)

SEASON_DATES = ['2018', '2019', '2020', '2021', '2022', '2023']

SEASON_DATES = ['2024', '2025']
# SEASON_DATES = list(range(2016, 2024))

week = 1

for season in SEASON_DATES:

    if os.path.exists(f'{save_path}/{season}/') is False:
        os.makedirs(f'{save_path}/{season}/')

    url_boxscore = "https://www.footballdb.com/transactions/injuries.html?yr={season}&wk={week}&type=reg"
    print('\n=========================================================')
    print(f"Scraping injury reports from the {season} regular season")
    print('=========================================================')
    sleep(0., 20.)
    for week in range(1, 18):
        data_storage = []

        sleep(10., 60.)

        # BeautifulSoup object for a list of boxscores on a given day
        url_summaries = url_boxscore.format(season=season,
                                            week=week)

        print(f'Week {week}: url-{url_summaries}')
        response = requests.get(url_summaries, headers=headers)
        response.raise_for_status()  # Raises an error for bad responses
        soup_summaries = BeautifulSoup(response.text, 'lxml')
        tables = soup_summaries.find_all(
            'div', class_='divtable divtable-striped divtable-mobile')
        teams = soup_summaries.find_all('div', class_='teamsectlabel')
        teams = [team.get_text() for team in teams]

        t = 0
        weekly_data = []
        for table in tables:

            data = table.find_all('div', class_='tr')
            for row in data:
                # Extract player name and position
                player_info = row.find("div", class_="td w20")
                player_name = player_info.find("a").text
                position = player_info.text.split(")")[0].split("(")[-1]

                # Extract injury type
                injury_type = row.find(
                    "div", class_="td w15 d-none d-md-table-cell").text

                # Extract practice status
                practice_status = [div.text for div in row.find_all(
                    "div", style="width:50%;margin:0;padding-top:4px;float:left;")]

                # Extract game status
                game_status = row.find(
                    "div", class_="td w20 d-none d-md-table-cell").text

                if len(game_status.split()) == 1:
                    game_status = None
                else:
                    game_status = game_status.split()[1]

                data_storage.append({
                    "name": player_name,
                    "team": teams[t],
                    "position": position,
                    "injury": injury_type,
                    "Wed 09/05": practice_status[0].split(':')[-1].strip() if len(practice_status) > 0 else None,
                    "Thu 09/06": practice_status[1].split(':')[-1].strip() if len(practice_status) > 1 else None,
                    "Fri 09/07": practice_status[2].split(':')[-1].strip() if len(practice_status) > 2 else None,
                    "game_status": game_status
                })

            t += 1

        weekly_df = pd.DataFrame(data_storage)
        weekly_df.to_csv(f'{save_path}/{season}/week_{week}.csv')

# %%

# Define the folder path containing injury report CSV files
folder_path = "injury_report"

# Find all CSV files within the folder and subfolders
csv_files = glob.glob(os.path.join(folder_path, "**/*.csv"), recursive=True)

# Read all CSV files into a list of DataFrames and concatenate them into one DataFrame
final_df = pd.concat(
    [pd.read_csv(file, index_col=0) for file in csv_files],
    ignore_index=True
)

# Remove rows where 'game_status' is missing
final_df.dropna(subset=['game_status'], inplace=True)

# Filter data to include only relevant football positions
valid_positions = ['QB', 'RB', 'FB', 'WR', 'TE']
final_df = final_df[final_df['position'].isin(valid_positions)]

# Count occurrences of each "name" and "position" combination
injury_counts = final_df.groupby(
    ["name", "position"]).size().reset_index(name="count")

# Display the number of weeks injured per player
print('Weeks Injured')
print(injury_counts)

# Load the game data
games_df = pd.read_csv('basic_game_data/games/full_data.csv')

# Filter game data to include only relevant positions
games_df = games_df[games_df['position'].isin(valid_positions)]

# Count occurrences of each "player" and "position" combination
game_counts = games_df.groupby(
    ["player", "position"]).size().reset_index(name="count")

# Keep only players with more than 10 games played
game_counts = game_counts[game_counts['count'] > 10]

# Display the number of games played per player
print('Games Played')
print(game_counts)

# Calculate the injury rate (weeks injured / games played)
game_counts['injury_rate'] = round(
    injury_counts['count'].div(game_counts['count'], fill_value=0), 3
).clip(upper=1)  # Ensure rates are capped at 1

# Create the final DataFrame with player names, positions, and injury rates
df = game_counts[['player', 'position', 'injury_rate']]

# Replace NaN values with 0 in the injury rate column
df.fillna(0, inplace=True)

# Print and save the final dataset
print(df)
df.to_csv('injury_rates.csv', index=False)
