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


SEASON_DATES = ['2018', '2019', '2020', '2021', '2022', '2023']

# SEASON_DATES = ['2024']
# start at 2000
SEASON_DATES = list(range(2000, 2026))

week = 1


url_parent = "https://www.sports-reference.com"
url_boxscore = "https://www.sports-reference.com/cfb/boxscores/index.cgi?month={month}&day={day}&year={season}&conf_id="

data_storage = []
game_df = pd.DataFrame()

for season in SEASON_DATES:

    print("Scraping boxscores from the {} regular season".format(season))
    sleep(0., 20.)
    for month in range(8, 13):

        sleep(5., 40.)
        for day in range(1, 31, 7):
            if os.path.exists(f'cfb_data/{season}/') is False:
                os.makedirs(f'cfb_data/{season}/')

            if os.path.exists(f'cfb_data/{season}/CFB_data-{day}-{month}-{season}.csv'):
                print(
                    f'\nThe data for this week exists: {day}/{month}/{season}\n')
                continue

            sleep(10., 60.)

            # BeautifulSoup object for a list of boxscores on a given day
            url_summaries = url_boxscore.format(season=season,
                                                month=month, day=day)
            print(f'Date: {day}-{month}-{season}\n')
            print('url-', url_summaries)
            response = requests.get(url_summaries, headers=headers)
            response.raise_for_status()  # Raises an error for bad responses
            soup_summaries = BeautifulSoup(response.text, 'lxml')
            tables = soup_summaries.find_all(
                'div', class_='game_summaries')

            games = [game.find_all('a') for game in tables]

            games2 = [item for sublist in games for item in sublist]

            links = [game.get('href')
                     for game in games2 if game.has_attr('href')]

            game_links = [x for x in links if 'boxscores' in x]
            for link in game_links:
                print(link)

            print('--------------\n')
            for link in game_links:
                try:

                    sleep(5, 30)

                    url = url_parent+link
                    print(f'\n\n=========================\n{url}')
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()  # Raises an error for bad responses
                    soup = BeautifulSoup(response.text, 'lxml')

                    # some tables are hidden in comments
                    # get the data from comments and then select the range which contains tables
                    comments = soup.findAll(
                        text=lambda text: isinstance(text, Comment))[0:-23]
                    comments = [BeautifulSoup(table, 'lxml').find_all('table')
                                for table in comments]
                    try:
                        passing = [
                            x for x in comments if 'passing' in str(x)][0][0]

                        cols_passing = [th.get_text() for th in passing.find(
                            'thead').find_all('tr')[1].find_all('th')][:]

                        cols_passing = ['player', 'school', 'completions', 'attempts',
                                        'completion_percentage', 'passing_yards',
                                        'pass_yards_per_attempt',
                                        'adjusted_pass_yards_per_attempt',
                                        'pass_td', 'ints_thrown', 'passer_rating']

                        passing = passing.find(
                            'tbody').find_all('tr', class_=None)

                        passing_data = []
                        for i in passing:
                            name = i.find('th').get_text()
                            passing_data.append([name]+[td.get_text()
                                                for td in i.find_all('td')])

                        rush = [
                            x for x in comments if 'rushing' in str(x)][0][0]

                        cols_rushing = [th.get_text() for th in rush.find(
                            'thead').find_all('tr')[1].find_all('th')][:]

                        cols_rushing = ['player', 'school', 'rush_attempts', 'rush_yards',
                                        'avg_rush_yards', 'rush_td', 'receptions',
                                        'rec_yards', 'avg_rec_yards', 'rec_td',
                                        'rush_rec_plays', 'tot_rush_rec_yards',
                                        'avg_rush_rec_yards', 'total__rush_rec_tds']

                        rush = rush.find(
                            'tbody').find_all('tr', class_=None)
                        rush_data = []

                        for i in rush:
                            name = i.find('th').get_text()
                            rush_data.append([name]+[td.get_text()
                                                     for td in i.find_all('td')])

                        print('###############################')
                        print(cols_passing)
                        print(passing_data)

                        print('-----------------\n')
                        print(cols_rushing)
                        print(rush_data)

                        df_pass = pd.DataFrame(
                            passing_data, columns=cols_passing)

                        df_rush = pd.DataFrame(rush_data, columns=cols_rushing)
                        df_game = df_pass.merge(
                            df_rush, on=['player', 'school'], how='outer')
                        data_storage.append(df_game)
                        game_df = pd.concat([game_df, df_game])
                    except IndexError:
                        print('No Data')

                except requests.exceptions.HTTPError as errh:
                    print('######################')
                    print(errh.args[0])
                    print('######################')

            if len(game_df) > 0:
                game_df.to_csv(
                    f'cfb_data/{season}/CFB_data-{day}-{month}-{season}.csv')

    # Combine all games in a year into a single dataframe
    season_games = []
    # Read all files into a list
    for file in os.listdir(f'cfb_data/{season}'):
        if not file.endswith('.csv'):
            continue
        if str(season) in file:
            season_games.append(pd.read_csv(f'cfb_data/{season}/{file}'))

    # Combine all files into a single dataframe
    final_df = pd.concat(season_games)
    # Save dataframe for the season
    final_df.to_csv(f'cfb_data/college_season_{season}.csv')
