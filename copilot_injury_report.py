#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 19:31:27 2025

Revised on: [Insert Current Date]
Revised by: Senior Python Developer

This script scrapes injury reports from a football transactions website.
It mimics human browsing behavior by using random delays based on a log-normal 
distribution, and it logs the scraping process.

Usage:
    Simply run this module. The data for each season/week is saved as a CSV file.
"""

import logging
import os
import time
import random
import warnings
from pathlib import Path

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# Configure logging for better runtime messaging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Headers to mimic a Google Chrome request
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/98.0.4758.102 Safari/537.36'
    )
}

# Base URL template for scraping box scores/injury reports
URL_TEMPLATE = (
    "https://www.footballdb.com/transactions/injuries.html?yr={season}&wk={week}&type=reg"
)

# Base directory to store scraped CSV files
BASE_SAVE_PATH = Path('./injury_report')


def deprecated_function():
    """
    This is a deprecated function used as an illustration.
    """
    warnings.warn("deprecated", DeprecationWarning)


# Suppress deprecation warnings for the deprecated_function
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    deprecated_function()


def sleep_random(min_sec: float, max_sec: float):
    """
    Sleep for a random duration generated from a log-normal distribution.

    Using a log-normal distribution yields a delay pattern that mimics human browsing,
    which makes it harder for anti-scraping systems to detect a uniform pattern.

    The median of the distribution is set to the average of min_sec and max_sec.
    The final sleep time is clipped to be within the specified range.

    Args:
        min_sec (float): Minimum seconds to sleep.
        max_sec (float): Maximum seconds to sleep.
    """
    # Calculate the median and set mu accordingly
    median = (min_sec + max_sec) / 2
    mu = np.log(median)
    # Adjust this value (between 0.5 and 1.0) to control variability
    sigma = 0.75

    sleep_time = random.lognormvariate(mu, sigma)
    # Clip the sleep time to the range [min_sec, max_sec]
    sleep_time = max(min_sec, min(sleep_time, max_sec))
    logging.info(f"Sleeping for {round(sleep_time, 2)} seconds")
    time.sleep(sleep_time)


def fetch_injury_data(season: int, week: int, session: requests.Session) -> pd.DataFrame:
    """
    Fetch injury data for a given season and week.

    This function downloads HTML content using the provided session,
    parses it with BeautifulSoup, and extracts injury-related data.

    Args:
        season (int): Year of the season.
        week (int): Week number.
        session (requests.Session): A session object to perform HTTP requests.

    Returns:
        pd.DataFrame: A DataFrame containing injury data for the specified week.
    """
    url = URL_TEMPLATE.format(season=season, week=week)
    logging.info(f"Scraping URL: {url}")

    response = session.get(url, headers=HEADERS)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'lxml')

    # Extract team names; assume order matches tables order
    team_labels = soup.find_all('div', class_='teamsectlabel')
    teams = [team.get_text(strip=True) for team in team_labels]

    tables = soup.find_all(
        'div', class_='divtable divtable-striped divtable-mobile')
    weekly_data = []

    table_index = 0
    for table in tables:
        rows = table.find_all('div', class_='tr')
        for row in rows:
            # Extract player information
            player_info = row.find("div", class_="td w20")
            if player_info is None:
                continue
            anchor = player_info.find("a")
            player_name = anchor.text.strip() if anchor else None

            # Extract position information from text (from within parenthesis)
            # Example format: "Player Name (Position)"
            raw_text = player_info.get_text(strip=True)
            if "(" in raw_text and ")" in raw_text:
                position = raw_text.split("(")[-1].split(")")[0].strip()
            else:
                position = None

            # Extract injury type
            injury_div = row.find(
                "div", class_="td w15 d-none d-md-table-cell")
            injury_type = injury_div.text.strip() if injury_div else None

            # Extract practice status: there might be multiple status entries
            practice_divs = row.find_all("div",
                                         style="width:50%;margin:0;padding-top:4px;float:left;")
            practice_statuses = [
                div.get_text(strip=True).split(':')[-1].strip() for div in practice_divs
            ]

            # Extract game status
            game_status = row.find(
                "div", class_="td w20 d-none d-md-table-cell").text

            if len(game_status.split()) == 1:
                game_status = None
            else:
                game_status = game_status.split()[1]

            # Create record for the current row
            record = {
                "name": player_name,
                "team": teams[table_index] if table_index < len(teams) else None,
                "position": position,
                "injury": injury_type,
                "Wed 09/05": practice_statuses[0] if len(practice_statuses) > 0 else None,
                "Thu 09/06": practice_statuses[1] if len(practice_statuses) > 1 else None,
                "Fri 09/07": practice_statuses[2] if len(practice_statuses) > 2 else None,
                "game_status": game_status,
            }
            weekly_data.append(record)
        table_index += 1

    return pd.DataFrame(weekly_data)


def main():
    """
    Main function to iterate over seasons and weeks, scrape injury data,
    and save the results to CSV files.
    """
    # Define seasons and weeks
    # Using a range for seasons; adjust as needed.
    season_years = list(range(2016, 2024))
    weeks = range(1, 17)

    # Ensure the base save path exists
    BASE_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # Use a session for efficient HTTP connections
    with requests.Session() as session:
        for season in season_years:
            season_path = BASE_SAVE_PATH / str(season)
            season_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Scraping data for season: {season}")

            for week in weeks:
                # Sleep between weekly requests
                sleep_random(10.0, 60.0)

                try:
                    weekly_df = fetch_injury_data(season, week, session)
                    if weekly_df.empty:
                        logging.warning(
                            f"No data found for season {season} week {week}")
                    else:
                        csv_file = season_path / f"week_{week}.csv"
                        weekly_df.to_csv(csv_file, index=False)
                        logging.info(f"Saved data to {csv_file}")
                except requests.HTTPError as http_err:
                    logging.error(
                        f"HTTP error encountered for season {season} week {week}: {http_err}")
                except Exception as err:
                    logging.error(
                        f"Unexpected error for season {season} week {week}: {err}")


if __name__ == '__main__':
    main()
