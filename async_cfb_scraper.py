#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 19:31:27 2025
Author: sambi

Asynchronous sports stats scraper for sports-reference.com using configuration files.
This script scrapes college football boxscores across multiple seasons,
extracting passing and rushing statistics that are embedded inside HTML comments.
Data can be stored in memory and concatenated at the end or streamed immediately to CSV.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import warnings
import math
import aiohttp
import pandas as pd
from aiohttp_retry import RetryClient, ExponentialRetry
from bs4 import BeautifulSoup, Comment
import nest_asyncio

# Patch the event loop so that asyncio.run() can be safely invoked even
# if there's already a running event loop (e.g., in interactive environments).
nest_asyncio.apply()
# -----------------------------------------------------------------------------
# Setup logging and warning suppression
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# -----------------------------------------------------------------------------
# URL Templates (base parts)
# -----------------------------------------------------------------------------
URL_PARENT = "https://www.sports-reference.com"
URL_BOXSCORE_TEMPLATE = (
    "https://www.sports-reference.com/cfb/boxscores/index.cgi?"
    "month={month}&day={day}&year={season}&conf_id="
)

# -----------------------------------------------------------------------------
# Asynchronous Helper Functions
# -----------------------------------------------------------------------------


async def async_sleep(min_sec: float, max_sec: float):
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
    mu = math.log(median)
    # Sigma determines the variability. Adjust sigma (recommended between 0.5 and 1.0) for more or less variance.
    sigma = 0.75

    # Generate a sleep time from the log-normal distribution.
    sleep_time = random.lognormvariate(mu, sigma)
    # Bound the sleep_time within [min_sec, max_sec]
    sleep_time = max(min_sec, min(sleep_time, max_sec))

    logging.info(
        f"Sleeping for {sleep_time:.2f} seconds (using log-normal jitter)...")
    await asyncio.sleep(sleep_time)


async def fetch_soup(session: aiohttp.ClientSession, url: str) -> BeautifulSoup:
    """
    Asynchronously fetches a URL and returns a BeautifulSoup object.
    Uses the aiohttp_retry RetryClient to handle transient failures.
    """
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()
            html = await response.text()
            return BeautifulSoup(html, "lxml")
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None


def extract_game_links(soup: BeautifulSoup) -> list:
    """
    Extracts game (boxscore) links from the summary page soup by selecting
    anchor tags within divs with class 'game_summaries' that have 'boxscores' in the href.
    """
    if not soup:
        return []
    anchors = soup.select("div.game_summaries a[href*='boxscores']")
    return [anchor.get("href") for anchor in anchors if anchor.get("href")]


def parse_stats_from_comments(soup: BeautifulSoup, keyword: str) -> (list, list):
    """
    Parses a stats table from within HTML comments by searching for a comment
    that contains the specified keyword (e.g., 'passing' or 'rushing').

    First the function attempts to dynamically extract the table's column names
    from the <thead> section (preferring the second row if available). If that
    fails, it falls back on a default list of column names.

    Returns:
        tuple: (list of column headers, list of data rows)
    """
    comments = soup.find_all(text=lambda text: isinstance(text, Comment))
    table = None
    for comment in comments:
        if keyword in comment:
            comment_soup = BeautifulSoup(comment, "lxml")
            table = comment_soup.find("table")
            if table:
                break
    if not table:
        logging.warning(f"Table with keyword '{keyword}' not found.")
        return None, None

    # Attempt to extract header names dynamically.
    cols = []
    thead = table.find("thead")
    if thead:
        rows = thead.find_all("tr")
        # Use second row if available (sometimes the first row is a super-header)
        row_idx = 1 if len(rows) > 1 else 0
        cols = [th.get_text(strip=True) for th in rows[row_idx].find_all("th")]

    # If dynamic extraction fails, fall back on hard-coded column names.
    if not cols or len(cols) == 0:
        if keyword == "passing":
            cols = ['player', 'school', 'completions', 'attempts', 'completion_percentage',
                    'passing_yards', 'pass_yards_per_attempt', 'adjusted_pass_yards_per_attempt',
                    'pass_td', 'ints_thrown', 'passer_rating']
        elif keyword == "rushing":
            cols = ['player', 'school', 'rush_attempts', 'rush_yards', 'avg_rush_yards',
                    'rush_td', 'receptions', 'rec_yards', 'avg_rec_yards', 'rec_td',
                    'rush_rec_plays', 'tot_rush_rec_yards', 'avg_rush_rec_yards', 'total__rush_rec_tds']

    # Extract table body data.
    data = []
    tbody = table.find("tbody")
    if tbody:
        rows = tbody.find_all("tr", class_=None)
        for row in rows:
            player = row.find("th").get_text(strip=True)
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            data.append([player] + cells)
    return cols, data

# -----------------------------------------------------------------------------
# Data Writing Helper
# -----------------------------------------------------------------------------


def write_game_data_to_csv(df_game: pd.DataFrame, config: dict, csv_lock: asyncio.Lock):
    """
    Synchronously writes a single game's DataFrame to a CSV file.
    Uses the csv_lock to ensure only one task writes at a time.
    The first write writes the headers; subsequent writes append without the header.
    """
    csv_file = config.get("csv_output", "CFB_data.csv")
    mode = "w" if not os.path.exists(csv_file) else "a"
    header = True if mode == "w" else False
    try:
        df_game.to_csv(csv_file, mode=mode, header=header, index=False)
        logging.info(f"Wrote game data to {csv_file}")
    except Exception as e:
        logging.error(f"Error writing to CSV: {e}")

# -----------------------------------------------------------------------------
# Asynchronous Processing Functions
# -----------------------------------------------------------------------------


async def process_game(session: aiohttp.ClientSession, game_link: str,
                       data_storage: list, semaphore: asyncio.Semaphore,
                       config: dict, csv_lock: asyncio.Lock):
    """
    Processes an individual game page:
      • Waits asynchronously (non-blocking delay) before fetching.
      • Extracts passing and rushing stats from tables within HTML comments.
      • Attempts to merge these stats into a single DataFrame.
      • Either appends the DataFrame to in-memory storage or streams it directly to CSV.
    """
    game_url = URL_PARENT + game_link
    logging.info(f"Processing game URL: {game_url}")
    await async_sleep(config["min_sleep"], config["max_sleep"])
    async with semaphore:
        soup = await fetch_soup(session, game_url)
    if not soup:
        return

    pass_cols, pass_data = parse_stats_from_comments(soup, "passing")
    rush_cols, rush_data = parse_stats_from_comments(soup, "rushing")
    if pass_data is None or rush_data is None:
        logging.warning(f"Missing data at {game_url}")
        return
    try:
        df_pass = pd.DataFrame(pass_data, columns=pass_cols)
        df_rush = pd.DataFrame(rush_data, columns=rush_cols)
        # Merge passing and rushing data on 'player' and 'school'
        df_game = pd.merge(df_pass, df_rush, on=[
                           "player", "school"], how="outer")
        # Either stream data directly to CSV or store in memory.
        if config.get("stream_to_csv", False):
            async with csv_lock:
                # File operations are synchronous; they are minimal here.
                write_game_data_to_csv(df_game, config, csv_lock)
        else:
            data_storage.append(df_game)
        logging.info(f"Game data processed for {game_url}")
    except Exception as e:
        logging.error(f"Error merging data for {game_url}: {e}")


async def process_summary(session: aiohttp.ClientSession, season: int, month: int, day: int,
                          data_storage: list, semaphore: asyncio.Semaphore,
                          config: dict, csv_lock: asyncio.Lock):
    """
    Processes a summary page for the specified season, month, and day:
      • Waits asynchronously before fetching the page.
      • Extracts game links and schedules processing of each game concurrently.
    """
    url_summary = URL_BOXSCORE_TEMPLATE.format(
        season=season, month=month, day=day)
    logging.info(
        f"Processing summary for {day}-{month}-{season}: {url_summary}")
    await async_sleep(config["min_sleep"], config["max_sleep"])
    soup = await fetch_soup(session, url_summary)
    if not soup:
        logging.warning(f"Failed to fetch summary for {day}-{month}-{season}")
        return
    game_links = extract_game_links(soup)
    logging.info(
        f"Found {len(game_links)} game links for {day}-{month}-{season}")
    tasks = [process_game(session, link, data_storage, semaphore, config, csv_lock)
             for link in game_links]
    if tasks:
        await asyncio.gather(*tasks)

# -----------------------------------------------------------------------------
# Main Asynchronous Function (with configuration file support)
# -----------------------------------------------------------------------------


async def main_async(args):
    """
    Main asynchronous function:
      • Loads configuration parameters from a JSON file.
      • Creates a RetryClient (aiohttp‑retry) wrapped around the HTTP session.
      • Iterates over the season, month, and day ranges (from config) to schedule scraping tasks.
      • Either aggregates the results in memory or streams each game immediately to CSV.
    """
    # Load config from JSON file.
    with open(args.config, "r") as f:
        config = json.load(f)

    # Setup CSV lock for streaming writes if enabled.
    csv_lock = asyncio.Lock()

    # Create the list of seasons and date ranges from config.
    start_year = config.get("start_year", 2000)
    end_year = config.get("end_year", 2023)
    season_years = list(range(start_year, end_year + 1))
    month_range = config.get("month_range", [8, 9, 10, 11, 12])
    day_range = config.get("day_range", [1, 8, 15, 22, 29])
    min_sleep = config.get("min_sleep", 2)
    max_sleep = config.get("max_sleep", 30)
    # Ensure these sleep times are in config for use in async_sleep (used in process_game and process_summary)
    config["min_sleep"] = min_sleep
    config["max_sleep"] = max_sleep

    data_storage = []  # In-memory storage if streaming is not enabled.
    max_concurrent = config.get("max_concurrent", 5)
    semaphore = asyncio.Semaphore(max_concurrent)

    headers = {
        "User-Agent": config.get(
            "user_agent",
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/98.0.4758.102 Safari/537.36'

        )
    }

    retry_options = ExponentialRetry(
        attempts=5,
        statuses=(429, 500, 502, 503, 504)
    )
    async with aiohttp.ClientSession(headers=headers) as session:
        # Wrap the session with RetryClient for automatic retries.
        retry_client = RetryClient(
            client_session=session, retry_options=retry_options)
        summary_tasks = []
        for season in season_years:
            logging.info(f"Scraping season: {season}")
            await async_sleep(config.get("season_sleep_min", 5), config.get("season_sleep_max", 20))
            for month in month_range:
                for day in day_range:
                    summary_tasks.append(
                        process_summary(
                            retry_client, season, month, day, data_storage, semaphore, config, csv_lock)
                    )
        if summary_tasks:
            await asyncio.gather(*summary_tasks)
        await retry_client.close()

    # If not streaming to CSV, concatenate in-memory DataFrames and write at the end.
    if not config.get("stream_to_csv", False) and data_storage:
        game_df = pd.concat(data_storage, ignore_index=True)
        csv_file = config.get("csv_output", "CFB_data.csv")
        try:
            game_df.to_csv(csv_file, index=False)
            logging.info(f"All game data saved to {csv_file}")
        except Exception as e:
            logging.error(f"Error saving data to CSV: {e}")
    elif not data_storage:
        logging.info("No game data collected.")


def main():
    parser = argparse.ArgumentParser(
        description="Asynchronous Sports Stats Scraper with Configuration File"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to JSON configuration file (default: config.json)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
