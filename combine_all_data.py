
"""
Script to process college football combine data, calculate college and NFL rookie scores,
and merge datasets for analysis.

Features:
- Combines multiple CSV files into DataFrames.
- Calculates fantasy scores for college and NFL players.
- Converts height from feet-inches to centimetres.
- Provides logging and print options for monitoring progress.

Created on Wed Dec 17 13:08:44 2025
Author: SHB6
"""

import os
import glob
import pandas as pd
import logging

# -----------------------------
# Configuration
# -----------------------------
COMBINE_DIR = 'cfb_combine/'
COLLEGE_DIR = 'cfb_data/'
NFL_DIR = 'cfb_nfl/'
USE_LOGGING = True  # Set to False to disable logging
LOG_FILE = 'process.log'

# -----------------------------
# Logging Setup
# -----------------------------
if USE_LOGGING:
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging started for data processing script.")


def log_and_print(message: str, level: str = "info"):
    """
    Log and/or print messages based on configuration.

    Args:
        message (str): Message to display.
        level (str): Logging level ('info', 'debug', 'error').

    Returns:
        None
    """
    if USE_LOGGING:
        getattr(logging, level)(message)
    print(message)


# -----------------------------
# Helper Functions
# -----------------------------


def list_csv_files(directory: str) -> list:
    """
    List all CSV files in a given directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        list: List of file paths for CSV files.
    """
    files = glob.glob(os.path.join(directory, '*.csv'))
    log_and_print(f"Found {len(files)} files in {directory}")
    return files


def load_and_combine_csv(directory: str) -> pd.DataFrame:
    """
    Load all CSV files from a directory and combine them into a single DataFrame.

    Args:
        directory (str): Path to the directory containing CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame containing all rows from the CSV files.
    """
    files = list_csv_files(directory)
    combined_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    log_and_print(f"Combined {len(files)} files from {directory}")
    return combined_df


def college_score(df: pd.DataFrame) -> list:
    """
    Calculate college football fantasy scores for each player.

    Scoring rules:
    - Passing: 1 point per 25 yards, 6 points per TD, -2 per interception.
      Bonus: +2 for 300+ yards, +2 for 400+ yards.
    - Rushing: 1 point per 10 yards, 6 points per TD.
      Bonus: +2 for 100+ yards, +2 for 200+ yards.
    - Receiving: 0.5 points per reception, 6 points per TD, 1 point per 10 yards.
      Bonus: +2 for 100+ yards, +2 for 200+ yards.

    Args:
        df (pd.DataFrame): DataFrame containing player stats with columns:
            'passing_yards', 'pass_td', 'ints_thrown', 'rush_yards', 'rush_td',
            'receptions', 'rec_td', 'rec_yards'.

    Returns:
        list: List of calculated fantasy scores for each row in the DataFrame.
    """
    scores = []
    for _, row in df.iterrows():
        pass_score = row.get('passing_yards', 0) / 25 + row.get('pass_td', 0) * 6 - row.get('ints_thrown', 0) * 2
        if row.get('passing_yards', 0) >= 300:
            pass_score += 2
        if row.get('passing_yards', 0) >= 400:
            pass_score += 2

        rush_score = row.get('rush_yards', 0) / 10 + row.get('rush_td', 0) * 6
        if row.get('rush_yards', 0) >= 100:
            rush_score += 2
        if row.get('rush_yards', 0) >= 200:
            rush_score += 2

        rec_score = row.get('receptions', 0) * 0.5 + row.get('rec_td', 0) * 6 + row.get('rec_yards', 0) / 10
        if row.get('rec_yards', 0) >= 100:
            rec_score += 2
        if row.get('rec_yards', 0) >= 200:
            rec_score += 2

        scores.append(pass_score + rush_score + rec_score)
    return scores


def calc_nfl_score(file_name: str) -> pd.DataFrame:
    """
    Calculate NFL rookie scores from a given CSV file.

    Scoring rules:
    - Passing: 1 point per 25 yards, 6 points per TD, -2 per interception, -1 per sack.
      Bonus: +2 for 300+ yards, +2 for 400+ yards.
    - Rushing: 1 point per 10 yards, 6 points per TD.
      Bonus: +2 for 100+ yards, +2 for 200+ yards.
    - Receiving: 0.5 points per reception, 6 points per TD, 1 point per 10 yards.
      Bonus: +2 for 100+ yards, +2 for 200+ yards.
    - Returns: 1 point per 10 yards, 6 points per TD.
    - Fumbles: -2 points per lost fumble.

    Args:
        file_name (str): Path to the CSV file containing NFL stats.

    Returns:
        pd.DataFrame: DataFrame with an additional 'score' column for calculated fantasy scores.
    """
    df = pd.read_csv(file_name)
    df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True, errors='ignore')

    if 'offence' in file_name:
        scores = []
        for _, row in df.iterrows():
            pass_score = row.get('Pass Yds', 0) / 25 + row.get('Pass TD', 0) * 6 - row.get('Int', 0) * 2 - row.get('Sack', 0)
            if row.get('Pass Yds', 0) >= 300:
                pass_score += 2
            if row.get('Pass Yds', 0) >= 400:
                pass_score += 2

            rush_score = row.get('Rush Yds', 0) / 10 + row.get('Rush TD', 0) * 6
            if row.get('Rush Yds', 0) >= 100:
                rush_score += 2
            if row.get('Rush Yds', 0) >= 200:
                rush_score += 2

            rec_score = row.get('Rec', 0) * 0.5 + row.get('Rec TD', 0) * 6 + row.get('Rec Yds', 0) / 10
            if row.get('Rec Yds', 0) >= 100:
                rec_score += 2
            if row.get('Rec Yds', 0) >= 200:
                rec_score += 2

            ret_score = row.get('punt yards', 0) / 10 + row.get('punt td', 0) * 6 + row.get('kick return yards', 0) / 10 + row.get('kickoff td', 0) * 6
            score = pass_score + rush_score + rec_score + ret_score - 2 * row.get('Fumble Lost', 0)
            scores.append(score)

        df['score'] = scores
    log_and_print(f"Processed NFL file: {file_name}")
    return df


def convert_height_to_cm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert height from feet-inches format to centimetres.

    Args:
        df (pd.DataFrame): DataFrame containing 'Ht' column in format 'X-Y' (feet-inches).

    Returns:
        pd.DataFrame: Updated DataFrame with 'Ht' converted to centimetres.
    """
    df['Ht'] = df['Ht'].fillna('')
    df['feet'] = df['Ht'].apply(lambda x: int(str(x).split('-')[0]) if '-' in str(x) else 0)
    df['inches'] = df['Ht'].apply(lambda x: int(str(x).split('-')[1]) if '-' in str(x) else 0)
    df['Ht'] = df['feet'] * 30.48 + df['inches'] * 2.54
    log_and_print("Converted height to cm.")
    return df.drop(columns=['feet', 'inches'])


# -----------------------------
# Main Processing
# -----------------------------

log_and_print("Starting data processing...")

# Load and clean combine data
combine_df = load_and_combine_csv(COMBINE_DIR)
# ensure that we only care about offence players- defence not recorded well.
combine_df = combine_df[combine_df['position'].isin(['QB', 'WR', 'RB', 'FB', 'TE'])]
# drop any players without names
combine_df.dropna(subset=['player', 'yr'], inplace=True)
combine_df['yr'] = combine_df['yr'].astype(int)
combine_df.reset_index(drop=True, inplace=True)

# if there are any nans, for example a player didnt participate in that commbine drill add extreme values
combine_df.fillna({'40yd': 100, 'Vertical': 0, 'Bench': 0, 'Broad Jump': 0, '3Cone': 100, 'Shuttle': 100}, inplace=True)

# Load and process college data
college_df = pd.DataFrame()
for f in list_csv_files(COLLEGE_DIR):
    tmp = pd.read_csv(f)
    tmp.drop(columns=[col for col in tmp.columns if 'unnamed' in col.lower()], inplace=True)
    tmp = tmp[tmp['player'].isin(combine_df['player'].unique())]
    tmp['college_score'] = college_score(tmp)
    college_df = pd.concat([college_df, tmp], ignore_index=True)

# there are multiple josh johnsons across the data its confusing so remove them for simplicity.
college_df = college_df[college_df['player'] != 'Josh Johnson']

# Aggregate college stats
df_total = college_df.groupby('player').sum(numeric_only=True)
df_total = df_total.loc[:, ~df_total.columns.str.contains('avg')]
df_total.drop(['pass_yards_per_attempt', 'adjusted_pass_yards_per_attempt', 'completion_percentage', 'passer_rating'], axis=1, inplace=True)

df_avg = college_df.groupby('player').median(numeric_only=True)
df_avg = df_avg.loc[:, ~df_avg.columns.str.contains('tot_')]
df_avg.drop(['total__rush_rec_tds'], axis=1, inplace=True)

# Merge combine and college data
df = combine_df.merge(df_total, on='player', how='left', 
                      suffixes=('_combine', '_totals'))

df= df.merge(df_avg, on='player', how='left', suffixes=('_total', '_avg'))

# Process NFL rookie data
nfl_rookie_df = pd.DataFrame()
for f in glob.glob(os.path.join(NFL_DIR, '**/*.csv'), recursive=True):
    if 'offence' in f:
        year = int(f.split('_')[-2])
        rookies = combine_df[combine_df['yr'] == year]['player'].unique()
        tmp = calc_nfl_score(f)
        tmp = tmp[tmp['player'].isin(rookies)]
        nfl_rookie_df = pd.concat([nfl_rookie_df, tmp], ignore_index=True)

nfl_rookie_df_sum = nfl_rookie_df.groupby('player').median(numeric_only=True).reset_index()
df_final = df.merge(nfl_rookie_df_sum[['player', 'score']], on='player', how='left').rename(columns={'score': 'nfl_score'})
df_final.drop(['tm', 'yr'], axis=1, inplace=True)
df_final['nfl_score'] = df_final['nfl_score'].fillna(0)
df_final = df_final.dropna(subset=['nfl_score']).reset_index(drop=True)

# Convert height to cm
df_final = convert_height_to_cm(df_final)

log_and_print(f"Final DataFrame columns:\n{df_final.columns}")

df_final.to_csv('college_and_nfl_dataset.csv')