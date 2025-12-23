# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 20:30:45 2024

@author: sambi
"""

import pandas as pd
import os

#curr_season = 2023
#os.chdir('basic_game_data')
df = pd.DataFrame()
for file in os.listdir('basic_game_data/games/seasons/'):

    if 'offence' in file and file.endswith('csv'):
        df_tmp = pd.read_csv(
            'basic_game_data/games/seasons/'+file, index_col=False)

        # Convert the 'date' to datetime
        df_tmp['date'] = pd.to_datetime(df_tmp['date'], format='%Y%m%d')
        df = pd.concat([df, df_tmp])

df.drop(['index', 'unnamed: 0'], axis=1, inplace=True)


def get_last_n_entries(df, group_col, n):
    """
    Retrieve the last n entries for each group in a DataFrame.

    This function groups the DataFrame by a specified column and extracts the last n entries 
    for each group. It then creates new columns for each of the last n values of the other 
    columns in the DataFrame, excluding the grouping column.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    group_col (str): The name of the column to group by.
    n (int): The number of last entries to retrieve for each group.

    Returns:
    pd.DataFrame: A new DataFrame containing the last n values for each column (except the 
                  grouping column) for each group defined by group_col. The new columns are 
                  named in the format '{column_name}_last_{i+1}' where i is the index of the 
                  last entry (0-based).
    """
    # Step 1: Group by the specified column and get the last n entries for each group
    last_n_df = df.groupby(group_col).tail(n)

    # Step 2: Sort the last n entries by date in ascending order
    sorted_last_n_df = last_n_df.sort_values(['date'], ascending=True)

    # Step 3: Initialize a dictionary to hold the aggregation expressions
    aggregation_dict = {}

    # Step 4: Create aggregation expressions for each column (excluding group_col)
    for col in df.columns:
        if col != group_col:
            for i in range(n):
                # Create a new column name for the last (i+1)th entry
                new_col_name = f"{col}_last_{i+1}"

                # Define the aggregation function with a default argument to capture the current value of i
                aggregation_dict[new_col_name] = (
                    col, lambda x, i=i: x.iloc[-(i+1)] if len(x) > i else None)

    # Step 5: Group by the specified column and apply the aggregation
    last_values_df = sorted_last_n_df.groupby(
        group_col).agg(**aggregation_dict).reset_index()

    # Step 6: Return the resulting DataFrame
    return last_values_df


dropped_data = df[['player', 'position', 'week',
                   'season', 'tm', 'against', 'score']]
df.drop(['tm', 'against', 'position'], axis=1, inplace=True)
df_full = pd.DataFrame()
seasons = os.listdir('basic_game_data')
for curr_season in [int(x) for x in seasons if x.isdigit()]:
    #curr_season = str(curr_season)
    df_avg_prev_season = df[df['season'] == curr_season -
                            1].drop('date', axis=1).groupby('player').mean().reset_index()
    df_avg_prev_season = df_avg_prev_season.rename(
        columns=lambda x: x + '_prev_season_avg' if x != 'player' else x)

    for week in range(df[df['season'] == curr_season].week.min(), df[df['season'] == curr_season].week.max()+1):
        print(f'Season:{curr_season} -- week:{week}')
        df_avg = df[(df['season'] <= curr_season) & (
            df['week'] < week)].groupby('player').mean().reset_index()
        df_avg = df_avg.rename(columns=lambda x: x +
                               '_career_avg' if x != 'player' else x)

        dt = df[(df['season'] <= curr_season) & (
            df['week'] < week)]['date'].max()
        # Group by 'name' and get the last 3 entries for each group
        last_3_df = df[df['date'] <= dt].groupby('player').tail(3)

        # last_3_df.drop(['date', 'week'], axis=1, inplace=True)

        # Calculate the average for the last 3 entries for each unique name
        averaged_last_3_df = last_3_df.drop(
            ['date', 'week'], axis=1).groupby('player').mean().reset_index()

        # Add a suffix to all columns except 'name'
        averaged_last_3_df = averaged_last_3_df.rename(
            columns=lambda x: x + '_avg_last_3_games' if x != 'player' else x)

        last_3_games = get_last_n_entries(last_3_df, 'player', n=3).fillna(0)
        # last_3_games['week'] = week
        '''
        df_weekly_data = pd.merge(
            [df_avg, last_3_games, df_avg_prev_season], on='player', how='inner')
        '''
        df_weekly_data = pd.merge(
            df_avg, last_3_games, on='player', how='outer')
        df_weekly_data = pd.merge(
            df_weekly_data, df_avg_prev_season, on='player', how='left')

        week_info = df[(df['season'] == curr_season) & (
            df['week'] == week)][['player', 'date']]

        df_weekly_data = pd.merge(
            df_weekly_data, week_info, on='player', how='left')
        df_weekly_data['week'] = week
        df_weekly_data['season'] = curr_season
        df_full = pd.concat([df_full, df_weekly_data])

    print('\n')
df_points = pd.merge(df_full, dropped_data, on=[
                     'player', 'week', 'season'])

df_points = df_points.sort_values(by=['player', 'date'])
df_points['rest_days'] = df_points.groupby('player')['date'].diff().dt.days

df_points = df_points.loc[:, ~df_points.columns.str.contains('date_')]
df_points = df_points.loc[:, ~df_points.columns.str.contains('season_')]
df_points.to_csv('basic_game_data//games//full_data.csv', index=False)

# !###
print('\nsetting data to predict')
df_full = pd.DataFrame()

curr_season = df.season.max()
df_avg_prev_season = df[df['season'] == curr_season -
                        1].drop('date', axis=1).groupby('player').median().reset_index()
df_avg_prev_season = df_avg_prev_season.rename(
    columns=lambda x: x + '_prev_season_avg' if x != 'player' else x)

week = df[df['season'] == curr_season].week.max()

print(f'Season:{curr_season} -- week:{week}')
df_avg = df[(df['season'] <= curr_season) & (
    df['week'] <= week)].groupby('player').median().reset_index()

df_avg = df_avg.rename(columns=lambda x: x +
                       '_career_avg' if x != 'player' else x)

dt = df[(df['season'] <= curr_season) & (
    df['week'] <= week)]['date'].max()
# Group by 'name' and get the last 3 entries for each group
last_3_df = df[df['date'] <= dt].groupby('player').tail(3)

# last_3_df.drop(['date', 'week'], axis=1, inplace=True)

# Calculate the average for the last 3 entries for each unique name
averaged_last_3_df = last_3_df.drop(
    ['date', 'week'], axis=1).groupby('player').median().reset_index()

# Add a suffix to all columns except 'name'
averaged_last_3_df = averaged_last_3_df.rename(
    columns=lambda x: x + '_avg_last_3_games' if x != 'player' else x)

last_3_games = get_last_n_entries(last_3_df, 'player', n=3).fillna(0)
# last_3_games['week'] = week
'''
df_weekly_data = pd.merge(
    [df_avg, last_3_games, df_avg_prev_season], on='player', how='inner')
'''
df_weekly_data = pd.merge(
    df_avg, last_3_games, on='player', how='outer')
df_weekly_data = pd.merge(
    df_weekly_data, df_avg_prev_season, on='player', how='left')

week_info = df[(df['season'] == curr_season) & (
    df['week'] == week)][['player', 'date']]

df_weekly_data = pd.merge(
    df_weekly_data, week_info, on='player', how='left')
df_weekly_data['week'] = week
df_weekly_data['season'] = curr_season
df_full = pd.concat([df_full, df_weekly_data])

print('\n')
df_points = pd.merge(df_full, dropped_data.drop(['score'], axis=1), on=[
                     'player', 'week', 'season'])

df_points = df_points.sort_values(by=['player', 'date'])
df_points['rest_days'] = df_points.groupby('player')['date'].diff().dt.days

df_points = df_points.loc[:, ~df_points.columns.str.contains('date_')]
df_points = df_points.loc[:, ~df_points.columns.str.contains('season_')]
df_points.to_csv('basic_game_data//games//to_predict.csv', index=False)
