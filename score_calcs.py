# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:32:58 2024

@author: sambi
"""
import pandas as pd
import os
import numpy as np


def calc_score(file_name):
    df = pd.read_csv(file_name, index_col=False)
    try:
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
    except:
        pass
    if 'offence' in file_name:
        scores = []
        for i, row in df.iterrows():
            pass_score = row['Pass Yds']/25 + \
                row['Pass TD']*6+row['Int']*-2+row['Sack']*-1

            if row['Pass Yds'] >= 300:
                pass_score += 2
            if row['Pass Yds'] >= 400:
                pass_score += 2

            rush_score = row['Rush Yds']/10+row['Rush TD']*6
            if row['Rush Yds'] >= 100:
                rush_score += 2
            if row['Rush Yds'] >= 200:
                rush_score += 2

            rec_score = row['Rec']*0.5+row['Rec TD']*6+row['Rec Yds']/10
            if row['Rec Yds'] >= 100:
                rush_score += 2
            if row['Rec Yds'] >= 200:
                rush_score += 2
            try:
                ret_score = row['punt yards']/10+row['punt td']*6+row['kick return yards']/10 + \
                    row['kickoff td']*6
            except KeyError:
                ret_score = 0
            score = pass_score+rush_score+rec_score + \
                ret_score-2*row['Fumble Lost']

            scores.append(score)

        df['score'] = scores
    elif 'defence' in file_name:
        if 'DEF' not in df.columns:
            df = create_defence(df)
        scores = []
        for i, row in df.iterrows():
            try:
                score = row['kickoff td']*6+row['punt td']*6
            except KeyError:
                score = 0
            if row['Position'] == 'DEF':
                score += row['Fumble Return TD']*6+row['Sacks']+row['Interceptions']*2 + \
                    row['Intercept. Ret. TD']*6+row['Fumbles Recovered']*2
                try:
                    score += row['kickoff td']*6+row['punt td']*6
                except KeyError:
                    pass
                if row['points allowed'] == 0:
                    score += 10
                elif row['points allowed'] > 1 and row['points allowed'] <= 6:
                    score += 7
                elif row['points allowed'] > 6 and row['points allowed'] <= 13:
                    score += 4
                elif row['points allowed'] > 13 and row['points allowed'] <= 20:
                    score += 1
                elif row['points allowed'] > 20 and row['points allowed'] <= 27:
                    score += 0
                elif row['points allowed'] > 27 and row['points allowed'] <= 34:
                    score += -1
                elif row['points allowed'] > 34:
                    score += -4

                if row['points allowed'] < 100:
                    score += 10
                elif row['points allowed'] >= 100 and row['points allowed'] < 200:
                    score += 7
                elif row['points allowed'] >= 200 and row['points allowed'] < 300:
                    score += 4
                elif row['points allowed'] >= 300 and row['points allowed'] < 400:
                    score += 1
            else:

                score += row['Tackles Solo']+row['Assists']+row['Sacks']*2+row['Interceptions']*2 + \
                    row['Fumbles Forced']*2+row['Fumbles Recovered'] + \
                    row['Fumble Return TD']*6 + \
                    row['Intercept. Ret. TD']*6+row['Passes Defended']

                if row['Tackles Combined'] >= 10:
                    score += 2

                if row['Sacks'] >= 2:
                    score += 2

            scores.append(score)

        df['score'] = scores

    return df


def fix_teams(file_name):

    df = pd.read_csv(file_name, index_col=False)
    if any(item in file_name for item in ['offence', 'defence', 'kicking']):
        host = file_name.split('_')[0].upper()
        teams = {'ATL': 'FALCONS', 'BUF': 'BILLS', 'CAR': 'PANTHERS', 'CHI': 'BEARS',
                 'CIN': 'BENGALS', 'CLE': 'BROWNS', 'CLT': 'COLTS', 'CRD': 'CARDINALS',
                 'DAL': 'COWBOYS', 'DEN': 'BRONCOS', 'DET': 'LIONS', 'GNB': 'PACKERS',
                 'HOU': 'TEXANS', 'JAX': 'JAGUARS', 'KAN': 'CHIEFS', 'LAC': 'CHARGERS',
                 'MIA': 'DOLPHINS', 'MIN': 'VIKINGS', 'NOR': 'SAINTS', 'NWE': 'PATRIOTS',
                 'NYG': 'GIANTS', 'NYJ': 'JETS', 'PHI': 'EAGLES', 'PIT': 'STEELERS',
                 'RAI': 'RAIDERS', 'RAM': 'RAMS', 'RAV': 'RAVENS', 'SEA': 'SEAHAWKS',
                 'SFO': '49ERS', 'TAM': 'BUCCANEERS', 'TEN': 'TITANS',
                 'WAS': 'COMMANDERS', 'OAK': 'RAIDERS', 'OTI': 'TITANS', 'HTX': 'TEXANS',
                 'STL': 'RAMS', 'SDG': 'CHARGERS', 'BUC': 'BUCCANEERS'}
        host = teams[host]
        try:
            df['Tm'] = df['Tm'].map(teams)
            df['home'] = np.where(df['Tm'].str.upper() == host, 1, 0)
        except KeyError:
            df['tm'] = df['tm'].map(teams)
            df['home'] = np.where(df['tm'].str.upper() == host, 1, 0)

    return df


def create_defence(df):
    # Define aggregation functions
    agg_funcs = {}
    for col in df.columns:
        if 'average' in col or 'per' in col or 'avg ' in col or 'rate ' in col:
            # Average for columns with 'average' in their name
            agg_funcs[col] = 'mean'
        elif 'long' in col or 'home' in col or 'week' in col or 'date' in col or 'points' in col or col == 'yards allowed':
            agg_funcs[col] = 'max'
        else:
            agg_funcs[col] = 'sum'    # Sum for all other columns
    agg_funcs.pop('tm')
    # Group by 'team' and aggregate
    result = df.groupby('tm').agg(agg_funcs).reset_index()
    result['Position'] = 'DEF'
    result['player'] = result['tm']
    df = pd.concat([df, result])
    return df


for file in os.listdir():
    if file.endswith('.csv'):
        df = calc_score(file)

        unique_columns = {col.lower(): col for col in df.columns}
        df.rename(columns=dict(unique_columns), inplace=True)
        df.to_csv(file)
'''
filename = 'atl_vs_car_2018_defence.csv'
for file in os.listdir():
    if file.endswith('.csv'):
        df = pd.read_csv(filename)
        df_tmp = df[['points allowed', 'yards allowed', 'date', 'home', 'score']]
        df2 = df.drop(columns=['points allowed', 'yards allowed',
                      'date', 'home', 'score']).groupby('tm').sum()
        df2.reset_index(inplace=True)
        print(df2)
        raise
'''

for season in ['2024', '2025']:
    for pos in ['offence', 'defence']:
        df_main = pd.DataFrame()
        for file in os.listdir():
            if season in file and pos in file:
                df_tmp = pd.read_csv(file, index_col=False)
                df_main = pd.concat([df_main, df_tmp])

        try:
            os.makedirs('games/seasons')
        except:
            pass

        df_main_rsk = pd.DataFrame()

        for name in df_main.player.unique():
            df_tmp = df_main[df_main['player'] == name].sort_values(
                by="date").reset_index()
            df_tmp['avg_score'] = df_tmp['score'].rolling(3).mean()
            df_tmp['risk'] = df_tmp['score'].rolling(3).std()
            df_main_rsk = pd.concat([df_main_rsk, df_tmp])

        df_main_rsk.columns = [x.lower() for x in df_main_rsk.columns]
        df_main_rsk = df_main_rsk.loc[:, ~
                                      df_main_rsk.columns.duplicated()].copy()

        df_main_rsk.to_csv('games/seasons/'+season+'_'+pos+'.csv')
