# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:31:24 2024

@author: sambi
"""


from bs4 import BeautifulSoup, Comment
from urllib.request import urlopen
import pandas as pd
import numpy as np
import time
import os
import random
import warnings
import datetime

warnings.filterwarnings("ignore")

basic_dir = 'basic_game_data'


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
                 'STL': 'RAMS', 'SDG': 'CHARGERS', 'BAL': 'RAVENS', 'ARI': 'CARDINALS', 'LAR': 'RAMS',
                 'IND': 'COLTS'}
        host = teams[host]
        try:
            df['tm'] = df['tm'].map(teams)
            df['home'] = np.where(df['tm'].str.upper() == host, 1, 0)
        except KeyError:
            df['tm'] = df['tm'].map(teams)
            df['home'] = np.where(df['tm'].str.upper() == host, 1, 0)

    return df


def swap_values(df, column, value1, value2, new_column):
    """
    Swap two specified values in a given column and store the result in a new column.

    Args:
    df (pd.DataFrame): The DataFrame to modify.
    column (str): The column containing values to swap.
    value1 (str): First value to swap.
    value2 (str): Second value to swap.
    new_column (str): Name of the new column to store swapped values.

    Returns:
    pd.DataFrame: DataFrame with new swapped column.
    """
    mapping = {value1: value2, value2: value1}
    df[new_column] = df[column].map(mapping)
    return df


teams = {'ATL': 'FALCONS', 'BUF': 'BILLS', 'CAR': 'PANTHERS', 'CHI': 'BEARS',
         'CIN': 'BENGALS', 'CLE': 'BROWNS', 'CLT': 'COLTS', 'CRD': 'CARDINALS',
         'DAL': 'COWBOYS', 'DEN': 'BRONCOS', 'DET': 'LIONS', 'GNB': 'PACKERS',
         'HOU': 'TEXANS', 'JAX': 'JAGUARS', 'KAN': 'CHIEFS', 'LAC': 'CHARGERS',
         'MIA': 'DOLPHINS', 'MIN': 'VIKINGS', 'NOR': 'SAINTS', 'NWE': 'PATRIOTS',
         'NYG': 'GIANTS', 'NYJ': 'JETS', 'PHI': 'EAGLES', 'PIT': 'STEELERS',
         'RAI': 'RAIDERS', 'RAM': 'RAMS', 'RAV': 'RAVENS', 'SEA': 'SEAHAWKS',
         'SFO': '49ERS', 'TAM': 'BUCCANEERS', 'TEN': 'TITANS',
         'WAS': 'COMMANDERS', 'OAK': 'RAIDERS', 'OTI': 'TITANS', 'HTX': 'TEXANS',
         'STL': 'RAMS', 'SDG': 'CHARGERS', 'BAL': 'RAVENS', 'ARI': 'CARDINALS', 'LAR': 'RAMS',
         'IND': 'COLTS'}


# Scraping
SECONDS_SLEEP = random.uniform(2, 30)


SEASON_DATES = ['2023', '2022', '2021', '2020', '2019', '2018', '2024']

SEASON_DATES = ['2025']
week = 1
current_year = datetime.datetime.now().year
#MIN DATE 2012
for season in range(2024, current_year+1):

    if os.path.exists(basic_dir+'//'+str(season)) is False:
        os.makedirs(basic_dir+'//'+str(season))
    url_parent = "https://www.pro-football-reference.com"
    url_boxscore = "https://www.pro-football-reference.com/years/{season}/week_{week}.htm"

    print(f"Scraping boxscores from the {season} regular season")

    for date in range(1, 17):
        sleep = random.uniform(2, 30)
        time.sleep(sleep)
        # BeautifulSoup object for a list of boxscores on a given day
        url_summaries = url_boxscore.format(season=season, week=date)
        print('url-', url_summaries)
        soup_summaries = BeautifulSoup(urlopen(url_summaries), 'lxml')
        games = soup_summaries.find_all(
            'div', class_='game_summary expanded nohover')
        # print('html\n',games)
        # date=11
        for game in games:
            sleep = random.uniform(2, 30)
            time.sleep(sleep)

            summary = {}

            host = game.find_all('table')[0].find_all('a')[
                0]['href'][7:10].upper()

            try:
                winner = game.find('tr', class_='winner').find_all('td')
                loser = game.find('tr', class_='loser').find_all('td')
            except:
                tms = game.find_all('tr', class_='draw')
                winner = tms[0].find_all('td')
                loser = tms[1].find_all('td')
                # print(tms)

                # print(f'\n\nWinner-{winner}\nLoser-{loser}')
            summary['winner'] = [winner[0].find(
                'a')['href'][7:10], int(winner[1].get_text())]
            summary['loser'] = [loser[0].find(
                'a')['href'][7:10], int(loser[1].get_text())]
            points = {value[0].upper(): value[1]
                      for key, value in summary.items() if len(value) == 2}
            # print('\n\n',summary)

            url_game = url_parent+game.find_all('a')[1]['href']
            print('\n\n\ngame url-', url_game)
            game_date = url_game.split('/')[-1][0:8]
            print('Week-', date, 'game date-', game_date)
            host = url_game[-7:-4].upper()

            if host == 'STL':
                host = 'RAM'
            elif host == 'HTX':
                host = 'HOU'
            elif host == 'OTI':
                host = 'TEN'

            for tm in ['winner', 'loser']:
                if summary[tm][0] == 'oti':
                    summary[tm][0] = 'ten'
                elif summary[tm][0] == 'htx':
                    summary[tm][0] = 'hou'
                elif summary[tm][0] == 'stl':
                    summary[tm][0] = 'ram'
                elif summary[tm][0] == 'sdg':
                    summary[tm][0] = 'lac'

            print('Host-', host)
            # print(game)
            soup_game = BeautifulSoup(urlopen(url_game), 'lxml')

            # gets the columns for each tables
            o_tables = soup_game.find_all('table')[:]
            '''
                print('\n\n\n\n\n\nTABLES')
                for table in o_tables:
                    print('\n\ntable\n', table)
                '''

            # some tables are hidden in comments, get the data from comments and then select the range which contains tables
            comments = soup_game.findAll(
                text=lambda text: isinstance(text, Comment))[30:-23]
            f = 0

            tables = [BeautifulSoup(table, 'lxml').find_all('table')
                      for table in comments]

            # removes all empty comments
            tables = [x for x in tables if x != []]
            # tables = [x for x in tables if len(x[0].find('thead').find_all('tr')) > 1] #removes all data which isnt a table
            # print('\n\n\nAll tables')

            tables = o_tables[2:]+tables[1:]

            general_stats = [
                x for x in tables if 'team_stats' in str(x)][0][0]
            # print(general_stats)
            columns_gen = [th.get_text() for th in general_stats.find(
                'thead').find('tr').find_all('th')][:]
            # print(columns_gen)
            general_stats = general_stats.find(
                'tbody').find_all('tr', class_=None)
            gen_stats_data = []
            for i in general_stats:
                name = i.find('th').get_text()
                gen_stats_data.append([name]+[td.get_text()
                                      for td in i.find_all('td')])
            # print(gen_stats_data)
            gen_stats = pd.DataFrame(gen_stats_data, columns=columns_gen)

            i = 0
            table_names = ['player_offence', 'player_defense', 'returns', 'kicking',
                           'passing_advanced', 'rushing_advanced', 'receiving_advanced', 'defense_advanced']
            basic_offence = [
                x for x in tables if 'player_offense' in str(x)][0]
            columns_offence = [th.get_text() for th in basic_offence.find(
                'thead').find_all('tr')[1].find_all('th')][:]
            columns_offence = ['player',	'tm',	'Completion',	'Pass Att',	'Pass Yds',	'Pass TD',
                               'Int',	'Sack',	'Yds Lost',	'Pass Long',	'Passer Rating',	'Rush Att',
                               'Rush Yds',	'Rush TD',	'Rush Long',	'Tgt',	'Rec',	'Rec Yds',	'Rec TD',
                               'Rec Long',	'Fumble Rec',	'Fumble Lost']
            # print(columns_offence)
            basic_offence = basic_offence.find(
                'tbody').find_all('tr', class_=None)
            basic_offence_data = []
            for i in basic_offence:
                name = i.find('th').get_text()
                basic_offence_data.append([name]+[td.get_text()
                                          for td in i.find_all('td')])
            # print(basic_offence_data)

            basic_defence = [
                x for x in tables if 'player_defense' in str(x)][0][0]
            # print(basic_defence)
            columns_defence = [th.get_text() for th in basic_defence.find(
                'thead').find_all('tr')[1].find_all('th')][:]
            columns_defence = ['player', 'tm',	'Interceptions',	'Intercept. Ret. Yds',	'Intercept. Ret. TD',
                               'Long Intercep. Return',	'Passes Defended',	'Sacks',	'Tackles Combined',
                               'Tackles Solo',	'Assists',	'Tackles For Loss',	'QB Hits', 'Fumbles Recovered',
                               'Fumble Return Yds',	'Fumble Return TD',	'Fumbles Forced']
            # print(columns_defence)
            basic_defence = basic_defence.find(
                'tbody').find_all('tr', class_=None)
            basic_defence_data = []
            for i in basic_defence:
                name = i.find('th').get_text()
                basic_defence_data.append([name]+[td.get_text()
                                          for td in i.find_all('td')])
            # print(basic_defence_data)
            def_data = []
            yards_allowed = gen_stats.iloc[5].to_dict()
            # print(yards_allowed)

            def swap_keys(dictionary, first_key, second_key):
                if first_key not in dictionary or second_key not in dictionary:
                    return "One or both of the specified keys are missing from the dictionary."

                temp = dictionary[first_key]
                dictionary[first_key] = dictionary[second_key]
                dictionary[second_key] = temp
                return dictionary
            yards_allowed = swap_keys(yards_allowed, list(yards_allowed.keys())[
                                      1], list(yards_allowed.keys())[-1])
            points = swap_keys(points, list(points.keys())[
                0], list(points.keys())[-1])
            # print(yards_allowed)

            # print(def_data)
            try:
                returns = [x for x in tables if 'returns' in str(x)][0][0]
                # print(returns)
                columns_return = [th.get_text() for th in returns.find(
                    'thead').find_all('tr')[1].find_all('th')][:]
                columns_return = ['player', 'tm', 'kickoff returns', 'kick return yards', 'kick yards per return', 'kickoff td',
                                  'longest kick return', 'punt returns', 'punt yards', 'punt yards per return', 'punt td', 'longest punt return']
                # print(columns_return)
                returns = returns.find('tbody').find_all('tr', class_=None)
                returns_data = []
                for i in returns:
                    name = i.find('th').get_text()
                    returns_data.append([name]+[td.get_text()
                                        for td in i.find_all('td')])
                return_exists = True
            except IndexError:
                return_exists = False

            # print(returns_data)

            kicking = [x for x in tables if 'kicking' in str(x)][0][0]
            # print(kicking)
            columns_kicking = [th.get_text() for th in kicking.find(
                'thead').find_all('tr')[1].find_all('th')][:]
            columns_kicking = ['player', 'tm', 'extra points made', 'extra point attempts', 'field goals made', 'field goals attempted', 'punts',
                               'punt yards', 'yards per punt', 'longest punt']
            # print(columns_kicking)
            kicking = kicking.find('tbody').find_all('tr', class_=None)
            kicking_data = []
            for i in kicking:
                name = i.find('th').get_text()
                kicking_data.append([name]+[td.get_text()
                                    for td in i.find_all('td')])

            # print(kicking_data)
            try:
                passing = [
                    x for x in tables if 'passing_advanced' in str(x)][0][0]
                # print(passing.find('thead').find_all('tr'))
                columns_passing = [th.get_text() for th in passing.find(
                    'thead').find_all('tr')[0].find_all('th')][:]
                columns_passing = ['player', 'tm', 'completions', 'attempts', 'passing yards', 'passing first downs', 'passing first down percent',
                                   'intended air yards', 'intended air yards per attempt', 'completed air yards', 'completed air yards per catch',
                                   'completed air yards per attempt', 'pass yac', 'pass yac per completion', 'dropped passes', 'percentage passes dropped',
                                   'bad throws', 'bad throw percent', 'sacked_pass', 'times blitzed', 'hurried', 'hit', 'pressure',
                                   'pressure percent', 'scrambles', 'yards per scramble']
            except Exception as e:
                print(f'No advanced passing stats\nReason- Error:{e}\n')

            home_snap = [
                x for x in tables if 'home_snap_counts' in str(x)][0][0]
            # print(passing.find('thead').find_all('tr'))
            columns_snap = [th.get_text() for th in home_snap.find(
                'thead').find_all('tr')[1].find_all('th')][:]
            columns_snap = ['player', 'Position', 'O_num snaps', 'O_percentage',
                            'D_num snaps', 'D_percentage', 'ST_num snaps', 'ST_percentage']
            # print(columns_snap)
            home_snap = home_snap.find('tbody').find_all('tr', class_=None)
            home_snap_data = []
            for i in home_snap:
                name = i.find('th').get_text()
                home_snap_data.append([name]+[td.get_text()
                                      for td in i.find_all('td')])

            # print(home_snap_data)

            away_snap = [
                x for x in tables if 'vis_snap_counts' in str(x)][0][0]
            # print(passing.find('thead').find_all('tr'))
            columns_snap = [th.get_text() for th in away_snap.find(
                'thead').find_all('tr')[1].find_all('th')][:]
            columns_snap = ['player', 'Position', 'O_num snaps', 'O_percentage',
                            'D_num snaps', 'D_percentage', 'ST_num snaps', 'ST_percentage']
            # print(columns_snap)
            away_snap = away_snap.find('tbody').find_all('tr', class_=None)
            away_snap_data = []
            for i in away_snap:
                name = i.find('th').get_text()
                away_snap_data.append([name]+[td.get_text()
                                      for td in i.find_all('td')])

            # print(away_snap_data)

            basic_offence = pd.DataFrame(
                basic_offence_data, columns=columns_offence)
            if return_exists is True:
                returns = pd.DataFrame(
                    returns_data, columns=columns_return).drop('tm', axis=1)

            kicking = pd.DataFrame(kicking_data, columns=columns_kicking)

            basic_defence = pd.DataFrame(
                basic_defence_data, columns=columns_defence)

            home_snap = pd.DataFrame(home_snap_data, columns=columns_snap)
            away_snap = pd.DataFrame(away_snap_data, columns=columns_snap)
            snaps = pd.concat([home_snap, away_snap], axis=0)
            # snaps.drop(columns=['tm'],inplace=True)

            # print(basic_offence.dtypes)
            # print(rushing.dtypes)
            # print(rec.dtypes)

            offence = basic_offence
            if return_exists is True:
                offence = offence.merge(returns, on='player', how='left')
            offence = offence.merge(snaps, on='player', how='left')

            offence = offence.replace({'%': ''}, regex=True)
            cols = offence.columns.drop(['player', 'tm', 'Position'])
            offence[cols] = offence[cols].apply(
                pd.to_numeric, errors='coerce')
            offence = offence.fillna(0)

            defence = basic_defence
            if return_exists is True:
                defence = defence.merge(returns, on='player', how='left')

            defence = defence.merge(snaps, on='player', how='left').drop(
                ['O_num snaps', 'O_percentage'], axis=1)
            defence = defence.replace({'%': ''}, regex=True)
            cols_d = defence.columns.drop(['player', 'tm', 'Position'])

            for col in cols_d:
                defence[col] = pd.to_numeric(
                    defence[col].to_numpy(), errors='coerce')
            defence = defence.fillna(0)

            kicking = kicking.replace({'%': ''}, regex=True).fillna(0)
            cols = kicking.columns.drop(['player', 'tm'])
            kicking[cols] = kicking[cols].apply(
                pd.to_numeric, errors='coerce')
            kicking = kicking.fillna(0)

            offence['date'] = game_date
            offence['week'] = date
            offence['home'] = np.where(
                offence['tm'].str.lower() == host.lower(), 1, 0)

            offence['against'] = np.where(
                offence['tm'].str.lower() == summary['winner'][0].lower(), summary['winner'][0].upper(), summary['loser'][0].upper())

            defence['date'] = game_date
            defence['week'] = date
            defence['home'] = np.where(
                defence['tm'].str.lower() == host.lower(), 1, 0)
            defence['yards allowed'] = defence['tm'].map(yards_allowed)
            defence['points allowed'] = defence['tm'].map(points)

            defence['against'] = np.where(
                defence['tm'].str.lower() == summary['winner'][0].lower(), summary['winner'][0].upper(), summary['loser'][0].upper())
            kicking['date'] = game_date
            kicking['week'] = date
            kicking['home'] = np.where(
                kicking['tm'].str.lower() == host.lower(), 1, 0)

            kicking['against'] = np.where(
                kicking['tm'].str.lower() == summary['winner'][0].lower(), summary['winner'][0].upper(), summary['loser'][0].upper())

            host = teams[host]
            for x in [offence, defence, kicking]:
                x['tm'] = x['tm'].map(teams)

                x = swap_values(x, "tm", teams[summary['winner'][0].upper()],
                                teams[summary['loser'][0].upper()], "against")

            offence.to_csv(basic_dir+'//'+str(season)+'//' +
                           f'{summary["winner"][0]}_vs_{summary["loser"][0]}_{season}_offence_week_{date}.csv', index=False)
            defence.to_csv(basic_dir+'//'+str(season)+'//' +
                           f'{summary["winner"][0]}_vs_{summary["loser"][0]}_{season}_defence_week_{date}.csv', index=False)
            kicking.to_csv(basic_dir+'//'+str(season)+'//' +
                           f'{summary["winner"][0]}_vs_{summary["loser"][0]}_{season}_kicking_week_{date}.csv', index=False)
