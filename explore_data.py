# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:29:41 2024

@author: sambi
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('games//seasons//2025_offence.csv')

# Count occurrences of each row
row_counts = df['player'].value_counts()

# Filter rows that occur less than 6 times
rows_to_keep = row_counts[row_counts >= 6].index

# Create a new DataFrame with only the rows that have values in 'A' occurring 6 or more times
df = df[df['player'].isin(rows_to_keep)]

df = df[df['o_percentage'] > 60]

for pos in df['position'].unique():
    plt.figure()
    plt.hist(df['score'], bins=20, alpha=0.5, label='all players')
    df2 = df[df['position'] == pos]
    plt.hist(df2['score'], bins=20, alpha=0.5, label=pos)
    plt.xlabel('Score')
    plt.title(
        f'Histogram of {pos} scores across the 2023 season\n for players with greater than 5 games played\n and more than 60% of snaps in a given game. n={len(df2)}')
    plt.legend()

for pos in df['position'].unique():
    plt.figure()
    # plt.hist(df['score'], bins=20, alpha=0.5,label='all players')
    df2 = df[df['position'] == pos]
    plt.hist(df2['score'], bins=20, alpha=0.5, label=pos)
    plt.xlabel('Score')
    plt.title(
        f'Histogram of {pos} scores across the 2023 season\n for players with greater than 5 games played\n and more than 60% of snaps in a given game. n={len(df2)}')
    plt.legend()
