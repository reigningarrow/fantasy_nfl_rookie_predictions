# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 19:09:24 2024

@author: sambi
"""

import requests

'https://api.fantasy.nfl.com/v2/batchservices?services=[{"leaderboard":"siteId=1%26count=2%26offset=4"},{"gameStats":""}]'


i = requests.get(
    'https://stage.api.fantasy.nfl.com/v3/players/stats?season=2022&week=4&page[size]=200')

x = i.json()
