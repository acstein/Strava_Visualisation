import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import pandas as pd
import requests
import urllib3
import seaborn as sns
from pandas import json_normalize

import login as login

"""
API Set-up and Handling
"""
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

auth_url = 'https://www.strava.com/oauth/token'
activities_url = 'https://www.strava.com/api/v3/athlete/activities'

payload = {
    'client_id': f'{login.client_id}',
    'client_secret': f'{login.client_secret}',
    'refresh_token': f'{login.refresh_token}',
    'grant_type': 'refresh_token',
    'f': 'json'
}

res = requests.post(auth_url, data=payload, verify=False)
access_token = res.json()['access_token']

header = {'Authorization': 'Bearer ' + access_token}

"""
Data fetching and initial cleaning
"""
def get_data():
    page = 1
    still_more = True
    data = []
    while still_more:
        get_strava = requests.get(activities_url, headers=header, params={'per_page': 200, 'page': f'{page}'}).json()
        still_more = get_strava
        data.extend(get_strava)
        page += 1

    return data

data = get_data()

run_data = []
for activity in data:
    if activity['type'] == 'Run':
        run_data.append(activity)

run_data = json_normalize(run_data)

run_data['Distance / km'] = run_data['distance'].apply(lambda x: x/1000)
run_data['Time / minutes'] = run_data['elapsed_time'].apply(lambda x: x/60)

plt.scatter(run_data['Distance / km'], run_data['Time / minutes'])
plt.show()
