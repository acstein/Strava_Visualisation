import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import pandas as pd
import requests
import urllib3
import seaborn as sns
from pandas import json_normalize

import login as login

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

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
new_df = pd.DataFrame()

new_df['Distance / km'] = run_data['distance'].apply(lambda x: x/1000)
new_df['Time / minutes'] = run_data['elapsed_time'].apply(lambda x: x/60)
new_df['Avg Heartrate'] = run_data['average_heartrate'].apply(lambda x: x if not pd.isna(x) else 10)
new_df['Elevation / m'] = run_data['total_elevation_gain'].apply(lambda x: x if not pd.isna(x) else 10)
new_df['Average Speed / ms-1'] = run_data['average_speed']
new_df['Max Speed / ms-1'] = run_data['max_speed']
new_df['Start Date'] = run_data['start_date']

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Strava Running App'),

    html.P("Select Metric:"),
    dcc.Dropdown(
        id="dropdown",
        options=[
            {'label': 'Elevation / m', 'value': 'Elevation / m'},
            {'label': 'Average Speed / ms-1', 'value': 'Average Speed / ms-1'},
            {'label': 'Max Speed / ms-1', 'value': 'Max Speed / ms-1'},
        ],
        value='Elevation / m',
        clearable=False,
    ),

    html.P("Filter by Start Date:"),
    dcc.DatePickerRange(
        id='date-picker',
        start_date=new_df['Start Date'].min(),
        end_date=new_df['Start Date'].max(),
        display_format='YYYY-MM-DD'
    ),

    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"),
    Input("dropdown", "value"),
    Input("date-picker", "start_date"),
    Input("date-picker", "end_date")
)
def update_graph(selected_value, start_date, end_date):
    # Filter by date range
    mask = (new_df['Start Date'] >= start_date) & (new_df['Start Date'] <= end_date)
    filtered_df = new_df.loc[mask]

    # Create figure
    fig = px.scatter(
        filtered_df,
        x='Distance / km',
        y='Time / minutes',
        size='Avg Heartrate',
        color=selected_value
    )
    return fig


if __name__ == '__main__':
    app.run(debug=True)

# ToDo: unit tests and type stuff
