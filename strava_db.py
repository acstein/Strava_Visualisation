import requests
import urllib3
import pandas as pd
import sqlite3
from pandas import json_normalize
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv() 

CLIENT_ID = os.getenv("client_id")
CLIENT_SECRET = os.getenv("client_secret")
REFRESH_TOKEN = os.getenv("refresh_token")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===============================
# STRAVA AUTHENTICATION
# ===============================

def get_access_token():
    auth_url = 'https://www.strava.com/oauth/token'
    payload = {
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
    'refresh_token': REFRESH_TOKEN,
    'grant_type': 'refresh_token',
    'f': 'json'
    }
    res = requests.post(auth_url, data=payload, verify=False)
    return res.json()['access_token']

# ===============================
# FETCH STRAVA ACTIVITIES
# ===============================

def fetch_strava_activities(header, after: int | None = None) -> pd.DataFrame:
    """
    Fetch Strava runs, optionally only after a given timestamp.
    Returns a pandas DataFrame with useful columns.
    """
    activities_url = 'https://www.strava.com/api/v3/athlete/activities'
    all_data = []
    page = 1

    while True: # fetch activities using HTTP requests library
        params = {'per_page': 200, 'page': page}
        if after:
            params['after'] = after # only get newer activities if after is specified

        response = requests.get(activities_url, headers=header, params=params).json()
        if not response:
            break
        all_data.extend(response)
        page += 1

    # Filter for runs recorded on Garmin only
    run_data = [a for a in all_data if a['type'] == 'Run' and a['has_heartrate']] 
    if not run_data:
        return pd.DataFrame()

    df = json_normalize(run_data) # turn from json format into a dataframe
    clean_df = pd.DataFrame()
    clean_df['Distance_km'] = df['distance'] / 1000
    clean_df['Time_min'] = df['elapsed_time'] / 60 
    clean_df['Avg_Heartrate'] = df['average_heartrate'].fillna(0)
    clean_df['Max_Heartrate'] = df['max_heartrate']
    clean_df['Elevation_m'] = df['total_elevation_gain'].fillna(0)
    clean_df['Avg_Pace'] = pd.to_timedelta(16.666666667 * 60 / df['average_speed'], unit="s") # convert to pace
    clean_df["Avg_Pace_Readable"] = clean_df["Avg_Pace"].apply(lambda td: str(td).split(".")[0].replace("0 days ", ""))
    clean_df['Max_Pace'] = pd.to_timedelta(16.666666667 * 60 / df['max_speed'], unit="s")
    clean_df["Max_Pace_Readable"] = clean_df["Max_Pace"].apply(lambda td: str(td).split(".")[0].replace("0 days ", ""))
    clean_df['Start_Date'] = pd.to_datetime(df['start_date']).dt.strftime("%d-%m-%Y %H:%M")
    clean_df['Activity_ID'] = df['id']

    return clean_df

# ===============================
# DATABASE FUNCTIONS
# ===============================

DB_FILE = "strava_runs.db"

def init_db():
    """
    Connect to the database and create a runs table if not already present
    """
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''
    CREATE TABLE IF NOT EXISTS runs (
    Activity_ID INTEGER KEY,
    Start_Date TEXT,
    Distance_km REAL,
    Time_min REAL,
    Avg_Heartrate REAL,
    Max_Heartrate REAL,
    Elevation_m REAL,
    Avg_Speed REAL,
    Max_Speed REAL,
    Avg_Pace TEXT,
    Avg_Pace_Readable TEXT,
    Max_Pace TEXT,
    Max_Pace_Readable TEXT
    )
    ''')
    conn.commit()
    conn.close()

def get_latest_activity_date() -> datetime | None:
    """
    Get the date of the latest run in the database
    """
    conn = sqlite3.connect(DB_FILE)
    result = conn.execute("SELECT MAX(Start_Date) FROM runs").fetchone()[0]
    conn.close()
    return datetime.fromisoformat(result) if result else None

def save_to_db(df: pd.DataFrame):
    """
    Save any new runs found to the table in the database
    """
    if df.empty:
        print("No new runs to add.")
        return
    conn = sqlite3.connect(DB_FILE)
    df.to_sql("runs", conn, if_exists="append", index=False)
    conn.close()
    print(f"Added {len(df)} new runs to the database.")

# ===============================
# MAIN FLOW (if run standalone)
# ===============================

def main():
    print("Connecting to Strava...")
    token = get_access_token()
    header = {'Authorization': f'Bearer {token}'}

    init_db()

    last_date = get_latest_activity_date()
    after_ts = int(last_date.timestamp()) if last_date else None

    print(f"Fetching runs {'after ' + str(last_date) if last_date else '(all time)'}...")
    df_new = fetch_strava_activities(header, after=after_ts)

    save_to_db(df_new)

    print("Done!")

if __name__ == "__main__":
    main()

# Expose a clear API when imported and reduce namespace clutter
__all__ = [
'get_access_token', 'fetch_strava_activities', 'init_db',
'get_latest_activity_date', 'save_to_db', 'DB_FILE'
]



