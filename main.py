import os
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import sqlite3
import requests
import faiss
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

import strava_db

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# -----------------------
# Dataclass to represent a run
# -----------------------
@dataclass
class RunRecord:
    activity_id: int
    start_date: datetime
    distance_km: float
    time_min: float
    avg_heartrate: float
    elevation_m: float
    avg_speed: float


    def summary_text(self) -> str:
        """Create a compact text representation of the run that will be embedded.
        This includes numeric values turned into human-readable text so the embedding model
        picks up on them naturally.
        """
        pace_s_per_km = (self.time_min * 60) / max(self.distance_km, 1e-6)
        return (
        f"Date: {self.start_date.date()}. "
        f"Distance: {self.distance_km:.1f} km. "
        f"Time: {self.time_min:.1f} min. "
        f"Pace: {pace_s_per_km:.1f} s/km. "
        f"Avg HR: {self.avg_heartrate:.0f}. "
        f"Elevation gain: {self.elevation_m:.0f} m."
        )

# -----------------------
# Step 1: Read runs from your local DB
# -----------------------

def read_runs_from_db(limit: int = 100) -> List[RunRecord]:
    """Read the most recent runs from the SQLite DB created by strava_db.py.
    Returns a list of RunRecord sorted newest first.
    """
    conn = sqlite3.connect(strava_db.DB_FILE)
    df = pd.read_sql_query('SELECT * FROM runs ORDER BY Start_Date DESC LIMIT ?', conn, params=(limit,))
    conn.close()

    runs: List[RunRecord] = []
    for _, r in df.iterrows():
        runs.append(RunRecord(
        activity_id=int(r['Activity_ID']),
        start_date=pd.to_datetime(r['Start_Date']),
        distance_km=float(r['Distance_km']),
        time_min=float(r['Time_min']),
        avg_heartrate=float(r['Avg_Heartrate']),
        elevation_m=float(r['Elevation_m']),
        avg_speed=float(r['Avg_Speed']) if 'Avg_Speed' in r else 0.0
        ))
    return runs

# -----------------------
# Step 2: Embeddings (OpenAI)
# -----------------------

def make_text_embedding(text: str, model: str = 'text-embedding-3-small') -> np.ndarray:
    """Send text to OpenAI embeddings endpoint and return numpy vector.
    Keep it simple with requests; the official SDK works too.
    """
    assert OPENAI_API_KEY, "OPENAI_API_KEY not set in .env"
    url = 'https://api.openai.com/v1/embeddings'
    headers = { 'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json' }
    payload = { 'input': text, 'model': model }
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return np.array(r.json()['data'][0]['embedding'], dtype=np.float32)

# -----------------------
# Step 3: Vector store using FAISS
# -----------------------
class SimpleVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metas: List[Dict[str, Any]]):
        # normalize and add
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        vectors = vectors / norms
        self.index.add(vectors.astype(np.float32))
        self.metadatas.extend(metas)

    def search(self, q: np.ndarray, k: int = 5):
        q = q / np.linalg.norm(q)
        D, I = self.index.search(q.reshape(1, -1).astype(np.float32), k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0: continue
            results.append((self.metadatas[idx], float(score)))
        return results
    
# -----------------------
# Helper: build store from runs
# -----------------------

def build_store_from_runs(runs: List[RunRecord]) -> SimpleVectorStore:
    texts = [r.summary_text() for r in runs]
    embeddings = np.stack([make_text_embedding(t) for t in texts], axis=0)
    store = SimpleVectorStore(dim=embeddings.shape[1])
    metas = [
        {
            'activity_id': r.activity_id,
            'date': r.start_date.isoformat(),
            'distance_km': r.distance_km,
            'time_min': r.time_min,
            'avg_heartrate': r.avg_heartrate,
            'elevation_m': r.elevation_m
        }
        for r in runs
    ]
    store.add(embeddings, metas)
    return store

# -----------------------
# LLM chat helper
# -----------------------

def llm_query(system_prompt: str, user_prompt: str, model: str = 'gpt-4o-mini') -> str:
    assert OPENAI_API_KEY
    url = 'https://api.openai.com/v1/chat/completions'
    headers = { 'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json' }
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        'max_tokens': 400,
        'temperature': 0.7
    }
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']

# -----------------------
# Streamlit UI
# -----------------------

def streamlit_app():
    st.set_page_config(page_title='RunSage — Simple LLM running coach')
    st.title('RunSage — Simple LLM running coach')


    # Read runs from DB
    runs = read_runs_from_db(200)
    if not runs:
        st.warning('No runs found in DB. Run strava_db.py to populate the database first.')
        return


    # Build DataFrame to show in dashboard
    df = pd.DataFrame([
        {
            'date': r.start_date,
            'distance_km': r.distance_km,
            'time_min': r.time_min,
            'avg_hr': r.avg_heartrate,
            'elevation_m': r.elevation_m
        }
    for r in runs
    ])

    tabs = st.tabs(['Dashboard', 'Chat', 'Semantic Search'])

    with tabs[0]:
        st.header('Dashboard')
        st.dataframe(df[['date','distance_km','time_min','avg_hr']])
        df['pace_s_per_km'] = (df['time_min'] * 60) / df['distance_km']
        fig, ax = plt.subplots()
        ax.plot(pd.to_datetime(df['date']), df['pace_s_per_km'], marker='o')
        ax.set_ylabel('s per km')
        ax.set_title('Pace over time')
        st.pyplot(fig)


    with tabs[1]:
        st.header('Chat with RunSage')
        user_q = st.text_input('Ask about your training', value='Summarise my last 2 weeks of training')
        if st.button('Ask'):
            # Provide recent runs as context to the LLM
            recent = runs[:10]
            context = ''.join([r.summary_text() for r in recent])
            system_prompt = 'You are an experienced running coach. Give concise, practical feedback.'
            user_prompt = f"Here are recent runs: {context}. Question: {user_q}"
            with st.spinner('Thinking...'):
                ans = llm_query(system_prompt, user_prompt)
                st.write(ans)

    with tabs[2]:
        st.header('Semantic search')
        query = st.text_input('Find runs similar to', value='easy long run with low heart rate')
        if st.button('Search'):
            store = build_store_from_runs(runs)
            q_emb = make_text_embedding(query)
            results = store.search(q_emb, k=5)
            for meta, score in results:
                st.write(meta)
                st.write(f'score {score:.3f}')

if __name__ == '__main__':
    streamlit_app()