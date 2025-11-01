# Strava_Visualisation

# RunBot — AI Running Coach

RunBot is a Python-based running analytics and coaching tool that combines your Strava running data with OpenAI’s language and embedding models. It allows you to:

* Explore your runs via an interactive **dashboard**
* Ask an AI “coach” for insights in natural language (**Chat**)
* Perform **semantic search** to find runs similar to a query

The project focuses on **application** rather than training: embeddings are generated via OpenAI API, and uses FAISS for efficient semantic search. No neural network training is required.

---

## Features

1. **Strava integration**: Fetches runs from Strava and store them in a local SQLite database.
2. **Embeddings & semantic search**: Runs are converted to text summaries, embedded using OpenAI embeddings, and stored in FAISS.
3. **LLM coaching**: Ask natural language questions about training. RunBot provides coaching insights based on your recent runs.
4. **Streamlit UI**: Dashboard with tables, pace-over-time plots, chat interface, and semantic search.

---

## Getting Started

### 1. Clone the repo

```bash
git clone <your-repo-url>
```

### 2. Install dependencies

---

### 3. Set up `.env`

Create a `.env` file in the project root:

```bash
# OpenAI
OPENAI_API_KEY=sk-your-openai-key-here

# Strava
STRAVA_CLIENT_ID=12345
STRAVA_CLIENT_SECRET=your-strava-secret
STRAVA_REFRESH_TOKEN=your-refresh-token

---

### 4. Populate your Strava database

Run the Strava ingestion script:

```bash
python strava_db.py
```

This fetches all your runs and saves them into `strava_runs.db`.

---

### 5. Run the app

```bash
streamlit run runsage.py
```

* **Dashboard**: view recent runs and pace charts
* **Chat**: ask AI coach about recent training
* **Semantic Search**: find runs similar to your natural-language query

---

## Project Structure

```
project/
├─ strava_db.py        # Fetches Strava data, writes to local SQLite DB
├─ runbot.py          # Streamlit app: dashboard, chat, semantic search
├─ .env                # Environment variables for OpenAI & Strava
├─ requirements.txt    # Python dependencies
├─ strava_runs.db      # SQLite database of your runs (auto-created)
├─ README.md
```

---

## How It Works (High Level)

1. `strava_db.py` fetches your Strava runs and saves them to a local SQLite database.
2. `runsage.py` reads the runs and converts each to a human-readable summary.
3. OpenAI embeddings API generates vectors for each run summary.
4. FAISS stores embeddings for semantic similarity search.
5. Chat interface sends recent runs and user queries to an LLM (GPT) to generate coaching advice.
6. Dashboard plots pace over time and shows key run metrics.

---

## Notes

* This project is **for personal analysis and portfolio demonstration**; it is **not production-grade** for large-scale usage.
* All secrets (OpenAI key, Strava credentials) are stored locally via `.env`. Never push to public repos.

---

