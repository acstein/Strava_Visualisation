# Strava_Visualisation

# RunSage — AI Running Coach

RunSage is a Python-based running analytics and coaching tool that combines your Strava running data with OpenAI’s language and embedding models. It allows you to:

* Explore your runs via an interactive **dashboard**
* Ask an AI “coach” for insights in natural language (**Chat**)
* Perform **semantic search** to find runs similar to a query

The project focuses on **application** rather than training: embeddings are generated via OpenAI API, and we use FAISS for efficient semantic search. No neural network training is required.

---

## Features

1. **Strava integration**: Fetches your runs from Strava and stores them in a local SQLite database.
2. **Embeddings & semantic search**: Runs are converted to text summaries, embedded using OpenAI embeddings, and stored in FAISS.
3. **LLM coaching**: Ask natural language questions about your training. RunSage provides coaching insights based on your recent runs.
4. **Streamlit UI**: Dashboard with tables, pace-over-time plots, chat interface, and semantic search.

---

## Getting Started

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd RunSage
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Typical `requirements.txt` includes:

```
pandas
requests
sqlite3
python-dotenv
streamlit
faiss-cpu
matplotlib
```

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
```

> ⚠️ Never commit `.env` to GitHub. Add `.env` to `.gitignore`.

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
* **Chat**: ask your AI coach about recent training
* **Semantic Search**: find runs similar to your natural-language query

---

## Project Structure

```
project/
├─ strava_db.py        # Fetches Strava data, writes to local SQLite DB
├─ runsage.py          # Streamlit app: dashboard, chat, semantic search
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

## TODO / Next Improvements

1. **Embedding caching**: Save embeddings to disk so they’re not re-requested every app run.
2. **Persist FAISS index**: Save and reload vector store to reduce startup time.
3. **Add weather context**: Fetch historical weather for each run and include in summary text.
4. **Generate weekly reports**: PDF or HTML summaries with charts + LLM feedback.
5. **Streamlit UX improvements**: Add search filters, date range selectors, and plot options.
6. **Error handling & rate-limit protection**: Retry on 429 errors from OpenAI API, and add caching to reduce API calls.
7. **Authentication / security**: Optional password protection for web app.
8. **Support other activity types**: Include cycling, swimming, etc., with embeddings and summaries.

---

## Notes

* This project is **for personal analysis and portfolio demonstration**; it is **not production-grade** for large-scale usage.
* All secrets (OpenAI key, Strava credentials) are stored locally via `.env`. Never push to public repos.

---

