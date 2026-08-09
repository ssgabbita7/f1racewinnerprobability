# F1 Winner Predictability Model

Ask a natural-language question about a live F1 race situation and receive a data-driven win-probability estimate, backed by historical race data.

**Example:** *"It just rained at Silverstone, Hamilton is in the lead, what is the probability of his win?"*

---

## Architecture

```
┌─────────────────┐   natural language   ┌──────────────────┐   features+text   ┌──────────────────┐
│  React UI       │ ──────────────────► │  Node.js/Express │ ────────────────► │  FastAPI ML svc  │
│  frontend/      │ ◄────────────────── │  backend/        │ ◄──────────────── │  ml-service/     │
│  :5173          │   probability +      │  :3000           │   kNN + model     │  :8000           │
└─────────────────┘   supporting cases   └──────────────────┘                   └──────────────────┘
                                                │                                        │
                                          Claude API                             FAISS + sklearn
                                          (NLP parsing)                         sentence-transformers
                                                                Ergast-schema historical data (results,
                                                                lap times, standings, incidents)
```

| Service | Technology | Responsibility |
|---|---|---|
| Frontend | React + Vite + Tailwind CSS | F1-themed UI, probability display |
| Backend | Node.js + Express | NLP parsing via Claude API, orchestration |
| ML service | FastAPI (Python) | Embeddings, FAISS retrieval, probability model |

---

## Prerequisites

- Python 3.11+
- Node.js 20+
- An [Anthropic API key](https://console.anthropic.com/)

---

## Setup

### 1. Clone and configure environment

```bash
git clone <repo>
cd f1-predictor
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY
```

Key variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Used for NLP entity extraction |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model for NLP |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers embedding model name |
| `ML_SERVICE_URL` | `http://localhost:8000` | Python service URL seen by Node |
| `FAISS_K` | `10` | Number of nearest neighbours to retrieve |
| `PROBABILITY_BLEND_WEIGHT` | `0.5` | 0 = kNN only, 1 = sklearn model only |

---

### 2. Python ML service

```bash
cd ml-service
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

#### 2a. Provide the Ergast dataset

The training pipeline reads from the classic Ergast-schema "Formula 1 World
Championship" tables (self-consistent on `raceId`/`driverId`/`constructorId`) —
`circuits, races, drivers, constructors, results, status, driver_standings,
constructor_standings, pit_stops, lap_times, safety_cars, virtual_safety_cars,
red_flags` (CSV) plus `virtual_safety_car_estimates.json`. This isn't fetched
by a script — copy those 14 files into `ml-service/data/raw/ergast/`.

> **Column name mismatch?** If ingest.py prints a warning about missing columns, open
> `data_pipeline/ingest.py` and update the `*_COLS` dicts at the top to match the
> actual column names in your dataset export (run `head -1 data/raw/ergast/<file>.csv` to inspect).

> **Optional: F1DB data.** `data_pipeline/download_data.py` still fetches the
> F1DB release from GitHub into `data/raw/` — it's no longer used by the core
> training pipeline, but `data_pipeline/fastf1_ingest.py` (see below) still
> matches pending races against it.

#### 2b. Build scenario documents

```bash
python -m data_pipeline.build_scenarios
```

Writes `data/processed/scenarios.jsonl`: one `pre_race` scenario per driver
per race (~27 k for 1950–2026), plus several `lap_snapshot` scenarios per
driver-race sampled from real running position in `lap_times.csv` at 25/50/75/90%
race distance (~55 k more, for 1982+ races with lap-time coverage) — roughly
**80 k+ scenarios** total.

#### 2c. Build FAISS index and train model

```bash
python -m data_pipeline.build_index
```

This embeds all scenario texts (a few minutes on CPU with the default
`all-MiniLM-L6-v2` model — scales with corpus size), builds the FAISS index, trains the gradient-boosting
classifier, and writes to `data/artifacts/`.

To **refresh** after new race data arrives: re-run steps 2b and 2c (or call
`POST http://localhost:8000/rebuild` while the service is running).

#### 2d. Start the ML service

```bash
uvicorn main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/health`

---

### 3. Node.js backend

```bash
cd backend
npm install
npm run dev       # or: npm start
```

Verify: `curl http://localhost:3000/health`

---

### 4. React frontend

```bash
cd frontend
npm install
cp .env.example .env    # defaults to VITE_BACKEND_URL=http://localhost:3000
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## How it works

1. **NLP parsing** — Your free-text query is sent to Claude with tool-use enabled. Claude extracts structured entities: driver, circuit, weather, position, lap, safety car / VSC, pit stops, etc.
2. **Scenario text** — The extracted entities are assembled into a sentence matching the style of the historical training texts (e.g. *"silverstone, hamilton, currently P1, wet conditions, lap 40 of 52, safety car deployed"*).
3. **FAISS retrieval** — The scenario text is embedded with `sentence-transformers` and the top-k most similar historical race scenarios are retrieved from the FAISS index — including real mid-race `lap_snapshot` scenarios, not just pre-race setups.
4. **kNN win rate** — The fraction of retrieved neighbours where the driver won gives a baseline empirical win rate.
5. **sklearn model** — A gradient-boosting classifier trained on structured features (grid/current position, weather, driver career win rates, championship-standing momentum, pit-stop progress, caution-period context) gives a calibrated statistical probability.
6. **Blending** — The two signals are blended (configurable weight) into a final probability.
7. **Response** — Probability, confidence label, and 2–3 supporting historical examples (now including the lap/race-progress they were snapshotted at) are returned to the UI.

### Probability model features

| Feature | Description |
|---|---|
| `grid_position_norm` | `(20 − grid) / 19` — 1.0 = pole, 0.0 = last |
| `weather_wet` | 1 if wet or mixed, 0 if dry |
| `driver_circuit_win_rate` | Career wins at this circuit / career starts |
| `driver_overall_win_rate` | Career wins / career starts |
| `current_position_norm` | Same normalisation as grid; defaults to grid if not supplied |
| `race_progress_pct` | `lap / total_laps`; 0.0 if not specified |
| `driver_standing_position_norm` | Rank-normalized championship position entering the race (previous round in that season); 0.5 default |
| `constructor_standing_position_norm` | Same, for the constructor |
| `pit_stops_completed_norm` | Stops completed so far / 4, capped at 1.0 |
| `incident_active` | 1 if a safety car / VSC / red flag has occurred by this point in the race |

---

## Project structure

```
f1-predictor/
├── .env.example
├── .gitignore
├── README.md
│
├── backend/                        Node.js + Express
│   ├── package.json
│   └── src/
│       ├── index.js
│       ├── routes/predict.js       POST /predict
│       └── services/
│           ├── nlpParser.js        Claude entity extraction
│           └── mlService.js        HTTP client → Python service
│
├── ml-service/                     Python FastAPI
│   ├── requirements.txt
│   ├── main.py                     FastAPI app
│   ├── embedder.py                 Swappable embedding provider
│   ├── faiss_index.py              FAISS build / load / search
│   ├── probability_model.py        sklearn gradient-boosting model (10 features)
│   └── data_pipeline/
│       ├── ergast_helpers.py       Status/DNF, total-laps, pit-stops, SC/VSC/red-flag windows,
│       │                           leakage-safe standings-entering-race joins
│       ├── ingest.py               Ergast CSVs → clean DataFrame
│       ├── build_scenarios.py      DataFrame → scenarios.jsonl (pre_race + lap_snapshot)
│       ├── build_index.py          Embed → FAISS + train model
│       ├── download_data.py        Download F1DB from GitHub (legacy; used only by fastf1_ingest.py)
│       └── fastf1_ingest.py        FastF1 session CSVs → pending_scenarios.jsonl
│   └── data/
│       ├── raw/ergast/             Ergast dataset CSVs + JSON (gitignored, manually provided)
│       ├── raw/f1db-*.csv          Legacy F1DB data (gitignored; fastf1_ingest.py only)
│       ├── processed/              scenarios.jsonl
│       ├── artifacts/              faiss.index, model.pkl, scenarios_enriched.jsonl
│       └── weather_lookup.csv      Known wet/mixed races
│
└── frontend/                       React + Vite + Tailwind CSS
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── index.css                CSS theme (F1 carbon-fiber aesthetic)
        ├── lib/api.js                POST /predict client
        └── components/
            ├── Header.jsx            Hero title + start-lights animation
            ├── StartLights.jsx
            ├── ExampleQueries.jsx
            ├── QueryForm.jsx
            ├── ResultsPanel.jsx
            ├── Gauge.jsx              Win-probability meter
            ├── ConfidenceBadge.jsx
            ├── ContextGrid.jsx        Parsed race context cards
            ├── CaseCard.jsx           Supporting historical case card
            └── ErrorPanel.jsx
```

---

## Swapping the embedding model

Set in `.env`:

```bash
EMBEDDING_MODEL=all-mpnet-base-v2
```

After changing the model, **rebuild the index** (step 2b-c) since existing
embeddings are incompatible with a different model.

---

## Ingesting FastF1 lap-by-lap sessions

Besides F1DB's aggregated per-race CSVs, you can ingest raw lap-by-lap session
exports from the [`fastf1`](https://github.com/theOehrly/Fast-F1) Python
package (one row per lap: sector times, tyre life, speed traps, etc.) — useful
for qualifying/practice sessions from races that haven't finished yet, so
F1DB doesn't have a result for them.

Drop the CSV in `ml-service/data/raw/fastf1/`, keeping FastF1's own filename
convention (`<year>-<Event Name>-<Session>.csv`, e.g.
`2026-Belgian Grand Prix-Qualifying.csv`), then run:

```bash
cd ml-service
python -m data_pipeline.fastf1_ingest "data/raw/fastf1/2026-Belgian Grand Prix-Qualifying.csv"
```

This derives each driver's qualifying position from their best valid lap,
matches the event/drivers/teams against F1DB, and writes the result to
`data/processed/pending_scenarios.jsonl` — **not** `scenarios.jsonl`. Sessions
without a finished F1DB race result have no known `won`/`finished_position`/
`dnf`, so they're kept out of `build_index.py`'s FAISS/model training rather
than fabricating an outcome. The script also prints a ready-to-POST
`{query_text, features}` payload per driver for the ML service's `/predict`
endpoint. Once F1DB publishes the real result for that race (rerun
`download_data.py`), re-run the normal `ingest.py` → `build_scenarios.py` →
`build_index.py` pipeline instead — the script will note when this is already
the case.

---

## Known limitations / roadmap

- **Lap-by-lap coverage**: `lap_times.csv` covers 1982–2026 with partial early-era gaps; pre-1982 races and any driver-races missing lap-time coverage only get a `pre_race` scenario, not `lap_snapshot` scenarios. Snapshots are also capped at 90% race distance by design (a snapshot at the finish line makes position near-tautologically predictive of the outcome).
- **Championship-standings momentum has no cross-season carryover**: it defaults to a neutral rank (0.5-normalized) for round-1-of-season snapshots and any live query that doesn't supply it, rather than carrying over the previous season's standing — avoids conflating different car generations across a winter break, at the cost of a slightly weaker signal early in each season.
- **Incident features are per-lap-snapshot only, never neighbour-averaged**: the pre-race scenario always encodes "no incident yet" (correct — these events are unknown before lights-out), and at inference time `safety_car_active`/`vsc_active`/`red_flag_occurred` default to "no incident" unless the user's query explicitly states one — averaging race-specific noise across "similar" historical races would inject spurious bias into unrelated predictions.
- **Weather coverage**: `weather_lookup.csv` covers ~30 known wet/mixed races. All others default to "dry". Contributions welcome.
- **Data leakage in driver/circuit win rates**: these are computed over the full dataset rather than only past races (unlike the newer championship-standings feature, which specifically uses a leakage-safe previous-round computation). A temporal split would improve calibration.
- **Single-session**: no auth, no persistent history, local dev only.
