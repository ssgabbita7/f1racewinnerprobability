# Make

Quick command reference for setup, running, and rebuilding the F1 win-probability app. There's no actual Makefile (mixed Python/Node repo) — this is the prose equivalent. See `README.md` for the full explanation of each step.

## Setup (one-time)

```bash
cp .env.example .env
# edit .env — set ANTHROPIC_API_KEY at minimum
```

```bash
cd ml-service
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```
Copy the 14 Ergast dataset files into `ml-service/data/raw/ergast/` (see README §2a — not fetched by a script).

```bash
cd backend && npm install
cd ../frontend && npm install
cp .env.example .env            # frontend/.env, defaults to VITE_BACKEND_URL=http://localhost:3000
```

## Build the ML pipeline

Run from `ml-service/` with the venv active:

```bash
python -m data_pipeline.ingest            # sanity check: row counts, DNF/win rate
python -m data_pipeline.build_scenarios   # -> data/processed/scenarios.jsonl
python -m data_pipeline.build_index       # -> data/artifacts/{faiss.index,model.pkl,scenarios_enriched.jsonl}
```

Re-run all three after adding new raw data. `build_index` prints the 5-fold ROC-AUC.

## Run (dev)

Three terminals, in order:

```bash
# 1. ML service — :8000
cd ml-service
.venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

```bash
# 2. Backend — :3000
cd backend
npm run dev
```

```bash
# 3. Frontend — :5173
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Verify

```bash
curl http://localhost:8000/health     # {"status":"ready","index_size":...,"scenarios":...}
curl http://localhost:3000/health     # {"status":"ok",...}
```

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"query":"Verstappen starts on pole at Monza this weekend"}'
```

## Hot-reload the model without restarting

After rebuilding artifacts (`build_index`) while the ML service is already running:

```bash
curl -X POST http://localhost:8000/rebuild
```

## Ingest a pending (outcome-unknown) FastF1 session

```bash
cd ml-service
python -m data_pipeline.fastf1_ingest "data/raw/fastf1/2026-Belgian Grand Prix-Qualifying.csv"
```
Writes to `data/processed/pending_scenarios.jsonl` — kept out of training until F1DB has a real result.
