"""
FastAPI microservice — owns FAISS retrieval, embeddings, and the probability model.

Endpoints:
  GET  /health
  POST /predict   {query_text, features} → {knn_win_rate, model_probability, blended_probability, confidence, supporting_cases}
  POST /rebuild   Rebuild index from current data/processed/scenarios_enriched.jsonl (admin)

Run:
  uvicorn main:app --reload --port 8000
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent / ".env")

import embedder as emb
import faiss_index as fi
import probability_model as pm

ARTIFACTS_DIR = Path(__file__).parent / "data" / "artifacts"
INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
ENRICHED_PATH = ARTIFACTS_DIR / "scenarios_enriched.jsonl"

FAISS_K = int(os.getenv("FAISS_K", "10"))
BLEND_WEIGHT = float(os.getenv("PROBABILITY_BLEND_WEIGHT", "0.5"))


# ── App state ─────────────────────────────────────────────────────────────────

class AppState:
    index = None
    model = None
    scenarios: list[dict] = []
    ready = False


state = AppState()


def _load_artifacts():
    if not INDEX_PATH.exists() or not MODEL_PATH.exists() or not ENRICHED_PATH.exists():
        print(
            "WARNING: Artifacts not found. Run the data pipeline first:\n"
            "  cd ml-service\n"
            "  python -m data_pipeline.download_data\n"
            "  python -m data_pipeline.build_scenarios\n"
            "  python -m data_pipeline.build_index"
        )
        state.ready = False
        return

    print("Loading FAISS index...")
    state.index = fi.load_index(INDEX_PATH)

    print("Loading probability model...")
    state.model = pm.load_model(MODEL_PATH)

    print("Loading scenario metadata...")
    with open(ENRICHED_PATH, encoding="utf-8") as f:
        state.scenarios = [json.loads(line) for line in f if line.strip()]

    # Warm up embedder
    print("Warming up embedder...")
    emb.embed("warmup")

    state.ready = True
    print(f"Ready. {state.index.ntotal} vectors in index, {len(state.scenarios)} scenarios.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(title="F1 Predictor ML Service", lifespan=lifespan)


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    query_text: str
    features: dict[str, Any] = {}


class SupportingCase(BaseModel):
    year: int | None
    race_name: str
    circuit_name: str
    driver_name: str
    grid_position: int | None
    weather: str
    finished_position: int | None
    won: bool
    scenario_text: str
    lap: int | None = None
    race_progress_pct: float | None = None


class PredictResponse(BaseModel):
    knn_win_rate: float
    model_probability: float
    blended_probability: float
    confidence: str
    supporting_cases: list[SupportingCase]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence(k_found: int, agreement: float, has_features: bool) -> str:
    """Heuristic confidence label."""
    if k_found < 3 or agreement < 0.4:
        return "low"
    if k_found >= 7 and agreement >= 0.6 and has_features:
        return "high"
    return "medium"


def _scenario_to_case(s: dict) -> SupportingCase:
    f = s.get("features", {})
    o = s.get("outcome", {})
    return SupportingCase(
        year=f.get("year"),
        race_name=f.get("race_name", ""),
        circuit_name=f.get("circuit_name", ""),
        driver_name=f.get("driver_name", ""),
        grid_position=f.get("grid_position"),
        weather=f.get("weather", "unknown"),
        finished_position=o.get("finished_position"),
        won=o.get("won", False),
        scenario_text=s.get("text", ""),
        lap=f.get("lap"),
        race_progress_pct=f.get("race_progress_pct"),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ready" if state.ready else "not_ready",
        "index_size": state.index.ntotal if state.index else 0,
        "scenarios": len(state.scenarios),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "ML service is not ready. Run the data pipeline first:\n"
                "  python -m data_pipeline.download_data\n"
                "  python -m data_pipeline.build_scenarios\n"
                "  python -m data_pipeline.build_index"
            ),
        )

    # 1. Embed the query text
    query_emb = emb.embed(req.query_text)  # shape (1, D)
    query_vec = query_emb[0] if query_emb.ndim == 2 else query_emb

    # 2. Search FAISS
    distances, indices = fi.search(state.index, query_vec, k=FAISS_K)

    # Filter out invalid indices (-1 can appear if index has fewer than k vectors)
    valid = [(d, i) for d, i in zip(distances, indices) if 0 <= i < len(state.scenarios)]
    k_found = len(valid)

    if k_found == 0:
        raise HTTPException(status_code=422, detail="No similar historical scenarios found.")

    neighbours = [state.scenarios[i] for _, i in valid]
    wins = [s["outcome"]["won"] for s in neighbours]
    knn_win_rate = float(np.mean(wins))

    # 3. ML model probability
    # Enrich features with driver stats from the nearest neighbours if not supplied
    features = dict(req.features)
    if "driver_overall_win_rate" not in features and neighbours:
        # Average the neighbour driver stats as a proxy when the driver is new/unknown
        ov_rates = [s["features"].get("driver_overall_win_rate", 0.05) for s in neighbours]
        features.setdefault("driver_overall_win_rate", float(np.mean(ov_rates)))
    if "driver_circuit_win_rate" not in features and neighbours:
        bc_rates = [s["features"].get("driver_circuit_win_rate", 0.05) for s in neighbours]
        features.setdefault("driver_circuit_win_rate", float(np.mean(bc_rates)))

    # Driver/constructor standing position and pit-stop progress behave like stable
    # driver/team traits, so neighbour-averaging is a reasonable fallback when the
    # live query doesn't supply them (mirrors the win-rate pattern above).
    if "driver_standing_position" not in features and neighbours:
        vals = [
            s["features"].get("driver_standing_position_entering") for s in neighbours
            if s["features"].get("driver_standing_position_entering") is not None
        ]
        if vals:
            features.setdefault("driver_standing_position", float(np.mean(vals)))
    if "constructor_standing_position" not in features and neighbours:
        vals = [
            s["features"].get("constructor_standing_position_entering") for s in neighbours
            if s["features"].get("constructor_standing_position_entering") is not None
        ]
        if vals:
            features.setdefault("constructor_standing_position", float(np.mean(vals)))
    if "pit_stops_completed" not in features and neighbours:
        vals = [s["features"].get("pit_stops_completed", 0) for s in neighbours]
        features.setdefault("pit_stops_completed", float(np.mean(vals)))

    # Incident flags are NOT neighbour-averaged: whether a safety car happened is
    # race-specific noise, not a driver/team trait — averaging it across "similar"
    # historical races would inject spurious bias into unrelated predictions. They
    # stay unset (→ build_feature_vector's default "no incident") unless the live
    # query explicitly states one.

    model_prob = pm.predict_proba(state.model, features)

    # 4. Blend
    blended = BLEND_WEIGHT * model_prob + (1 - BLEND_WEIGHT) * knn_win_rate

    # 5. Confidence
    agreement = max(knn_win_rate, 1 - knn_win_rate)  # how one-sided the neighbours are
    has_features = bool(req.features.get("grid_position") or req.features.get("current_position"))
    confidence = _confidence(k_found, agreement, has_features)

    # 6. Top-3 supporting cases (deduplicate by race+driver)
    seen = set()
    supporting = []
    for s in neighbours:
        key = (s["features"].get("year"), s["features"].get("circuit_id"), s["features"].get("driver_id"))
        if key not in seen:
            seen.add(key)
            supporting.append(_scenario_to_case(s))
        if len(supporting) == 3:
            break

    return PredictResponse(
        knn_win_rate=round(knn_win_rate, 4),
        model_probability=round(model_prob, 4),
        blended_probability=round(blended, 4),
        confidence=confidence,
        supporting_cases=supporting,
    )


@app.post("/rebuild")
def rebuild():
    """Admin endpoint: reload artifacts from disk without restarting the service."""
    _load_artifacts()
    return {"status": "reloaded", "index_size": state.index.ntotal if state.index else 0}
