"""
Build (or rebuild) the FAISS index and train the sklearn probability model.

Steps:
  1. Load scenarios.jsonl
  2. Compute derived stats: driver_circuit_win_rate, driver_overall_win_rate
  3. Embed all scenario texts
  4. Build and save FAISS index
  5. Train and save sklearn model
  6. Save enriched scenarios (with derived stats) for later use as supporting cases

Output artifacts (data/artifacts/):
  faiss.index
  model.pkl
  scenarios_enriched.jsonl   ← scenarios + derived stats, used for supporting cases

Usage (from ml-service/):
  python -m data_pipeline.build_index
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.build_scenarios import load_scenarios, build_scenarios, save_scenarios
from data_pipeline.ingest import load_raw
import embedder as emb
import faiss_index as fi
import probability_model as pm

ARTIFACTS_DIR = Path(__file__).parent.parent / "data" / "artifacts"
ENRICHED_PATH = ARTIFACTS_DIR / "scenarios_enriched.jsonl"
INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"


def compute_driver_stats(scenarios: list[dict]) -> dict:
    """
    Compute per-driver and per-driver-circuit win rates from the full scenario set.

    NOTE: This uses the full dataset (no temporal split) for the MVP.
    For production, use only past-race data at prediction time to avoid leakage.
    """
    # driver_id → {wins, races}
    overall: dict[str, dict] = defaultdict(lambda: {"wins": 0, "races": 0})
    # (driver_id, circuit_id) → {wins, races}
    by_circuit: dict[tuple, dict] = defaultdict(lambda: {"wins": 0, "races": 0})

    for s in scenarios:
        f = s["features"]
        o = s["outcome"]
        did = f["driver_id"]
        cid = f["circuit_id"]
        overall[did]["races"] += 1
        by_circuit[(did, cid)]["races"] += 1
        if o["won"]:
            overall[did]["wins"] += 1
            by_circuit[(did, cid)]["wins"] += 1

    return {"overall": overall, "by_circuit": by_circuit}


def enrich(scenarios: list[dict], stats: dict) -> list[dict]:
    enriched = []
    for s in scenarios:
        s = dict(s)
        f = dict(s["features"])
        did = f["driver_id"]
        cid = f["circuit_id"]
        ov = stats["overall"].get(did, {"wins": 0, "races": 1})
        bc = stats["by_circuit"].get((did, cid), {"wins": 0, "races": 1})
        f["driver_overall_win_rate"] = ov["wins"] / max(ov["races"], 1)
        f["driver_circuit_win_rate"] = bc["wins"] / max(bc["races"], 1)
        s["features"] = f
        enriched.append(s)
    return enriched


def build_feature_matrix(scenarios: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Build the (X, y) arrays for sklearn training."""
    X_rows, y_rows = [], []
    for s in scenarios:
        f = s["features"]
        o = s["outcome"]
        # Skip rows with missing grid position
        if f.get("grid_position") is None:
            continue
        fv = pm.build_feature_vector({
            "grid_position": f.get("grid_position"),
            "weather": f.get("weather", "dry"),
            "driver_circuit_win_rate": f.get("driver_circuit_win_rate", 0.05),
            "driver_overall_win_rate": f.get("driver_overall_win_rate", 0.05),
            # Real values from the scenario itself: pre_race scenarios correctly encode
            # current_position=grid/progress=0, lap_snapshot scenarios carry the true
            # mid-race position/progress reconstructed from lap_times.csv.
            "current_position": f.get("current_position", f.get("grid_position")),
            "race_progress_pct": f.get("race_progress_pct", 0.0),
            "driver_standing_position": f.get("driver_standing_position_entering"),
            "constructor_standing_position": f.get("constructor_standing_position_entering"),
            "pit_stops_completed": f.get("pit_stops_completed", 0),
            "safety_car_active": f.get("safety_car_active", False),
            "vsc_active": f.get("vsc_active", False),
            "red_flag_occurred": f.get("red_flag_occurred", False),
        })
        X_rows.append(fv)
        y_rows.append(1 if o["won"] else 0)
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


def run():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Loading scenarios ===")
    scenarios = load_scenarios()
    print(f"  {len(scenarios):,} scenarios loaded")

    print("\n=== Step 2: Computing driver stats ===")
    stats = compute_driver_stats(scenarios)
    print(f"  {len(stats['overall'])} unique drivers")
    enriched = enrich(scenarios, stats)

    with open(ENRICHED_PATH, "w", encoding="utf-8") as f:
        for doc in enriched:
            f.write(json.dumps(doc) + "\n")
    print(f"  Enriched scenarios saved → {ENRICHED_PATH}")

    print("\n=== Step 3: Embedding scenario texts ===")
    texts = [s["text"] for s in enriched]
    batch_size = 256
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings_list.append(emb.embed(batch))
        print(f"  Embedded {min(i + batch_size, len(texts)):,}/{len(texts):,}", end="\r")
    embeddings = np.vstack(embeddings_list)
    print(f"\n  Embedding shape: {embeddings.shape}")

    print("\n=== Step 4: Building FAISS index ===")
    index = fi.build_index(embeddings)
    fi.save_index(index, INDEX_PATH)
    print(f"  Index saved → {INDEX_PATH}  ({index.ntotal} vectors, dim={embeddings.shape[1]})")

    print("\n=== Step 5: Training probability model ===")
    X, y = build_feature_matrix(enriched)
    print(f"  Training on {len(X):,} samples ({y.sum()} wins, {(~y.astype(bool)).sum()} non-wins)")
    model = pm.train(X, y)
    pm.save_model(model, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")

    print("\nBuild complete.")


if __name__ == "__main__":
    run()
