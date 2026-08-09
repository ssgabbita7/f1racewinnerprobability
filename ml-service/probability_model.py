"""
Lightweight sklearn probability model for win prediction.

Feature vector (10 dimensions):
  0  grid_position_norm                   (20 - grid) / 19  → 1.0 = pole, 0.0 = last
  1  weather_wet                          1 if wet/mixed, else 0
  2  driver_circuit_win_rate              career wins at circuit / career starts at circuit
  3  driver_overall_win_rate              career wins / career starts
  4  current_position_norm                (20 - position) / 19  → defaults to grid if unknown
  5  race_progress_pct                    lap / total_laps  → 0.0 at start, 1.0 at finish
  6  driver_standing_position_norm        rank-normalized championship position entering the race
  7  constructor_standing_position_norm   rank-normalized constructor standing entering the race
  8  pit_stops_completed_norm             stops so far / 4, capped at 1.0
  9  incident_active                      1 if a safety car / VSC / red flag has occurred, else 0
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


FEATURE_NAMES = [
    "grid_position_norm",
    "weather_wet",
    "driver_circuit_win_rate",
    "driver_overall_win_rate",
    "current_position_norm",
    "race_progress_pct",
    "driver_standing_position_norm",
    "constructor_standing_position_norm",
    "pit_stops_completed_norm",
    "incident_active",
]


def build_feature_vector(features: dict) -> np.ndarray:
    """Convert a feature dict to the 10-dim numpy vector expected by the model."""
    grid_norm = _grid_norm(features.get("grid_position"))
    pit_stops = float(features.get("pit_stops_completed", 0) or 0)
    incident_active = 1.0 if (
        features.get("safety_car_active") or features.get("vsc_active") or features.get("red_flag_occurred")
    ) else 0.0
    return np.array([
        grid_norm,
        1.0 if str(features.get("weather", "dry")).lower() in ("wet", "mixed") else 0.0,
        float(features.get("driver_circuit_win_rate", 0.05)),
        float(features.get("driver_overall_win_rate", 0.05)),
        # current_position_norm falls back to grid if not supplied
        _grid_norm(features.get("current_position")) if features.get("current_position") else grid_norm,
        float(features.get("race_progress_pct", 0.0)),
        _rank_norm(features.get("driver_standing_position")),
        _rank_norm(features.get("constructor_standing_position")),
        min(pit_stops / 4.0, 1.0),
        incident_active,
    ], dtype=np.float32)


def _grid_norm(pos) -> float:
    if pos is None:
        return 0.5
    try:
        return (20 - int(pos)) / 19
    except (ValueError, TypeError):
        return 0.5


def _rank_norm(pos, default: float = 0.5, worst: int = 20) -> float:
    """Normalize a championship standing position to [0, 1], 1.0 = leader. Clipped (unlike _grid_norm) since standings positions can exceed `worst` in large historical fields."""
    if pos is None:
        return default
    try:
        return max(0.0, min(1.0, (worst + 1 - int(pos)) / worst))
    except (ValueError, TypeError):
        return default


def train(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Train a gradient-boosting classifier wrapped in a StandardScaler pipeline."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])
    model.fit(X, y)

    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(f"[model] 5-fold ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")
    return model


def predict_proba(model: Pipeline, features: dict) -> float:
    """Return the probability of winning (class 1) for a single scenario."""
    x = build_feature_vector(features).reshape(1, -1)
    proba = model.predict_proba(x)[0]
    # Find the index for class=1 (win)
    classes = list(model.classes_)
    win_idx = classes.index(1) if 1 in classes else 1
    return float(proba[win_idx])


def save_model(model: Pipeline, path: str | Path) -> None:
    joblib.dump(model, str(path))


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(str(path))
