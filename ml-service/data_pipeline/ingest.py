"""
Ingest the Ergast-schema dataset into a single clean DataFrame.

Source: ml-service/data/raw/ergast/*.csv (see README for how to obtain this
dataset — it's the classic Ergast/Kaggle "Formula 1 World Championship"
tables, self-consistent on raceId/driverId/constructorId).

Usage (from ml-service/):
  python -m data_pipeline.ingest
"""
from pathlib import Path

import pandas as pd

from data_pipeline import ergast_helpers as eh

RAW_DIR = eh.RAW_DIR
WEATHER_CSV = Path(__file__).parent.parent / "data" / "weather_lookup.csv"

# ── Column mappings (Ergast → our snake_case) ──────────────────────────────────

RACES_COLS = {
    "raceId": "race_id",
    "year": "year",
    "round": "round",
    "circuitId": "circuit_id",
    "name": "race_name",
    "date": "date",
}

RESULTS_COLS = {
    "raceId": "race_id",
    "driverId": "driver_id",
    "constructorId": "constructor_id",
    "grid": "grid_position",
    "position": "finished_position",
    "laps": "laps_completed",
    "points": "points",
    "statusId": "status_id",
}

DRIVERS_COLS = {
    "driverId": "driver_id",
    "forename": "forename",
    "surname": "surname",
    "code": "driver_code",
}

CIRCUITS_COLS = {
    "circuitId": "circuit_id",
    "circuitRef": "circuit_ref",
    "name": "circuit_name",
    "location": "location",
    "country": "country",
}

CONSTRUCTORS_COLS = {
    "constructorId": "constructor_id",
    "name": "team_name",
}

# weather_lookup.csv uses F1DB-style hyphenated slugs; remap to Ergast circuitRef
WEATHER_CIRCUIT_REMAP = {
    "korea": "yeongam",
    "montreal": "villeneuve",
    "hockenheim": "hockenheimring",
    "istanbul-park": "istanbul",
    "albert-park": "albert_park",
}


def _load(name: str, col_map: dict, required: list[str] | None = None) -> pd.DataFrame:
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            "Copy the Ergast dataset CSVs into ml-service/data/raw/ergast/ first."
        )
    df = pd.read_csv(path, low_memory=False, na_values=["\\N"])

    missing = [c for c in col_map if c not in df.columns]
    if missing:
        print(
            f"WARNING: {name} is missing expected columns: {missing}\n"
            f"  Available columns: {list(df.columns)}\n"
            "  Update *_COLS in ingest.py to match."
        )
        col_map = {k: v for k, v in col_map.items() if k in df.columns}

    df = df[list(col_map.keys())].rename(columns=col_map)

    if required:
        still_missing = [c for c in required if c not in df.columns]
        if still_missing:
            raise ValueError(f"Required columns missing after rename: {still_missing}")

    return df


def load_raw() -> pd.DataFrame:
    """Return a merged DataFrame with one row per driver per race (all entrants, not just classified finishers)."""
    races = _load("races.csv", RACES_COLS, required=["race_id", "year", "circuit_id"])
    results = _load("results.csv", RESULTS_COLS, required=["race_id", "driver_id"])
    drivers = _load("drivers.csv", DRIVERS_COLS, required=["driver_id"])
    circuits = _load("circuits.csv", CIRCUITS_COLS, required=["circuit_id"])
    constructors = _load("constructors.csv", CONSTRUCTORS_COLS, required=["constructor_id"])
    status = eh.load_status_map()
    lap_times = eh.load_lap_times()

    df = (
        results
        .merge(races, on="race_id", how="left")
        .merge(drivers, on="driver_id", how="left")
        .merge(circuits, on="circuit_id", how="left")
        .merge(constructors, on="constructor_id", how="left")
        .merge(status, on="status_id", how="left")
    )

    # Derived columns
    df["driver_name"] = (df["forename"].fillna("") + " " + df["surname"].fillna("")).str.strip()
    df["grid_position"] = pd.to_numeric(df.get("grid_position"), errors="coerce")
    df["finished_position"] = pd.to_numeric(df.get("finished_position"), errors="coerce")
    df["won"] = df["finished_position"] == 1
    df["dnf"] = df["is_dnf"].fillna(True)  # unknown status → treat conservatively as DNF
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    total_laps = eh.compute_total_laps(df[["race_id", "laps_completed"]], lap_times)
    df = df.merge(total_laps.rename("total_laps"), on="race_id", how="left")

    # Weather from lookup (remap F1DB-style slugs to Ergast circuitRef)
    weather = pd.read_csv(WEATHER_CSV)
    weather["circuit_id"] = weather["circuit_id"].str.lower().str.strip()
    weather["circuit_ref"] = weather["circuit_id"].map(lambda c: WEATHER_CIRCUIT_REMAP.get(c, c))
    df = df.merge(weather[["year", "circuit_ref", "weather"]], on=["year", "circuit_ref"], how="left")
    df["weather"] = df["weather"].fillna("dry")

    # Only require a driver identity — finished_position is legitimately null for
    # unclassified/retired entrants (Ergast leaves `position` blank for those and
    # records the real finishing order in `positionOrder` instead); dropping on it
    # would silently discard ~40% of rows, exactly the DNF/non-win examples the
    # model needs most.
    df.dropna(subset=["driver_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Loaded {len(df):,} race-driver rows spanning {df['year'].min():.0f}–{df['year'].max():.0f}")
    print(f"  DNF rate: {df['dnf'].mean():.1%}  |  Win rate: {df['won'].mean():.1%}")
    return df


if __name__ == "__main__":
    df = load_raw()
    print(df.head())
    print(df.dtypes)
