"""
Join/derivation helpers for the Ergast-schema dataset (data/raw/ergast/).

Owns everything that isn't a straight column-rename: DNF-correct status
classification, total-laps derivation (Ergast's races.csv has no laps
column), pit-stop event loading, safety-car/VSC/red-flag incident windows,
and leakage-safe "standings entering this race" computation.

Usage (from ml-service/): imported by ingest.py and build_scenarios.py.
"""
import json
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "ergast"

_LAPS_STATUS_RE = re.compile(r"^\+\d+ Laps?$")


def _read_csv(name: str) -> pd.DataFrame:
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            "Copy the Ergast dataset CSVs into ml-service/data/raw/ergast/ first."
        )
    # Ergast encodes NULL as the literal string "\N"
    return pd.read_csv(path, low_memory=False, na_values=["\\N"])


def load_status_map() -> pd.DataFrame:
    """status.csv -> [status_id, status_text, is_dnf]."""
    df = _read_csv("status.csv").rename(columns={"statusId": "status_id", "status": "status_text"})
    df["is_dnf"] = ~(
        (df["status_text"] == "Finished") | df["status_text"].str.match(_LAPS_STATUS_RE)
    )
    return df[["status_id", "status_text", "is_dnf"]]


def compute_total_laps(results: pd.DataFrame, lap_times_raw: pd.DataFrame) -> pd.Series:
    """Return a Series indexed by race_id giving the race distance in laps."""
    from_laps = lap_times_raw.groupby("race_id")["lap"].max()
    from_results = results.groupby("race_id")["laps_completed"].max()
    total_laps = from_laps.combine_first(from_results)
    total_laps.name = "total_laps"
    return total_laps


def load_pit_stops() -> pd.DataFrame:
    """pit_stops.csv -> [race_id, driver_id, stop_number, lap] (raw event table)."""
    df = _read_csv("pit_stops.csv").rename(
        columns={"raceId": "race_id", "driverId": "driver_id", "stop": "stop_number", "lap": "lap"}
    )
    return df[["race_id", "driver_id", "stop_number", "lap"]]


def load_races_min() -> pd.DataFrame:
    """races.csv -> [race_id, year, round, race_name] (lightweight, for standings/incident joins)."""
    df = _read_csv("races.csv").rename(columns={"raceId": "race_id", "name": "race_name"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df[["race_id", "year", "round", "race_name"]]


def load_lap_times() -> pd.DataFrame:
    """lap_times.csv -> [race_id, driver_id, lap, position]."""
    df = _read_csv("lap_times.csv").rename(
        columns={"raceId": "race_id", "driverId": "driver_id", "lap": "lap", "position": "position"}
    )
    return df[["race_id", "driver_id", "lap", "position"]]


def _race_text_lookup(races: pd.DataFrame) -> dict:
    """{'{year} {race_name}': race_id} for joining the text-keyed incident tables."""
    keys = races["year"].astype(int).astype(str) + " " + races["race_name"].astype(str)
    return dict(zip(keys, races["race_id"]))


def load_incident_windows(races: pd.DataFrame) -> pd.DataFrame:
    """
    Build [race_id, incident_type, start_lap, end_lap] from safety_cars.csv,
    virtual_safety_cars.csv, virtual_safety_car_estimates.json, and red_flags.csv.

    incident_type in {"SC", "VSC", "VSC_EST", "RED"}.
    """
    lookup = _race_text_lookup(races)
    rows = []

    def _add(race_key: str, incident_type: str, start_lap, end_lap):
        race_id = lookup.get(race_key)
        if race_id is None:
            return
        rows.append({"race_id": race_id, "incident_type": incident_type, "start_lap": start_lap, "end_lap": end_lap})

    sc = _read_csv("safety_cars.csv")
    for _, r in sc.iterrows():
        end_lap = r["Retreated"] if pd.notna(r.get("Retreated")) else r["Deployed"]
        _add(r["Race"], "SC", r["Deployed"], end_lap)

    vsc = _read_csv("virtual_safety_cars.csv")
    vsc_race_keys = set(vsc["Race"])
    for _, r in vsc.iterrows():
        end_lap = r["Retreated"] if pd.notna(r.get("Retreated")) else r["Deployed"]
        _add(r["Race"], "VSC", r["Deployed"], end_lap)

    vsc_est_path = RAW_DIR / "virtual_safety_car_estimates.json"
    if vsc_est_path.exists():
        with open(vsc_est_path, encoding="utf-8") as f:
            estimates = json.load(f)
        for race_key, laps in estimates.items():
            if race_key in vsc_race_keys or not laps:
                continue
            _add(race_key, "VSC_EST", min(laps), max(laps))

    red = _read_csv("red_flags.csv")
    for _, r in red.iterrows():
        _add(r["Race"], "RED", r["Lap"], r["Lap"])

    if not rows:
        return pd.DataFrame(columns=["race_id", "incident_type", "start_lap", "end_lap"])
    return pd.DataFrame(rows)


def _standings_entering_race(races: pd.DataFrame, standings: pd.DataFrame, id_col: str, prefix: str) -> pd.DataFrame:
    """
    Shared algorithm for driver/constructor standings-entering-race.

    standings is expected to already carry [race_id, <id_col>, points, position, wins].
    Returns [race_id, <id_col>, {prefix}_position_entering, {prefix}_points_entering,
             {prefix}_wins_entering].
    """
    merged = standings.merge(races[["race_id", "year", "round"]], on="race_id", how="left")
    merged = merged.sort_values([id_col, "year", "round"])

    grouped = merged.groupby([id_col, "year"], group_keys=False)
    merged[f"{prefix}_position_entering"] = grouped["position"].shift(1)
    merged[f"{prefix}_points_entering"] = grouped["points"].shift(1)
    merged[f"{prefix}_wins_entering"] = grouped["wins"].shift(1)

    return merged[
        ["race_id", id_col, f"{prefix}_position_entering", f"{prefix}_points_entering", f"{prefix}_wins_entering"]
    ]


def compute_driver_standings_entering_race(races: pd.DataFrame) -> pd.DataFrame:
    standings = _read_csv("driver_standings.csv").rename(
        columns={"raceId": "race_id", "driverId": "driver_id", "points": "points", "position": "position", "wins": "wins"}
    )[["race_id", "driver_id", "points", "position", "wins"]]
    return _standings_entering_race(races, standings, "driver_id", "driver_standing")


def compute_constructor_standings_entering_race(races: pd.DataFrame) -> pd.DataFrame:
    standings = _read_csv("constructor_standings.csv").rename(
        columns={"raceId": "race_id", "constructorId": "constructor_id", "points": "points", "position": "position", "wins": "wins"}
    )[["race_id", "constructor_id", "points", "position", "wins"]]
    return _standings_entering_race(races, standings, "constructor_id", "constructor_standing")
