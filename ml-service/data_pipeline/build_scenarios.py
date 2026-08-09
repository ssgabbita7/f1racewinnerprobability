"""
Build "race scenario" documents from the ingested DataFrame.

Two kinds of scenario are produced:
  - pre_race:    one per driver per race, snapshot at the start (grid position,
                  progress 0%) — always available.
  - lap_snapshot: several per driver per race, snapshot at a real mid-race lap
                  using true running position from lap_times.csv — only for
                  driver-races with lap-time coverage (1982+, partial gaps).

Each scenario is a JSON object with:
  - scenario_id: unique string key
  - text: natural-language description (used for embedding)
  - features: structured dict (used by the sklearn model)
  - outcome: {won, finished_position, dnf}

Output: data/processed/scenarios.jsonl

Usage (from ml-service/):
  python -m data_pipeline.build_scenarios
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from data_pipeline.ingest import load_raw
from data_pipeline import ergast_helpers as eh

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "scenarios.jsonl"

LAP_SNAPSHOT_FRACTIONS = [0.25, 0.5, 0.75, 0.90]


def _era(year: int) -> str:
    if year < 1966:
        return "1.5L era"
    if year < 1977:
        return "3L era"
    if year < 1989:
        return "turbo era"
    if year < 2006:
        return "V10/V12 era"
    if year < 2014:
        return "V8 era"
    return "hybrid era"


def _ordinal(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "unknown"
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return {1: f"{n}st", 2: f"{n}nd", 3: f"{n}rd"}.get(n % 10, f"{n}th")


def _safe_int(v):
    return int(v) if pd.notna(v) else None


def _safe_float(v):
    return float(v) if pd.notna(v) else None


def build_scenario_text(row: pd.Series) -> str:
    """Produce a natural-language sentence describing the pre-race setup."""
    driver = row.get("driver_name") or "Unknown Driver"
    circuit = row.get("circuit_name") or row.get("circuit_id") or "Unknown Circuit"
    team = row.get("team_name") or "Unknown Team"
    year = int(row["year"]) if pd.notna(row.get("year")) else "?"
    race_name = row.get("race_name") or f"{year} race"
    weather = str(row.get("weather", "dry")).lower()
    grid = row.get("grid_position")
    era = _era(year) if isinstance(year, int) else ""

    grid_str = f"started {_ordinal(grid)}" if pd.notna(grid) else "grid position unknown"

    return (
        f"{year} {race_name} at {circuit}: "
        f"{driver} driving for {team}, {grid_str}, "
        f"{weather} conditions, {era}"
    ).strip(", ")


def build_lap_snapshot_text(
    row: pd.Series,
    lap: int,
    position: int,
    total_laps: int,
    pit_stops_completed: int,
    safety_car_active: bool,
    vsc_active: bool,
    red_flag_occurred: bool,
) -> str:
    """Produce a natural-language sentence describing a real mid-race snapshot."""
    driver = row.get("driver_name") or "Unknown Driver"
    circuit = row.get("circuit_name") or row.get("circuit_id") or "Unknown Circuit"
    team = row.get("team_name") or "Unknown Team"
    year = int(row["year"]) if pd.notna(row.get("year")) else "?"
    race_name = row.get("race_name") or f"{year} race"
    weather = str(row.get("weather", "dry")).lower()
    grid = row.get("grid_position")
    era = _era(year) if isinstance(year, int) else ""

    grid_str = f"started {_ordinal(grid)}" if pd.notna(grid) else "grid position unknown"

    if safety_car_active:
        incident_clause = "safety car deployed, "
    elif vsc_active:
        incident_clause = "virtual safety car active, "
    elif red_flag_occurred:
        incident_clause = "red flag earlier in the race, "
    else:
        incident_clause = ""

    pit_clause = ""
    if pit_stops_completed:
        s = "s" if pit_stops_completed != 1 else ""
        pit_clause = f"completed {pit_stops_completed} pit stop{s}, "

    return (
        f"{year} {race_name} at {circuit}: "
        f"{driver} driving for {team}, {grid_str}, "
        f"now {_ordinal(position)} on lap {lap} of {total_laps}, "
        f"{weather} conditions, {incident_clause}{pit_clause}{era}"
    ).strip(", ")


def attach_standings(df: pd.DataFrame) -> pd.DataFrame:
    """Merge in leakage-safe driver/constructor championship standings entering each race."""
    races_min = eh.load_races_min()
    driver_standings = eh.compute_driver_standings_entering_race(races_min)
    constructor_standings = eh.compute_constructor_standings_entering_race(races_min)

    df = df.merge(
        driver_standings[
            ["race_id", "driver_id", "driver_standing_position_entering", "driver_standing_points_entering"]
        ],
        on=["race_id", "driver_id"],
        how="left",
    )
    df = df.merge(
        constructor_standings[
            ["race_id", "constructor_id", "constructor_standing_position_entering", "constructor_standing_points_entering"]
        ],
        on=["race_id", "constructor_id"],
        how="left",
    )
    return df


def build_scenarios(df: pd.DataFrame) -> list[dict]:
    """One pre-race scenario per driver-race. `df` should already carry standings-entering columns."""
    scenarios = []
    for _, row in df.iterrows():
        grid = row.get("grid_position")
        pos = row.get("finished_position")
        scenario_id = (
            f"{int(row['year']) if pd.notna(row.get('year')) else 'X'}_"
            f"{str(row.get('circuit_id', 'X')).replace(' ', '_')}_"
            f"{str(row.get('driver_id', 'X')).replace(' ', '_')}"
        )
        doc = {
            "scenario_id": scenario_id,
            "text": build_scenario_text(row),
            "features": {
                "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                "circuit_id": str(row.get("circuit_id", "")),
                "circuit_name": str(row.get("circuit_name", "")),
                "driver_id": str(row.get("driver_id", "")),
                "driver_name": str(row.get("driver_name", "")),
                "team_name": str(row.get("team_name", "")),
                "race_name": str(row.get("race_name", "")),
                "grid_position": int(grid) if pd.notna(grid) else None,
                "weather": str(row.get("weather", "dry")),
                "total_laps": int(row["total_laps"]) if pd.notna(row.get("total_laps")) else None,
                "scenario_type": "pre_race",
                "lap": None,
                # Pre-race: nothing has happened yet — these are the correct, not hardcoded, values.
                "current_position": int(grid) if pd.notna(grid) else None,
                "race_progress_pct": 0.0,
                "pit_stops_completed": 0,
                "safety_car_active": False,
                "vsc_active": False,
                "red_flag_occurred": False,
                "driver_standing_position_entering": _safe_int(row.get("driver_standing_position_entering")),
                "driver_standing_points_entering": _safe_float(row.get("driver_standing_points_entering")),
                "constructor_standing_position_entering": _safe_int(row.get("constructor_standing_position_entering")),
                "constructor_standing_points_entering": _safe_float(row.get("constructor_standing_points_entering")),
            },
            "outcome": {
                "finished_position": int(pos) if pd.notna(pos) else None,
                "won": bool(row.get("won", False)),
                "dnf": bool(row.get("dnf", False)),
            },
        }
        scenarios.append(doc)
    return scenarios


def build_lap_snapshots(df: pd.DataFrame) -> list[dict]:
    """
    Sample real mid-race scenarios from lap_times.csv, at LAP_SNAPSHOT_FRACTIONS of
    each race's distance. `df` should already carry standings-entering columns.
    """
    lap_times_raw = eh.load_lap_times()
    pit_stops = eh.load_pit_stops()
    races_min = eh.load_races_min()
    incident_windows = eh.load_incident_windows(races_min)

    lap_times_by_key = {
        key: g.sort_values("lap") for key, g in lap_times_raw.groupby(["race_id", "driver_id"])
    }
    pit_laps_by_key = pit_stops.groupby(["race_id", "driver_id"])["lap"].apply(list).to_dict()

    incidents_by_race = defaultdict(list)
    for row in incident_windows.itertuples(index=False):
        incidents_by_race[row.race_id].append((row.incident_type, row.start_lap, row.end_lap))

    scenarios = []
    for _, row in df.iterrows():
        race_id = row.get("race_id")
        driver_id = row.get("driver_id")
        total_laps = row.get("total_laps")
        if pd.isna(total_laps) or total_laps <= 1:
            continue

        driver_laps = lap_times_by_key.get((race_id, driver_id))
        if driver_laps is None or driver_laps.empty:
            continue

        total_laps = int(total_laps)
        max_recorded_lap = int(driver_laps["lap"].max())
        pit_laps = pit_laps_by_key.get((race_id, driver_id), [])
        race_incidents = incidents_by_race.get(race_id, [])

        for frac in LAP_SNAPSHOT_FRACTIONS:
            target_lap = max(1, min(int(round(frac * total_laps)), total_laps - 1))
            if target_lap > max_recorded_lap:
                continue  # driver retired before this snapshot point

            candidates = driver_laps[driver_laps["lap"] <= target_lap]
            if candidates.empty:
                continue
            snap = candidates.iloc[-1]
            snapshot_lap = int(snap["lap"])
            current_position = snap["position"]
            if pd.isna(current_position):
                continue
            current_position = int(current_position)

            pit_stops_completed = sum(1 for lp in pit_laps if lp <= snapshot_lap)
            safety_car_active = any(t == "SC" and s <= snapshot_lap <= e for t, s, e in race_incidents)
            vsc_active = any(t in ("VSC", "VSC_EST") and s <= snapshot_lap <= e for t, s, e in race_incidents)
            red_flag_occurred = any(t == "RED" and s <= snapshot_lap for t, s, e in race_incidents)

            year = int(row["year"]) if pd.notna(row.get("year")) else None
            grid = row.get("grid_position")

            scenario_id = f"{year}_{row.get('circuit_id', 'X')}_{driver_id}_lap{snapshot_lap}"
            text = build_lap_snapshot_text(
                row, snapshot_lap, current_position, total_laps,
                pit_stops_completed, safety_car_active, vsc_active, red_flag_occurred,
            )

            doc = {
                "scenario_id": scenario_id,
                "text": text,
                "features": {
                    "year": year,
                    "circuit_id": str(row.get("circuit_id", "")),
                    "circuit_name": str(row.get("circuit_name", "")),
                    "driver_id": str(driver_id),
                    "driver_name": str(row.get("driver_name", "")),
                    "team_name": str(row.get("team_name", "")),
                    "race_name": str(row.get("race_name", "")),
                    "grid_position": int(grid) if pd.notna(grid) else None,
                    "weather": str(row.get("weather", "dry")),
                    "total_laps": total_laps,
                    "scenario_type": "lap_snapshot",
                    "lap": snapshot_lap,
                    "current_position": current_position,
                    "race_progress_pct": snapshot_lap / total_laps,
                    "pit_stops_completed": pit_stops_completed,
                    "safety_car_active": bool(safety_car_active),
                    "vsc_active": bool(vsc_active),
                    "red_flag_occurred": bool(red_flag_occurred),
                    "driver_standing_position_entering": _safe_int(row.get("driver_standing_position_entering")),
                    "driver_standing_points_entering": _safe_float(row.get("driver_standing_points_entering")),
                    "constructor_standing_position_entering": _safe_int(row.get("constructor_standing_position_entering")),
                    "constructor_standing_points_entering": _safe_float(row.get("constructor_standing_points_entering")),
                },
                "outcome": {
                    "finished_position": _safe_int(row.get("finished_position")),
                    "won": bool(row.get("won", False)),
                    "dnf": bool(row.get("dnf", False)),
                },
            }
            scenarios.append(doc)

    return scenarios


def save_scenarios(scenarios: list[dict]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for doc in scenarios:
            f.write(json.dumps(doc) + "\n")
    print(f"Wrote {len(scenarios):,} scenarios to {OUT_PATH}")


def load_scenarios() -> list[dict]:
    with open(OUT_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    df = load_raw()
    df = attach_standings(df)

    print("Building pre-race scenarios...")
    pre_race = build_scenarios(df)
    print(f"  {len(pre_race):,} pre-race scenarios")

    print("Building lap-snapshot scenarios...")
    lap_snapshots = build_lap_snapshots(df)
    print(f"  {len(lap_snapshots):,} lap-snapshot scenarios")

    all_scenarios = pre_race + lap_snapshots
    save_scenarios(all_scenarios)
    print(f"Sample pre-race:\n{json.dumps(pre_race[0], indent=2)}")
    if lap_snapshots:
        print(f"Sample lap-snapshot:\n{json.dumps(lap_snapshots[0], indent=2)}")
