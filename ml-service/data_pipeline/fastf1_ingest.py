"""
Ingest FastF1-style lap-by-lap session CSVs (one row per lap, with sector times,
tyre life, speed traps, etc. — as exported by the `fastf1` Python package).

This is a different shape from F1DB's aggregated per-race CSVs (see ingest.py):
FastF1 gives raw lap timing, not a final classification. For qualifying/practice
sessions this module derives a classification (best lap per driver, ranked) and
tries to match it against an existing F1DB race so the result can be used as
`grid_position`/context for a prediction query.

Important: if F1DB has no *result* for the matched race yet (e.g. the session
is from a race that hasn't been run), we do NOT fabricate `won`/`finished_position`/
`dnf`. The derived scenario is written to data/processed/pending_scenarios.jsonl
with `outcome: None` instead of data/processed/scenarios.jsonl, so it never
leaks into FAISS/model training (see build_index.py) until a real result exists.

Usage (from ml-service/):
  python -m data_pipeline.fastf1_ingest data/raw/fastf1/2026-belgian-grand-prix-qualifying.csv
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

from data_pipeline.build_scenarios import build_scenario_text

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_FASTF1_DIR = RAW_DIR / "fastf1"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PENDING_PATH = PROCESSED_DIR / "pending_scenarios.jsonl"

FILENAME_RE = re.compile(r"^(\d{4})-(.+)-(\w+)$")

# F1DB abbreviates some constructor names relative to how sessions export them.
TEAM_ALIASES = {
    "red bull racing": "red bull",
    "haas f1 team": "haas",
}


def parse_filename(path: Path) -> tuple[int, str, str]:
    """'2026-Belgian Grand Prix-Qualifying.csv' -> (2026, 'Belgian Grand Prix', 'Qualifying')."""
    stem = path.stem
    m = FILENAME_RE.match(stem)
    if not m:
        raise ValueError(
            f"Could not parse '{stem}' as '<year>-<Event Name>-<Session>'. "
            "Expected FastF1-style export filename."
        )
    year, event_name, session_type = m.groups()
    return int(year), event_name, session_type


def load_session_laps(path: Path) -> pd.DataFrame:
    cols = ["Driver", "DriverNumber", "Team", "LapTime_in_seconds", "Deleted"]
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    df["Deleted"] = df["Deleted"].astype(bool)
    return df


def derive_qualifying_positions(laps_df: pd.DataFrame) -> pd.DataFrame:
    """One row per driver: best valid lap time, ranked ascending (1 = pole)."""
    valid = laps_df[(~laps_df["Deleted"]) & laps_df["LapTime_in_seconds"].notna()]

    best = (
        valid.sort_values("LapTime_in_seconds")
        .groupby("Driver", as_index=False)
        .first()[["Driver", "DriverNumber", "Team", "LapTime_in_seconds"]]
        .rename(columns={"LapTime_in_seconds": "best_lap_seconds"})
    )
    lap_counts = laps_df.groupby("Driver").size().rename("laps_completed")
    best = best.merge(lap_counts, on="Driver", how="left")

    best = best.sort_values("best_lap_seconds", ascending=True, na_position="last").reset_index(drop=True)
    best["quali_position"] = best.index + 1
    return best


def match_f1db_race(year: int, event_name: str) -> dict | None:
    grands_prix = pd.read_csv(RAW_DIR / "f1db-grands-prix.csv")
    gp_match = grands_prix[grands_prix["fullName"].str.lower() == event_name.strip().lower()]
    if gp_match.empty:
        print(f"WARNING: No F1DB grand-prix entry matches event name '{event_name}'.")
        return None
    grand_prix_id = gp_match.iloc[0]["id"]

    races = pd.read_csv(RAW_DIR / "f1db-races.csv")
    race_match = races[(races["year"] == year) & (races["grandPrixId"] == grand_prix_id)]
    if race_match.empty:
        print(f"WARNING: No F1DB race found for {year} {event_name} (grandPrixId={grand_prix_id}).")
        return None

    row = race_match.iloc[0]
    return {"race_id": row["id"], "circuit_id": row["circuitId"], "race_name": row.get("name", event_name)}


def has_f1db_result(race_id) -> bool:
    results = pd.read_csv(RAW_DIR / "f1db-races-race-results.csv", usecols=["raceId"])
    return (results["raceId"] == race_id).any()


def match_driver(driver_code: str, drivers: pd.DataFrame) -> dict | None:
    matches = drivers[drivers["abbreviation"] == driver_code]
    if matches.empty:
        print(f"WARNING: No F1DB driver matches abbreviation '{driver_code}'.")
        return None
    if len(matches) > 1:
        print(
            f"WARNING: Abbreviation '{driver_code}' matches {len(matches)} F1DB drivers "
            f"(reused over F1 history); using the youngest (most likely currently active)."
        )
        matches = matches.assign(_dob=pd.to_datetime(matches["dateOfBirth"], errors="coerce"))
        matches = matches.sort_values("_dob", ascending=False, na_position="last").head(1)
    row = matches.iloc[0]
    return {"driver_id": row["id"], "driver_name": row["name"]}


def lookup_circuit_name(circuit_id: str) -> str:
    if not circuit_id:
        return "Unknown Circuit"
    circuits = pd.read_csv(RAW_DIR / "f1db-circuits.csv")
    match = circuits[circuits["id"] == circuit_id]
    return match.iloc[0]["name"] if not match.empty else circuit_id


def match_team(team_name: str, constructors: pd.DataFrame) -> str | None:
    lookup = TEAM_ALIASES.get(team_name.strip().lower(), team_name.strip().lower())
    match = constructors[constructors["name"].str.lower() == lookup]
    if match.empty:
        return None
    return match.iloc[0]["name"]


def build_pending_scenarios(csv_path: Path) -> list[dict]:
    year, event_name, session_type = parse_filename(csv_path)
    laps_df = load_session_laps(csv_path)
    quali = derive_qualifying_positions(laps_df)

    race_info = match_f1db_race(year, event_name) or {}
    race_id = race_info.get("race_id")
    circuit_id = race_info.get("circuit_id", "")
    race_name = race_info.get("race_name", event_name)
    circuit_name = lookup_circuit_name(circuit_id)

    if race_id is not None and has_f1db_result(race_id):
        print(
            f"NOTE: F1DB already has a race result for {year} {event_name} (race_id={race_id}). "
            "Re-run the normal ingest.py/build_scenarios.py pipeline instead of using pending "
            "scenarios for this race — real outcomes are available."
        )

    drivers_df = pd.read_csv(RAW_DIR / "f1db-drivers.csv")
    constructors_df = pd.read_csv(RAW_DIR / "f1db-constructors.csv")

    scenarios = []
    for _, row in quali.iterrows():
        driver_info = match_driver(row["Driver"], drivers_df) or {}
        driver_id = driver_info.get("driver_id", row["Driver"])
        driver_name = driver_info.get("driver_name", row["Driver"])
        team_name = match_team(row["Team"], constructors_df) or row["Team"]

        pos = int(row["quali_position"])
        scenario_row = pd.Series(
            {
                "year": year,
                "circuit_id": circuit_id,
                "circuit_name": circuit_name,
                "driver_id": driver_id,
                "driver_name": driver_name,
                "team_name": team_name,
                "race_name": race_name,
                "grid_position": pos,
                "weather": "dry",
            }
        )
        text = build_scenario_text(scenario_row)
        if pd.notna(row["best_lap_seconds"]):
            text += f", qualified P{pos} with a best lap of {row['best_lap_seconds']:.3f}s ({session_type})"
        else:
            text += f", qualified P{pos} ({session_type}, no valid lap time)"

        scenario_id = f"{year}_{circuit_id or event_name.replace(' ', '_')}_{driver_id}_pending"
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "status": "pending",
                "text": text,
                "features": {
                    "year": year,
                    "circuit_id": circuit_id,
                    "circuit_name": circuit_name,
                    "driver_id": driver_id,
                    "driver_name": driver_name,
                    "team_name": team_name,
                    "race_name": race_name,
                    "grid_position": pos,
                    "weather": "dry",
                    "total_laps": None,
                    "best_lap_seconds": float(row["best_lap_seconds"]) if pd.notna(row["best_lap_seconds"]) else None,
                    "laps_completed": int(row["laps_completed"]),
                },
                "outcome": None,
            }
        )
    return scenarios


def save_pending_scenarios(scenarios: list[dict]) -> Path:
    """Upsert by scenario_id into data/processed/pending_scenarios.jsonl."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    if PENDING_PATH.exists():
        with open(PENDING_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    existing[doc["scenario_id"]] = doc

    for doc in scenarios:
        existing[doc["scenario_id"]] = doc

    with open(PENDING_PATH, "w", encoding="utf-8") as f:
        for doc in existing.values():
            f.write(json.dumps(doc) + "\n")

    print(f"Wrote {len(existing):,} pending scenarios to {PENDING_PATH}")
    return PENDING_PATH


def build_predict_payloads(scenarios: list[dict]) -> list[dict]:
    """One {query_text, features} payload per driver, ready to POST to the ML /predict endpoint."""
    payloads = []
    for s in scenarios:
        f = s["features"]
        payloads.append(
            {
                "query_text": s["text"],
                "features": {
                    "grid_position": f["grid_position"],
                    "current_position": f["grid_position"],
                    "weather": f["weather"],
                    "race_progress_pct": 0.0,
                },
            }
        )
    return payloads


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else str(RAW_FASTF1_DIR / "2026-belgian-grand-prix-qualifying.csv")
    scenarios = build_pending_scenarios(Path(csv_arg))

    print(f"\nDerived qualifying classification ({len(scenarios)} drivers):")
    for s in sorted(scenarios, key=lambda s: s["features"]["grid_position"]):
        f = s["features"]
        print(f"  P{f['grid_position']:>2}  {f['driver_id']:<20} {f['team_name']:<20} best={f['best_lap_seconds']}")

    save_pending_scenarios(scenarios)

    print("\nSample /predict payload (POST to the ML service):")
    payloads = build_predict_payloads(scenarios)
    print(json.dumps(payloads[0], indent=2))
