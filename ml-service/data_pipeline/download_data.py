"""
Download the latest F1DB CSV release from GitHub.

Usage:
  python -m data_pipeline.download_data

Downloads and extracts to:
  ml-service/data/raw/

F1DB GitHub: https://github.com/f1db/f1db
"""
import io
import zipfile
import sys
from pathlib import Path

import requests

GITHUB_API = "https://api.github.com/repos/f1db/f1db/releases/latest"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def download_latest():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching latest F1DB release info from GitHub...")
    resp = requests.get(GITHUB_API, timeout=30)
    resp.raise_for_status()
    release = resp.json()

    tag = release["tag_name"]
    print(f"Latest release: {tag}")

    # Find the CSV zip asset
    asset = next(
        (a for a in release["assets"] if a["name"].endswith("-csv.zip")),
        None,
    )
    if asset is None:
        # Fallback: look for any zip
        asset = next(
            (a for a in release["assets"] if a["name"].endswith(".zip")),
            None,
        )
    if asset is None:
        print("ERROR: Could not find a CSV zip asset in the release.")
        print("Assets found:", [a["name"] for a in release["assets"]])
        sys.exit(1)

    url = asset["browser_download_url"]
    name = asset["name"]
    print(f"Downloading {name} ({asset['size'] // 1024} KB)...")

    dl = requests.get(url, timeout=120, stream=True)
    dl.raise_for_status()

    content = b""
    for chunk in dl.iter_content(chunk_size=65536):
        content += chunk
        print(f"\r  {len(content) // 1024} KB", end="", flush=True)
    print()

    print(f"Extracting to {RAW_DIR}...")
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        zf.extractall(RAW_DIR)

    print("Done. Files in data/raw/:")
    for p in sorted(RAW_DIR.rglob("*.csv")):
        print(f"  {p.relative_to(RAW_DIR)}")


if __name__ == "__main__":
    download_latest()
