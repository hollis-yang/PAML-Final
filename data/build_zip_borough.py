"""
Generate data/nyc_zip_borough.json from the engineered panel.

Run from the project root:
    uv run python data/build_zip_borough.py

Output: data/nyc_zip_borough.json
  {
    "zip_borough": {"10001": "Manhattan", ...},
    "valid_zips":  ["10001", "10002", ...]
  }

99999 is excluded (sentinel for unknown ZIP in the raw dataset).
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent


def _assign_borough(z: int) -> str:
    if 10000 <= z <= 10282:
        return "Manhattan"
    if 10300 <= z <= 10314:
        return "Staten Island"
    if 10450 <= z <= 10475:
        return "Bronx"
    if 11200 <= z <= 11239:
        return "Brooklyn"
    if z == 11004 or (11100 <= z <= 11109) or (11350 <= z <= 11385) \
            or (11410 <= z <= 11436) or (11690 <= z <= 11697):
        return "Queens"
    return "Unknown"


def main() -> None:
    df = pd.read_csv(DATA_DIR / "data_with_date.csv", usecols=["zip_code"])
    zips = sorted(int(z) for z in df["zip_code"].unique() if int(z) != 99999)

    zip_borough = {str(z): _assign_borough(z) for z in zips}
    valid_zips = [str(z) for z in zips]

    out = {"zip_borough": zip_borough, "valid_zips": valid_zips}
    out_path = DATA_DIR / "nyc_zip_borough.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(valid_zips)} ZIPs to {out_path}")

    for borough in ["Manhattan", "Staten Island", "Bronx", "Brooklyn", "Queens"]:
        count = sum(1 for b in zip_borough.values() if b == borough)
        print(f"  {borough}: {count} ZIPs")


if __name__ == "__main__":
    main()
