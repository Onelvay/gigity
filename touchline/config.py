from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
DB = DATA / "football.sqlite"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
SOURCE = "https://github.com/hudl/open-data"
SOURCE_COMMIT = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
COMPETITION = 2
SEASON = 27


def directories():
    for path in (DATA, RAW, ARTIFACTS, REPORTS):
        path.mkdir(parents=True, exist_ok=True)
