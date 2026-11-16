"""Local derived-cache location; never changes the raw-data boundary."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def cache_path(size):
    locations=ROOT/'reports/cache_locations.json'
    config=json.loads(locations.read_text(encoding='utf-8')) if locations.exists() else {}
    return Path(config.get(str(size),str(ROOT/'data'/f'cache{size}')))
