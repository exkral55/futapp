# src/extract/statsbomb.py
from __future__ import annotations
import json
from pathlib import Path
import requests

def _get_json(url: str, timeout: int = 60):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def download_competitions(base_url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"{base_url}/competitions.json"
    data = _get_json(url)
    out_path = out_dir / "competitions.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return out_path

def download_matches(base_url: str, out_dir: Path, competition_id: int, season_id: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"{base_url}/matches/{competition_id}/{season_id}.json"
    data = _get_json(url)
    out_path = out_dir / f"matches_{competition_id}_{season_id}.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return out_path

def download_events_and_lineups(
    base_url: str,
    out_events: Path,
    out_lineups: Path,
    match_ids: list[int],
) -> None:
    out_events.mkdir(parents=True, exist_ok=True)
    out_lineups.mkdir(parents=True, exist_ok=True)

    for mid in match_ids:
        # events
        ev_url = f"{base_url}/events/{mid}.json"
        ev_path = out_events / f"{mid}.json"
        if not ev_path.exists():
            ev = _get_json(ev_url)
            ev_path.write_text(json.dumps(ev, ensure_ascii=False), encoding="utf-8")

        # lineups
        lu_url = f"{base_url}/lineups/{mid}.json"
        lu_path = out_lineups / f"{mid}.json"
        if not lu_path.exists():
            lu = _get_json(lu_url)
            lu_path.write_text(json.dumps(lu, ensure_ascii=False), encoding="utf-8")