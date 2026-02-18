# src/transform/statsbomb_stage.py
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def load_matches(matches_json_path: Path) -> pd.DataFrame:
    matches = json.loads(matches_json_path.read_text(encoding="utf-8"))
    return pd.DataFrame(matches)

def stage_lineups(lineups_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(lineups_dir.glob("*.json")):
        match_id = int(p.stem)
        data = json.loads(p.read_text(encoding="utf-8"))
        # data: list of teams, each has 'lineup'
        for team in data:
            team_id = team.get("team_id")
            team_name = team.get("team_name")
            for pl in team.get("lineup", []):
                rows.append({
                    "match_id": match_id,
                    "team_source_id": team_id,
                    "team_name": team_name,
                    "player_source_id": pl.get("player_id"),
                    "player_name": pl.get("player_name"),
                    # position info may be inside positions list
                    "positions": pl.get("positions", []),
                })
    df = pd.DataFrame(rows)
    return df

def explode_positions(player_match_df: pd.DataFrame) -> pd.DataFrame:
    # positions list contains dicts with position, from, to, minutes
    out = []
    for _, r in player_match_df.iterrows():
        positions = r["positions"] or []
        if not positions:
            out.append({**r, "position": None, "minutes": None})
            continue
        for pos in positions:
            out.append({
                "match_id": r["match_id"],
                "team_source_id": r["team_source_id"],
                "team_name": r["team_name"],
                "player_source_id": r["player_source_id"],
                "player_name": r["player_name"],
                "position": (pos.get("position") or {}).get("name"),
                "minutes": pos.get("minutes"),
                "from": pos.get("from"),
                "to": pos.get("to"),
            })
    return pd.DataFrame(out)