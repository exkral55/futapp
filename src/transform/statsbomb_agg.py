# src/transform/statsbomb_agg.py
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def stage_events(events_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(events_dir.glob("*.json")):
        match_id = int(p.stem)
        evs = json.loads(p.read_text(encoding="utf-8"))
        for e in evs:
            rows.append({
                "match_id": match_id,
                "event_id": e.get("id"),
                "type": (e.get("type") or {}).get("name"),
                "team": (e.get("team") or {}).get("name"),
                "player": (e.get("player") or {}).get("name"),
                "player_source_id": (e.get("player") or {}).get("id"),
                "pass_assisted_shot_id": (e.get("pass") or {}).get("assisted_shot_id") if e.get("pass") else None,
                "shot_outcome": (e.get("shot") or {}).get("outcome", {}).get("name") if e.get("shot") else None,
                "shot_statsbomb_xg": (e.get("shot") or {}).get("statsbomb_xg") if e.get("shot") else None,
            })
    return pd.DataFrame(rows)

def aggregate_goals_assists(events_df: pd.DataFrame) -> pd.DataFrame:
    # goals: shot outcome == "Goal"
    shots = events_df[events_df["type"] == "Shot"].copy()
    shots["is_goal"] = shots["shot_outcome"].eq("Goal").astype(int)

    goals = (
        shots.groupby(["player_source_id"], dropna=False)["is_goal"]
        .sum()
        .reset_index()
        .rename(columns={"is_goal": "goals"})
    )

    # assists: passes that have assisted_shot_id which corresponds to a goal shot
    goal_shot_ids = set(shots.loc[shots["is_goal"] == 1, "event_id"].dropna().astype(str).tolist())
    passes = events_df[events_df["type"] == "Pass"].copy()
    passes["is_assist"] = passes["pass_assisted_shot_id"].astype(str).isin(goal_shot_ids).astype(int)

    assists = (
        passes.groupby(["player_source_id"], dropna=False)["is_assist"]
        .sum()
        .reset_index()
        .rename(columns={"is_assist": "assists"})
    )

    out = goals.merge(assists, on="player_source_id", how="outer").fillna(0)
    out["goals"] = out["goals"].astype(int)
    out["assists"] = out["assists"].astype(int)
    return out
