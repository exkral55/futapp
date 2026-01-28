from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
import yaml

# Extractors
from src.fbref_extractor import (
    extract_fbref_team_season_stats,
    extract_fbref_player_season_stats,
)
from src.understat_extractor import extract_understat_player_season_stats

# Normalize helper
from src.normalize_utils import standardize_name

# Transformers → DB Tables
from src.transform import (
    build_teams_from_fbref,
    build_team_season_from_fbref,
    build_players_from_fbref,
    build_player_season_stats,
)


# ----------------------------
# CONFIG MODEL
# ----------------------------
@dataclass
class LeagueConfig:
    country_code: str
    country_name: str
    level: int
    league_name: str
    season_format: str
    ids: Dict[str, Any]
    active: bool = True


def load_leagues_config(path: str) -> List[LeagueConfig]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    leagues = []
    for item in cfg.get("leagues", []):
        leagues.append(
            LeagueConfig(
                country_code=item["country_code"],
                country_name=item["country_name"],
                level=int(item.get("level", 1)),
                league_name=item["league_name"],
                season_format=item.get("season_format", "YYYY-YYYY"),
                ids=item.get("ids", {}),
                active=bool(item.get("active", True)),
            )
        )
    return leagues


# ----------------------------
# NORMALIZED EMPTY SCHEMA
# ----------------------------
def normalized_empty_tables_v1() -> Dict[str, pd.DataFrame]:
    return {
        "leagues": pd.DataFrame(columns=["id", "name", "country", "level"]),
        "seasons": pd.DataFrame(columns=["id", "league_id", "season_year"]),
        "teams": pd.DataFrame(columns=["id", "name", "country"]),
        "players": pd.DataFrame(columns=["id", "name", "birth_date", "nationality", "position"]),
        "team_season": pd.DataFrame(columns=["team_id", "season_id", "points", "rank"]),
        "player_season_stats": pd.DataFrame(columns=["player_id", "team_id", "season_id", "minutes", "goals", "xg", "assists"]),
        "matches": pd.DataFrame(columns=["id", "season_id", "home_team_id", "away_team_id", "date", "home_goals", "away_goals"]),
    }


# ----------------------------
# HELPERS
# ----------------------------
def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def write_tables_as_csv(tables, out_dir):
    ensure_dirs(out_dir)
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False, encoding="utf-8-sig")


def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()


def make_league_id(country_code, league_name, level):
    safe = league_name.lower()
    for tr, en in [("ı","i"),("ğ","g"),("ş","s"),("ö","o"),("ü","u"),("ç","c")]:
        safe = safe.replace(tr,en)
    safe = "".join(ch if ch.isalnum() else "_" for ch in safe)
    return f"{country_code}_{level}_{safe}"


def make_season_id(league_id, season_year):
    return f"{league_id}__{season_year}"


# ----------------------------
# BUILD LEAGUE + SEASON TABLES
# ----------------------------
def build_leagues_table(cfg_leagues):
    rows = []
    for l in cfg_leagues:
        rows.append({
            "id": make_league_id(l.country_code, l.league_name, l.level),
            "name": l.league_name,
            "country": l.country_code,
            "level": l.level,
        })
    return pd.DataFrame(rows)


def build_seasons_table(leagues_df, season_years):
    rows = []
    for _, row in leagues_df.iterrows():
        for y in season_years:
            rows.append({
                "id": make_season_id(row["id"], y),
                "league_id": row["id"],
                "season_year": y
            })
    return pd.DataFrame(rows)


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():
    root = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(root, "config", "leagues.yaml")
    out_dir = os.path.join(root, "data", "normalized")

    ensure_dirs(out_dir)

    # 1) Load config
    cfg_all = load_leagues_config(cfg_path)
    cfg_leagues = [l for l in cfg_all if l.active and l.level == 1]
    print(f"[INFO] Active top leagues: {len(cfg_leagues)}")

    # 2) Seasons
    season_years = [2019, 2020, 2021, 2022, 2023]
    print(f"[INFO] Seasons: {season_years}")

    # 3) Init schema
    tables = normalized_empty_tables_v1()

    # 4) Build league + season tables
    leagues_df = build_leagues_table(cfg_leagues)
    seasons_df = build_seasons_table(leagues_df, season_years)
    tables["leagues"] = leagues_df
    tables["seasons"] = seasons_df

    # ----------------------------
    # EXTRACT
    # ----------------------------
    fbref_leagues = [l.ids["fbref"] for l in cfg_leagues if l.ids.get("fbref")]
    understat_leagues = [l.ids["understat"] for l in cfg_leagues if l.ids.get("understat")]

    print("\n[STEP] FBref team stats")
    df_team_stats = extract_fbref_team_season_stats(fbref_leagues, season_years)

    print("\n[STEP] FBref player stats")
    df_player_stats_fb = extract_fbref_player_season_stats(fbref_leagues, season_years)

    print("\n[STEP] Understat player stats (xG)")
    df_player_stats_us = extract_understat_player_season_stats(understat_leagues, season_years)

    # Standardize names for xG merge
    if not df_player_stats_fb.empty:
        df_player_stats_fb["player_std"] = df_player_stats_fb["player"].apply(standardize_name)

    if not df_player_stats_us.empty:
        df_player_stats_us["player_std"] = df_player_stats_us["player"].apply(standardize_name)

    # Save raw debug CSVs
    df_team_stats.to_csv(os.path.join(out_dir, "fbref_team_season_raw.csv"), index=False, encoding="utf-8-sig")
    df_player_stats_fb.to_csv(os.path.join(out_dir, "fbref_player_season_raw.csv"), index=False, encoding="utf-8-sig")
    df_player_stats_us.to_csv(os.path.join(out_dir, "understat_player_season_raw.csv"), index=False, encoding="utf-8-sig")
    print("[OK] Raw extractor CSVs written.")

    # ----------------------------
    # TRANSFORM → DB TABLES
    # ----------------------------
    teams_df = build_teams_from_fbref(df_team_stats)
    players_df = build_players_from_fbref(df_player_stats_fb)
    team_season_df = build_team_season_from_fbref(df_team_stats, teams_df, seasons_df, leagues_df)
    player_season_df = build_player_season_stats(df_player_stats_fb, df_player_stats_us, players_df, teams_df)

    tables["teams"] = teams_df
    tables["players"] = players_df
    tables["team_season"] = team_season_df
    tables["player_season_stats"] = player_season_df

    print(f"[OK] teams: {len(teams_df)} | players: {len(players_df)} | player_season_stats: {len(player_season_df)}")

    # ----------------------------
    # WRITE NORMALIZED CSVs
    # ----------------------------
    write_tables_as_csv(tables, out_dir)
    print("[DONE] Normalized DB schema created!")
    print(f"📁 Output folder: {out_dir}")


if __name__ == "__main__":
    main()