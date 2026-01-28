from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
import yaml

# Yol B extractorlar
from src.fbref_extractor import (
    extract_fbref_team_season_stats,
    extract_fbref_player_season_stats,
)
from src.understat_extractor import extract_understat_player_season_stats
from src.normalize_utils import standardize_name


# ----------------------------
# Config models
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

    leagues_raw = cfg.get("leagues", [])
    leagues: List[LeagueConfig] = []
    for item in leagues_raw:
        leagues.append(
            LeagueConfig(
                country_code=item["country_code"],
                country_name=item["country_name"],
                level=int(item.get("level", item.get("tier", 1))),
                league_name=item["league_name"],
                season_format=item.get("season_format", "YYYY-YYYY"),
                ids=item.get("ids", {}),
                active=bool(item.get("active", True)),
            )
        )
    return leagues


# ----------------------------
# Normalized schema (your v1)
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
        "source_entity_map": pd.DataFrame(
            columns=[
                "entity_type",
                "source",
                "source_id",
                "canonical_id",
                "source_name",
                "season_id",
                "confidence",
                "match_method",
                "fetched_at",
            ]
        ),
    }


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def write_tables_as_csv(tables: Dict[str, pd.DataFrame], out_dir: str) -> None:
    ensure_dirs(out_dir)
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False, encoding="utf-8-sig")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_league_id(country_code: str, league_name: str, level: int) -> str:
    safe = (
        league_name.lower()
        .replace("ı", "i").replace("ğ", "g").replace("ş", "s").replace("ö", "o").replace("ü", "u").replace("ç", "c")
    )
    safe = "".join(ch if (ch.isalnum() or ch == " ") else " " for ch in safe)
    safe = "_".join(safe.split())
    return f"{country_code}_{level}_{safe}"


def make_season_id(league_id: str, season_year: int) -> str:
    return f"{league_id}__{season_year}"


def build_leagues_table(cfg_leagues: List[LeagueConfig]) -> pd.DataFrame:
    rows = []
    for l in cfg_leagues:
        rows.append(
            {
                "id": make_league_id(l.country_code, l.league_name, l.level),
                "name": l.league_name,
                "country": l.country_code,
                "level": l.level,
            }
        )
    return pd.DataFrame(rows)


def build_seasons_table(leagues_df: pd.DataFrame, season_years: List[int]) -> pd.DataFrame:
    rows = []
    for _, row in leagues_df.iterrows():
        league_id = row["id"]
        for y in season_years:
            rows.append({"id": make_season_id(league_id, int(y)), "league_id": league_id, "season_year": int(y)})
    return pd.DataFrame(rows)


def build_source_entity_map_for_leagues(cfg_leagues: List[LeagueConfig], fetched_at: str) -> pd.DataFrame:
    rows = []
    for l in cfg_leagues:
        canonical_league_id = make_league_id(l.country_code, l.league_name, l.level)
        for source in ["fbref", "understat"]:
            source_id = (l.ids or {}).get(source)
            if source_id:
                rows.append(
                    {
                        "entity_type": "league",
                        "source": source,
                        "source_id": str(source_id),
                        "canonical_id": canonical_league_id,
                        "source_name": l.league_name,
                        "season_id": "",
                        "confidence": 1.0,
                        "match_method": "config",
                        "fetched_at": fetched_at,
                    }
                )
    return pd.DataFrame(rows)


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(project_root, "config", "leagues.yaml")
    out_dir = os.path.join(project_root, "data", "normalized")

    ensure_dirs(out_dir)

    # 1) Load config
    cfg_leagues_all = load_leagues_config(cfg_path)
    cfg_leagues = [l for l in cfg_leagues_all if l.active and l.level == 1]
    print(f"[INFO] Active level=1 leagues in config: {len(cfg_leagues)}")

    # 2) Seasons (ilk etap)
    season_years = [2019, 2020, 2021, 2022, 2023]
    print(f"[INFO] Season years: {season_years}")

    # 3) Init normalized schema (empty)
    tables = normalized_empty_tables_v1()

    # 4) Build catalog tables
    leagues_df = build_leagues_table(cfg_leagues)
    seasons_df = build_seasons_table(leagues_df, season_years)
    tables["leagues"] = leagues_df
    tables["seasons"] = seasons_df

    # 5) Initial mapping for leagues
    fetched_at = now_utc_iso()
    tables["source_entity_map"] = build_source_entity_map_for_leagues(cfg_leagues, fetched_at)

    # =====================================================
    # ✅ EXTRACTOR BLOĞU (CSV yazmadan önce)
    # =====================================================

    fbref_leagues = [l.ids["fbref"] for l in cfg_leagues if l.ids.get("fbref")]
    understat_leagues = [l.ids["understat"] for l in cfg_leagues if l.ids.get("understat")]

    print("\n[STEP] FBref team season stats")
    df_team_stats = extract_fbref_team_season_stats(fbref_leagues, season_years)

    print("\n[STEP] FBref player season stats")
    df_player_stats_fb = extract_fbref_player_season_stats(fbref_leagues, season_years)

    print("\n[STEP] Understat player season stats (xG)")
    df_player_stats_us = extract_understat_player_season_stats(understat_leagues, season_years)

    # quick normalize "player_std" columns (inspection)
    if not df_player_stats_fb.empty:
        name_col = "player" if "player" in df_player_stats_fb.columns else "player_name"
        df_player_stats_fb["player_std"] = df_player_stats_fb[name_col].apply(standardize_name)

    if not df_player_stats_us.empty and "player" in df_player_stats_us.columns:
        df_player_stats_us["player_std"] = df_player_stats_us["player"].apply(standardize_name)

    # write raw extracts for debugging (these are extra files)
    df_team_stats.to_csv(os.path.join(out_dir, "fbref_team_season_raw.csv"), index=False, encoding="utf-8-sig")
    df_player_stats_fb.to_csv(os.path.join(out_dir, "fbref_player_season_raw.csv"), index=False, encoding="utf-8-sig")
    df_player_stats_us.to_csv(os.path.join(out_dir, "understat_player_season_raw.csv"), index=False, encoding="utf-8-sig")

    print("[OK] Raw extractor CSVs written.")

    # =====================================================
    # 6) Write normalized schema CSVs (keep this last)
    # =====================================================
    write_tables_as_csv(tables, out_dir)

    print("[DONE] Normalized schema v1 created.")
    print(f"       Output: {out_dir}")


if __name__ == "__main__":
    main()