from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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

# Transformers
from src.transform import (
    build_teams_from_fbref,
    build_team_season_from_fbref,
    build_players_from_fbref,
    build_player_season_stats,
    # Fallbacks (must exist in src/transform.py)
    build_teams_from_understat,
    build_players_from_understat,
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

    leagues: List[LeagueConfig] = []
    for item in cfg.get("leagues", []):
        leagues.append(
            LeagueConfig(
                country_code=item.get("country_code", ""),
                country_name=item.get("country_name", ""),
                level=int(item.get("level", 1)),
                league_name=item.get("league_name", ""),
                season_format=item.get("season_format", "YYYY-YYYY"),
                ids=item.get("ids", {}) or {},
                active=bool(item.get("active", True)),
            )
        )
    return leagues


# ----------------------------
# HELPERS
# ----------------------------
def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_to_csv(df: pd.DataFrame | None, path: str) -> None:
    """
    Avoid 0-byte CSVs. If df empty, writes header (if columns exist).
    """
    try:
        if df is None:
            pd.DataFrame().to_csv(path, index=False, encoding="utf-8-sig")
            return
        if isinstance(df, pd.DataFrame) and df.empty:
            df.head(0).to_csv(path, index=False, encoding="utf-8-sig")
            return
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception:
        # last-resort: write truly empty file with no crash
        pd.DataFrame().to_csv(path, index=False, encoding="utf-8-sig")


def write_tables_as_csv(tables: Dict[str, pd.DataFrame], out_dir: str) -> None:
    ensure_dirs(out_dir)
    for name, df in tables.items():
        safe_to_csv(df, os.path.join(out_dir, f"{name}.csv"))


def make_league_id(country_code: str, league_name: str, level: int) -> str:
    safe = (league_name or "").lower()
    for tr, en in [("ı", "i"), ("ğ", "g"), ("ş", "s"), ("ö", "o"), ("ü", "u"), ("ç", "c")]:
        safe = safe.replace(tr, en)
    safe = "".join(ch if (ch.isalnum() or ch == " ") else " " for ch in safe)
    safe = "_".join(safe.split())
    return f"{country_code}_{level}_{safe}"


def make_season_id(league_id: str, start_year: int) -> str:
    return f"{league_id}__{start_year}_{start_year + 1}"

def make_season_label(start_year: int) -> str:
    
    return f"{start_year}/{start_year + 1}"
def parse_understat_season_start_year(val: Any) -> Optional[int]:
    """
    Understat season örnekleri:
      '1920' -> 2019
      '2223' -> 2022
      '2324' -> 2023
    """
    if val is None:
        return None
    s = str(val).strip()

    # '1920' gibi formatlar
    if len(s) == 4 and s.isdigit():
        yy = int(s[:2])
        base = 2000 if yy < 50 else 1900
        return base + yy

    # fallback: içinde 4 haneli yıl varsa onu al
    import re
    m = re.search(r"(\d{4})", s)
    return int(m.group(1)) if m else None


# ----------------------------
# NORMALIZED EMPTY SCHEMA
# ----------------------------
def normalized_empty_tables_v1() -> Dict[str, pd.DataFrame]:
    return {
        "leagues": pd.DataFrame(columns=["id", "name", "country", "level"]),
        "seasons": pd.DataFrame(columns=["id", "league_id", "season_start_year", "season_label"]),
        "teams": pd.DataFrame(columns=["id", "name", "country"]),
        "players": pd.DataFrame(columns=["id", "name", "birth_date", "nationality", "position"]),
        "team_season": pd.DataFrame(columns=["team_id", "season_id", "points", "rank"]),
        "player_season_stats": pd.DataFrame(
            columns=["player_id", "team_id", "season_id", "minutes", "goals", "xg", "assists"]
        ),
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


# ----------------------------
# BUILD LEAGUE + SEASON TABLES
# ----------------------------
def build_leagues_table(cfg_leagues: List[LeagueConfig]) -> pd.DataFrame:
    rows = []
    for l in cfg_leagues:
        rows.append(
            {
                "id": make_league_id(l.country_code, l.league_name, l.level),
                "name": l.league_name,
                "country": l.country_code,
                "level": int(l.level),
            }
        )
    return pd.DataFrame(rows)


def build_source_entity_map_for_leagues(cfg_leagues: List[LeagueConfig], fetched_at: str) -> pd.DataFrame:
    rows = []
    for l in cfg_leagues:
        canonical_league_id = make_league_id(l.country_code, l.league_name, l.level)

        # ids içinden hem fbref hem understat varsa map’e yaz
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
def build_seasons_table(leagues_df: pd.DataFrame, season_years: List[int]) -> pd.DataFrame:
    rows = []
    for _, row in leagues_df.iterrows():
        league_id = row["id"]
        for y in season_years:
            y = int(y)
            rows.append(
                {
                    "id": make_season_id(league_id, y),
                    "league_id": league_id,
                    "season_start_year": y,
                    "season_label": make_season_label(y),
                }
            )

    df = pd.DataFrame(rows)

    # lig bazlı + sezon sıralı
    return df.sort_values(
        ["league_id", "season_start_year"]
    ).reset_index(drop=True)
    


# ----------------------------
# MAIN
# ----------------------------
def main():
    root = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(root, "config", "leagues.yaml")
    out_dir = os.path.join(root, "data", "normalized")
    ensure_dirs(out_dir)

    # 1) Load config
    cfg_all = load_leagues_config(cfg_path)
    cfg_leagues = [l for l in cfg_all if l.active and int(l.level) == 1]
    print(f"[INFO] Active top leagues: {len(cfg_leagues)}")

    # 2) Seasons (ilk etap)
    season_years = [2019, 2020, 2021, 2022, 2023]
    print(f"[INFO] Season years: {season_years}")

    # 3) Init schema
    tables = normalized_empty_tables_v1()

    # 4) Catalog tables
    leagues_df = build_leagues_table(cfg_leagues)
    seasons_df = build_seasons_table(leagues_df, season_years)
    tables["leagues"] = leagues_df
    tables["seasons"] = seasons_df

    fetched_at = now_utc_iso()
    tables["source_entity_map"] = build_source_entity_map_for_leagues(cfg_leagues, fetched_at)

    # ----------------------------
    # EXTRACT (RAW DEBUG OUTPUTS)
    # ----------------------------
    fbref_leagues = [l.ids.get("fbref") for l in cfg_leagues if l.ids.get("fbref")]
    understat_leagues = [l.ids.get("understat") for l in cfg_leagues if l.ids.get("understat")]

    print("[DEBUG] fbref_leagues:", fbref_leagues)
    print("[DEBUG] understat_leagues:", understat_leagues)

    df_team_stats = pd.DataFrame()
    df_player_stats_fb = pd.DataFrame()
    df_player_stats_us = pd.DataFrame()

    # FBref can 403; do not crash pipeline
    if fbref_leagues:
        print("\n[STEP] FBref team season stats")
        try:
            df_team_stats = extract_fbref_team_season_stats(fbref_leagues, season_years)
        except Exception as e:
            print("[WARN] FBref team stats skipped:", repr(e))

        print("\n[STEP] FBref player season stats")
        try:
            df_player_stats_fb = extract_fbref_player_season_stats(fbref_leagues, season_years)
        except Exception as e:
            print("[WARN] FBref player stats skipped:", repr(e))
    else:
        print("[WARN] fbref_leagues empty -> skipping FBref.")

    # Understat (Big5 mostly)
    if understat_leagues:
        print("\n[STEP] Understat player season stats (xG)")
        try:
            df_player_stats_us = extract_understat_player_season_stats(understat_leagues, season_years)
        except Exception as e:
            print("[WARN] Understat skipped:", repr(e))
            
    else:
        print("[WARN] understat_leagues empty -> skipping Understat.")
    
    # ----------------------------
    # UNDERSTAT -> CANONICAL season_id (league_id__YYYY_YYYY)
    # ----------------------------
    understat_to_canonical = {
        str(l.ids.get("understat")): make_league_id(l.country_code, l.league_name, l.level)
        for l in cfg_leagues
        if (l.ids or {}).get("understat")
    }

    if isinstance(df_player_stats_us, pd.DataFrame) and (not df_player_stats_us.empty):
        # league -> canonical league_id
        df_player_stats_us["canonical_league_id"] = df_player_stats_us["league"].astype(str).map(understat_to_canonical)

        # season -> season_start_year
        # --- FIX: Understat DF can have duplicate index labels
        df_player_stats_us = df_player_stats_us.copy()
        df_player_stats_us = df_player_stats_us.loc[:, ~df_player_stats_us.columns.duplicated()].reset_index(drop=True)
        df_player_stats_us["season_start_year"] = df_player_stats_us["season"].apply(parse_understat_season_start_year)

        # season_id + label
        df_player_stats_us["season_id"] = df_player_stats_us.apply(
            lambda r: make_season_id(r["canonical_league_id"], int(r["season_start_year"])),
            axis=1,
        )
        df_player_stats_us["season_label"] = df_player_stats_us["season_start_year"].apply(make_season_label)

        # unmapped ligleri raporla ve çıkar
        unmapped = df_player_stats_us[df_player_stats_us["canonical_league_id"].isna()]
        if not unmapped.empty:
            print("[WARN] Understat unmapped leagues:", unmapped["league"].value_counts().head(20).to_dict())

        df_player_stats_us = df_player_stats_us[df_player_stats_us["canonical_league_id"].notna()].copy()

    print(
        "SHAPES:",
        "team", getattr(df_team_stats, "shape", None),
        "fb_players", getattr(df_player_stats_fb, "shape", None),
        "us_players", getattr(df_player_stats_us, "shape", None),
    )

    # Standardize names (xG merge için)
    if isinstance(df_player_stats_fb, pd.DataFrame) and (not df_player_stats_fb.empty):
        if "player" in df_player_stats_fb.columns:
            df_player_stats_fb["player_std"] = df_player_stats_fb["player"].apply(standardize_name)

    if isinstance(df_player_stats_us, pd.DataFrame) and (not df_player_stats_us.empty):
        if "player" in df_player_stats_us.columns:
            df_player_stats_us["player_std"] = df_player_stats_us["player"].apply(standardize_name)

    # Save raw (safe)
    safe_to_csv(df_team_stats, os.path.join(out_dir, "fbref_team_season_raw.csv"))
    safe_to_csv(df_player_stats_fb, os.path.join(out_dir, "fbref_player_season_raw.csv"))
    safe_to_csv(df_player_stats_us, os.path.join(out_dir, "understat_player_season_raw.csv"))
    print("[OK] Raw extractor CSVs written.")

    # ----------------------------
    # TRANSFORM -> YOUR TABLES
    # ----------------------------
    teams_df = build_teams_from_fbref(df_team_stats)
    if teams_df.empty:
        teams_df = build_teams_from_understat(df_player_stats_us)

    players_df = build_players_from_fbref(df_player_stats_fb)
    if players_df.empty:
        players_df = build_players_from_understat(df_player_stats_us)

    team_season_df = build_team_season_from_fbref(df_team_stats, teams_df, seasons_df, leagues_df)
    player_season_df = build_player_season_stats(df_player_stats_fb, df_player_stats_us, players_df, teams_df)

    tables["teams"] = teams_df
    tables["players"] = players_df
    tables["team_season"] = team_season_df
    tables["player_season_stats"] = player_season_df

    print(
        f"[OK] teams: {len(teams_df)} | players: {len(players_df)} "
        f"| team_season: {len(team_season_df)} | player_season_stats: {len(player_season_df)}"
    )

    # ----------------------------
    # WRITE NORMALIZED CSVs
    # ----------------------------
    write_tables_as_csv(tables, out_dir)
    print("[DONE] Normalized schema v1 created.")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()