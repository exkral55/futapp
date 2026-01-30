# src/transform.py
from __future__ import annotations

import re
import hashlib
from typing import List, Optional

import pandas as pd


# ----------------------------
# Small helpers
# ----------------------------
def _dedup_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate columns created by pandas (.1 etc.) keeping the first occurrence."""
    if df is None or df.empty:
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Pick the first column that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _as_int_series(s) -> pd.Series:
    try:
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    except Exception:
        return pd.Series([0] * len(s))


def _as_float_series(s) -> pd.Series:
    try:
        return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    except Exception:
        return pd.Series([0.0] * len(s))


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = s.strip().replace(" ", "_")
    return s or "unknown"


def _stable_id(prefix: str, name: str) -> str:
    """
    Stable ID from name (so reruns don't create new ids).
    Uses sha1 hash of normalized name to avoid collisions.
    """
    base = _slug(name)
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{base}_{h}"


# ----------------------------
# TEAMS
# ----------------------------
def build_teams_from_fbref(df_team_stats: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["id", "name", "country"]
    if df_team_stats is None or df_team_stats.empty:
        return pd.DataFrame(columns=out_cols)

    df_team_stats = _dedup_cols(df_team_stats)
    tcol = _pick_col(df_team_stats, ["squad", "team", "club", "name"])
    if not tcol:
        return pd.DataFrame(columns=out_cols)

    teams = (
        df_team_stats[[tcol]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .rename(columns={tcol: "name"})
    )
    teams["id"] = teams["name"].apply(lambda x: _stable_id("team", x))
    teams["country"] = ""  # FBref'ten ülke her zaman gelmeyebilir; şimdilik boş
    return teams[["id", "name", "country"]].reset_index(drop=True)


def build_teams_from_understat(df_us: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["id", "name", "country"]
    if df_us is None or df_us.empty:
        return pd.DataFrame(columns=out_cols)

    df_us = _dedup_cols(df_us)
    if "team" not in df_us.columns:
        tcol = _pick_col(df_us, ["squad", "team_name", "club"])
        if not tcol:
            return pd.DataFrame(columns=out_cols)
        tmp = df_us[[tcol]].rename(columns={tcol: "name"})
    else:
        tmp = df_us[["team"]].rename(columns={"team": "name"})

    tmp = tmp.dropna().astype(str).drop_duplicates()
    tmp["id"] = tmp["name"].apply(lambda x: _stable_id("team", x))
    tmp["country"] = ""
    return tmp[["id", "name", "country"]].reset_index(drop=True)


# ----------------------------
# PLAYERS
# ----------------------------
def build_players_from_fbref(df_fb: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["id", "name", "birth_date", "nationality", "position"]
    if df_fb is None or df_fb.empty:
        return pd.DataFrame(columns=out_cols)

    df_fb = _dedup_cols(df_fb)
    pcol = _pick_col(df_fb, ["player", "player_name", "name"])
    if not pcol:
        return pd.DataFrame(columns=out_cols)

    players = df_fb[[pcol]].dropna().astype(str).drop_duplicates().rename(columns={pcol: "name"})
    players["id"] = players["name"].apply(lambda x: _stable_id("player", x))
    players["birth_date"] = ""
    players["nationality"] = ""
    players["position"] = ""
    return players[out_cols].reset_index(drop=True)


def build_players_from_understat(df_us: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["id", "name", "birth_date", "nationality", "position"]
    if df_us is None or df_us.empty:
        return pd.DataFrame(columns=out_cols)

    df_us = _dedup_cols(df_us)

    if "player" in df_us.columns:
        pcol = "player"
    else:
        pcol = _pick_col(df_us, ["player_name", "name"])
        if not pcol:
            return pd.DataFrame(columns=out_cols)

    players = df_us[[pcol]].dropna().astype(str).drop_duplicates().rename(columns={pcol: "name"})
    players["id"] = players["name"].apply(lambda x: _stable_id("player", x))
    players["birth_date"] = ""
    players["nationality"] = ""
    players["position"] = df_us["position"].astype(str) if "position" in df_us.columns else ""
    # position tek değer olmayabilir; basitçe boş bırakalım:
    players["position"] = ""
    return players[out_cols].reset_index(drop=True)


# ----------------------------
# TEAM_SEASON (FBref yoksa boş)
# ----------------------------
def build_team_season_from_fbref(
    df_team_stats: pd.DataFrame,
    teams_df: pd.DataFrame,
    seasons_df: pd.DataFrame,
    leagues_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    FBref 403 yüzünden genelde boş kalacak.
    Şimdilik yanlış veri üretmemek için boş döndürüyoruz.
    """
    return pd.DataFrame(columns=["team_id", "season_id", "points", "rank"])


# ----------------------------
# PLAYER_SEASON_STATS (ASIL KRİTİK)
# ----------------------------
def build_player_season_stats(
    df_fb: pd.DataFrame,
    df_us: pd.DataFrame,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    KURAL:
    - Understat tarafında main.py artık canonical season_id üretiyor:
        <league_id>__YYYY_YYYY
      Bu yüzden burada 'season' parse ETMİYORUZ.
      Varsa df_us['season_id'] direkt alınır.
    - FBref gelirse (ileride) basit fallback vardır.
    """
    out_cols = ["player_id", "team_id", "season_id", "minutes", "goals", "xg", "assists"]

    if (df_fb is None or df_fb.empty) and (df_us is None or df_us.empty):
        return pd.DataFrame(columns=out_cols)

    if players_df is None or players_df.empty:
        return pd.DataFrame(columns=out_cols)

    teams_map = {} if (teams_df is None or teams_df.empty) else dict(zip(teams_df["name"], teams_df["id"]))
    players_map = dict(zip(players_df["name"], players_df["id"]))

    rows = []

    # ----------------------------
    # FBref (varsa) - basit fallback
    # ----------------------------
    if df_fb is not None and not df_fb.empty:
        df_fb = _dedup_cols(df_fb)

        pcol = _pick_col(df_fb, ["player", "player_name", "name"])
        tcol = _pick_col(df_fb, ["squad", "team"])
        min_col = _pick_col(df_fb, ["minutes", "min"])
        g_col = _pick_col(df_fb, ["goals", "gls"])
        a_col = _pick_col(df_fb, ["assists", "ast"])

        # season_id varsa onu kullan, yoksa season/year'dan kaba çıkarım
        if "season_id" in df_fb.columns:
            season_col = "season_id"
        else:
            season_col = _pick_col(df_fb, ["season", "year"])

        tmp = df_fb.copy()
        tmp["_pname"] = tmp[pcol].astype(str) if pcol else ""
        tmp["_tname"] = tmp[tcol].astype(str) if tcol else ""
        tmp["_player_id"] = tmp["_pname"].map(players_map)
        tmp["_team_id"] = tmp["_tname"].map(teams_map) if teams_map else ""

        if season_col:
            tmp["_season_id"] = tmp[season_col].astype(str)
        else:
            tmp["_season_id"] = ""

        mins = _as_int_series(tmp[min_col]) if min_col else pd.Series([0] * len(tmp))
        goals = _as_int_series(tmp[g_col]) if g_col else pd.Series([0] * len(tmp))
        ast = _as_int_series(tmp[a_col]) if a_col else pd.Series([0] * len(tmp))

        for i in range(len(tmp)):
            pid = tmp["_player_id"].iloc[i]
            if pd.isna(pid) or pid is None:
                continue
            rows.append(
                {
                    "player_id": pid,
                    "team_id": tmp["_team_id"].iloc[i] if isinstance(tmp["_team_id"].iloc[i], str) else (tmp["_team_id"].iloc[i] or ""),
                    "season_id": tmp["_season_id"].iloc[i] or "",
                    "minutes": int(mins.iloc[i]),
                    "goals": int(goals.iloc[i]),
                    "assists": int(ast.iloc[i]),
                    "xg": 0.0,
                }
            )

    # ----------------------------
    # Understat (xg + team + canonical season_id)
    # ----------------------------
    if df_us is not None and not df_us.empty:
        df_us = _dedup_cols(df_us)

        pcol = "player" if "player" in df_us.columns else _pick_col(df_us, ["player_name", "name"])
        tcol = "team" if "team" in df_us.columns else _pick_col(df_us, ["squad", "team_name", "club"])
        xg_col = "xg" if "xg" in df_us.columns else _pick_col(df_us, ["xg_per90", "expected_goals"])
        min_col = "minutes" if "minutes" in df_us.columns else _pick_col(df_us, ["min"])
        g_col = "goals" if "goals" in df_us.columns else _pick_col(df_us, ["gls"])
        a_col = "assists" if "assists" in df_us.columns else _pick_col(df_us, ["ast"])

        # >>> KRİTİK: canonical season_id varsa DIRECT kullan <<<
        if "season_id" in df_us.columns:
            season_col = "season_id"
        else:
            # fallback (eski davranış): ham season
            season_col = "season" if "season" in df_us.columns else _pick_col(df_us, ["year"])

        tmp = _dedup_cols(df_us.copy())
        tmp["_pname"] = tmp[pcol].astype(str) if pcol else ""
        tmp["_tname"] = tmp[tcol].astype(str) if tcol else ""

        tmp["_player_id"] = tmp["_pname"].map(players_map)
        tmp["_team_id"] = tmp["_tname"].map(teams_map) if teams_map else ""

        tmp["_season_id"] = tmp[season_col].astype(str) if season_col else ""

        xg = _as_float_series(tmp[xg_col]) if xg_col else pd.Series([0.0] * len(tmp))
        mins = _as_int_series(tmp[min_col]) if min_col else pd.Series([0] * len(tmp))
        goals = _as_int_series(tmp[g_col]) if g_col else pd.Series([0] * len(tmp))
        ast = _as_int_series(tmp[a_col]) if a_col else pd.Series([0] * len(tmp))

        for i in range(len(tmp)):
            pid = tmp["_player_id"].iloc[i]
            if pd.isna(pid) or pid is None:
                continue
            rows.append(
                {
                    "player_id": pid,
                    "team_id": tmp["_team_id"].iloc[i] if isinstance(tmp["_team_id"].iloc[i], str) else (tmp["_team_id"].iloc[i] or ""),
                    "season_id": tmp["_season_id"].iloc[i] or "",
                    "minutes": int(mins.iloc[i]),
                    "goals": int(goals.iloc[i]),
                    "assists": int(ast.iloc[i]),
                    "xg": float(xg.iloc[i]),
                }
            )

    if not rows:
        return pd.DataFrame(columns=out_cols)

    out = pd.DataFrame(rows)

    # groupby ile birleştir
    out = (
        out.groupby(["player_id", "team_id", "season_id"], as_index=False)
        .agg(
            minutes=("minutes", "max"),
            goals=("goals", "max"),
            assists=("assists", "max"),
            xg=("xg", "max"),
        )
    )

    return out[out_cols].reset_index(drop=True)