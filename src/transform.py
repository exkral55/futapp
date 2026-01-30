from __future__ import annotations

import hashlib
import re
from typing import Optional

import pandas as pd


# ----------------------------
# Small utils
# ----------------------------
def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    # Turkish chars -> latin
    for tr, en in [("ı", "i"), ("ğ", "g"), ("ş", "s"), ("ö", "o"), ("ü", "u"), ("ç", "c")]:
        s = s.replace(tr, en)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = s.strip().replace(" ", "_")
    return s


def _stable_id(prefix: str, key: str) -> str:
    h = hashlib.md5((key or "").encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        c = lower_map.get(cand.lower())
        if c is not None:
            return c
    return None


def _as_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def _as_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)


def _dedup_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate column labels like 'season' and 'season.1' issues."""
    if df is None or df.empty:
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


# ----------------------------
# FBref -> Teams
# ----------------------------
def build_teams_from_fbref(df_team_stats: pd.DataFrame) -> pd.DataFrame:
    if df_team_stats is None or df_team_stats.empty:
        return pd.DataFrame(columns=["id", "name", "country"])

    squad_col = _pick_col(df_team_stats, ["squad", "team", "squad_name"])
    if not squad_col:
        return pd.DataFrame(columns=["id", "name", "country"])

    teams = (
        df_team_stats[[squad_col]]
        .drop_duplicates()
        .rename(columns={squad_col: "name"})
        .reset_index(drop=True)
    )
    teams["id"] = teams["name"].apply(lambda x: _stable_id("TEAM", _slug(str(x))))
    teams["country"] = ""
    return teams[["id", "name", "country"]]


# ----------------------------
# FBref -> Players
# ----------------------------
def build_players_from_fbref(df_player_stats: pd.DataFrame) -> pd.DataFrame:
    if df_player_stats is None or df_player_stats.empty:
        return pd.DataFrame(columns=["id", "name", "birth_date", "nationality", "position"])

    player_col = _pick_col(df_player_stats, ["player", "player_name", "name"])
    if not player_col:
        return pd.DataFrame(columns=["id", "name", "birth_date", "nationality", "position"])

    pos_col = _pick_col(df_player_stats, ["position", "pos"])

    base = df_player_stats[[player_col] + ([pos_col] if pos_col else [])].copy()
    base = base.drop_duplicates(subset=[player_col]).rename(columns={player_col: "name"})

    base["id"] = base["name"].apply(lambda x: _stable_id("PLAYER", _slug(str(x))))
    base["birth_date"] = ""
    base["nationality"] = ""
    base["position"] = base[pos_col].astype(str) if pos_col else ""

    return base[["id", "name", "birth_date", "nationality", "position"]].reset_index(drop=True)


# ----------------------------
# Understat fallback -> Teams/Players
# ----------------------------
def build_teams_from_understat(df_us: pd.DataFrame) -> pd.DataFrame:
    if df_us is None or df_us.empty:
        return pd.DataFrame(columns=["id", "name", "country"])

    df_us = _dedup_cols(df_us)

    if "team" not in df_us.columns:
        return pd.DataFrame(columns=["id", "name", "country"])

    teams = df_us[["team"]].drop_duplicates().rename(columns={"team": "name"}).reset_index(drop=True)
    teams["id"] = teams["name"].apply(lambda x: _stable_id("TEAM", _slug(str(x))))
    teams["country"] = ""
    return teams[["id", "name", "country"]]


def build_players_from_understat(df_us: pd.DataFrame) -> pd.DataFrame:
    if df_us is None or df_us.empty:
        return pd.DataFrame(columns=["id", "name", "birth_date", "nationality", "position"])

    df_us = _dedup_cols(df_us)

    if "player" not in df_us.columns:
        return pd.DataFrame(columns=["id", "name", "birth_date", "nationality", "position"])

    cols = ["player"] + (["position"] if "position" in df_us.columns else [])
    tmp = df_us[cols].drop_duplicates(subset=["player"]).copy()

    tmp = tmp.rename(columns={"player": "name"})
    tmp["id"] = tmp["name"].apply(lambda x: _stable_id("PLAYER", _slug(str(x))))
    tmp["birth_date"] = ""
    tmp["nationality"] = ""
    if "position" not in tmp.columns:
        tmp["position"] = ""

    return tmp[["id", "name", "birth_date", "nationality", "position"]].reset_index(drop=True)


# ----------------------------
# FBref -> Team Season
# ----------------------------
def build_team_season_from_fbref(
    df_team_stats: pd.DataFrame,
    teams_df: pd.DataFrame,
    seasons_df: pd.DataFrame,
    leagues_df: pd.DataFrame,
) -> pd.DataFrame:
    out_cols = ["team_id", "season_id", "points", "rank"]
    if df_team_stats is None or df_team_stats.empty or teams_df is None or teams_df.empty:
        return pd.DataFrame(columns=out_cols)

    squad_col = _pick_col(df_team_stats, ["squad", "team", "squad_name"])
    pts_col = _pick_col(df_team_stats, ["points", "pts"])
    rk_col = _pick_col(df_team_stats, ["rank", "rnk", "position"])

    if not squad_col:
        return pd.DataFrame(columns=out_cols)

    tmap = dict(zip(teams_df["name"], teams_df["id"]))

    df = df_team_stats.copy()
    df["team_id"] = df[squad_col].map(tmap)

    df["points"] = _as_int_series(df[pts_col]) if pts_col else 0
    df["rank"] = _as_int_series(df[rk_col]) if rk_col else 0

    season_col = _pick_col(df, ["season", "season_id", "year"])
    if season_col:
        def _to_year(x):
            s = str(x)
            m = re.match(r"(\d{4})", s)
            return int(m.group(1)) if m else None

        df["_season_year"] = df[season_col].apply(_to_year)
    else:
        df["_season_year"] = None

    df["season_id"] = df["_season_year"].apply(lambda y: str(y) if y else "")

    out = df[["team_id", "season_id", "points", "rank"]].dropna(subset=["team_id"]).copy()
    return out[out_cols].reset_index(drop=True)


# ----------------------------
# Player Season Stats (FBref + Understat)
# ----------------------------
def build_player_season_stats(
    df_fb: pd.DataFrame,
    df_us: pd.DataFrame,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    out_cols = ["player_id", "team_id", "season_id", "minutes", "goals", "xg", "assists"]

    if (df_fb is None or df_fb.empty) and (df_us is None or df_us.empty):
        return pd.DataFrame(columns=out_cols)

    if players_df is None or players_df.empty:
        return pd.DataFrame(columns=out_cols)

    teams_map = {} if (teams_df is None or teams_df.empty) else dict(zip(teams_df["name"], teams_df["id"]))
    players_map = dict(zip(players_df["name"], players_df["id"]))

    rows = []

    # --- FBref (if exists)
    if df_fb is not None and not df_fb.empty:
        pcol = _pick_col(df_fb, ["player", "player_name", "name"])
        tcol = _pick_col(df_fb, ["squad", "team"])
        min_col = _pick_col(df_fb, ["minutes", "min"])
        g_col = _pick_col(df_fb, ["goals", "gls"])
        a_col = _pick_col(df_fb, ["assists", "ast"])
        season_col = _pick_col(df_fb, ["season", "season_id", "year"])

        tmp = df_fb.copy()
        tmp["_pname"] = tmp[pcol].astype(str) if pcol else ""
        tmp["_tname"] = tmp[tcol].astype(str) if tcol else ""

        def _to_year(x):
            s = str(x)
            m = re.match(r"(\d{4})", s)
            return int(m.group(1)) if m else None

        tmp["_season_year"] = tmp[season_col].apply(_to_year) if season_col else None
        tmp["_player_id"] = tmp["_pname"].map(players_map)
        tmp["_team_id"] = tmp["_tname"].map(teams_map) if teams_map else ""

        mins = _as_int_series(tmp[min_col]) if min_col else 0
        goals = _as_int_series(tmp[g_col]) if g_col else 0
        ast = _as_int_series(tmp[a_col]) if a_col else 0

        for i in range(len(tmp)):
            pid = tmp["_player_id"].iloc[i]
            if pd.isna(pid) or pid is None:
                continue
            rows.append(
                {
                    "player_id": pid,
                    "team_id": tmp["_team_id"].iloc[i] if isinstance(tmp["_team_id"].iloc[i], str) else (tmp["_team_id"].iloc[i] or ""),
                    "season_id": str(tmp["_season_year"].iloc[i]) if tmp["_season_year"].iloc[i] else "",
                    "minutes": int(mins.iloc[i]) if hasattr(mins, "iloc") else int(mins),
                    "goals": int(goals.iloc[i]) if hasattr(goals, "iloc") else int(goals),
                    "assists": int(ast.iloc[i]) if hasattr(ast, "iloc") else int(ast),
                    "xg": 0.0,
                }
            )

    # --- Understat (xg + team + season)
    if df_us is not None and not df_us.empty:
        df_us = _dedup_cols(df_us)

        pcol = "player" if "player" in df_us.columns else _pick_col(df_us, ["player_name", "name"])
        tcol = "team" if "team" in df_us.columns else _pick_col(df_us, ["squad", "team"])
        xg_col = "xg" if "xg" in df_us.columns else _pick_col(df_us, ["xg_per90", "expected_goals"])
        min_col = "minutes" if "minutes" in df_us.columns else _pick_col(df_us, ["min"])
        g_col = "goals" if "goals" in df_us.columns else _pick_col(df_us, ["gls"])
        a_col = "assists" if "assists" in df_us.columns else _pick_col(df_us, ["ast"])

        # IMPORTANT: force season column preference
        if "season" in df_us.columns:
            season_col = "season"
        else:
            season_col = _pick_col(df_us, ["season_id", "year"])

        tmp = df_us.copy()
        tmp = _dedup_cols(tmp)

        tmp["_pname"] = tmp[pcol].astype(str) if pcol else ""
        tmp["_tname"] = tmp[tcol].astype(str) if tcol else ""

        def _to_year(x):
            s = str(x)
            m = re.match(r"(\d{4})", s)
            return int(m.group(1)) if m else None

        tmp["_season_year"] = tmp[season_col].apply(_to_year) if season_col else None

        tmp["_player_id"] = tmp["_pname"].map(players_map)
        tmp["_team_id"] = tmp["_tname"].map(teams_map) if teams_map else ""

        xg = _as_float_series(tmp[xg_col]) if xg_col else 0.0
        mins = _as_int_series(tmp[min_col]) if min_col else 0
        goals = _as_int_series(tmp[g_col]) if g_col else 0
        ast = _as_int_series(tmp[a_col]) if a_col else 0

        for i in range(len(tmp)):
            pid = tmp["_player_id"].iloc[i]
            if pd.isna(pid) or pid is None:
                continue
            rows.append(
                {
                    "player_id": pid,
                    "team_id": tmp["_team_id"].iloc[i] if isinstance(tmp["_team_id"].iloc[i], str) else (tmp["_team_id"].iloc[i] or ""),
                    "season_id": str(tmp["_season_year"].iloc[i]) if tmp["_season_year"].iloc[i] else "",
                    "minutes": int(mins.iloc[i]) if hasattr(mins, "iloc") else int(mins),
                    "goals": int(goals.iloc[i]) if hasattr(goals, "iloc") else int(goals),
                    "assists": int(ast.iloc[i]) if hasattr(ast, "iloc") else int(ast),
                    "xg": float(xg.iloc[i]) if hasattr(xg, "iloc") else float(xg),
                }
            )

    if not rows:
        return pd.DataFrame(columns=out_cols)

    out = pd.DataFrame(rows)

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