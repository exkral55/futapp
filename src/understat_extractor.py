import pandas as pd
from soccerdata import Understat


def extract_understat_player_season_stats(leagues, seasons):
    seasons_us = [str(s) for s in seasons]

    us = Understat(leagues=leagues, seasons=seasons_us)
    df = us.read_player_season_stats()

    if df is None or df.empty:
        return pd.DataFrame()

    # --- normalize columns robustly ---
    df = df.reset_index(drop=False)

    # common renames across soccerdata versions
    rename_map = {}
    for col in df.columns:
        c = str(col).lower().strip()
        if c in ["player", "player_name", "name"]:
            rename_map[col] = "player"
        elif c in ["team", "squad"]:
            rename_map[col] = "team"
        elif c in ["season", "year", "season_id"]:
            rename_map[col] = "season"
        elif c in ["min", "minutes", "time"]:
            rename_map[col] = "minutes"
        elif c in ["goals", "gls"]:
            rename_map[col] = "goals"
        elif c in ["assists", "ast"]:
            rename_map[col] = "assists"
        elif c == "xg":
            rename_map[col] = "xg"

    df = df.rename(columns=rename_map)

    # If still no player/team, try alternative columns
    if "player" not in df.columns:
        # soccerdata sometimes uses player_id but has player name in a different col
        for alt in ["player_id", "playerid", "playerId"]:
            if alt in df.columns:
                df["player"] = df[alt].astype(str)
                break

    if "team" not in df.columns:
        for alt in ["team_id", "teamid", "teamId"]:
            if alt in df.columns:
                df["team"] = df[alt].astype(str)
                break

    if "season" not in df.columns:
        # best-effort: if seasons were requested, set unknown season empty
        df["season"] = ""

    # guarantee numeric columns exist
    for col in ["minutes", "goals", "assists", "xg"]:
        if col not in df.columns:
            df[col] = 0

    df["source"] = "understat"
    return df.reset_index(drop=True)