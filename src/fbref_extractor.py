import pandas as pd
from soccerdata import FBref


def _to_fbref_season(season_year):
    # 2019 -> "2019-2020"
    y = int(season_year)
    return f"{y}-{y+1}"


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join([str(x) for x in tup if x and str(x) != "nan"]).strip("_")
            for tup in df.columns.to_list()
        ]
    df.columns = [c.strip() for c in df.columns]
    return df


def extract_fbref_team_season_stats(leagues, seasons):
    # seasons -> ["2019-2020", "2020-2021", ...]
    seasons_fb = [_to_fbref_season(s) for s in seasons]

    fb = FBref(leagues=leagues, seasons=seasons_fb)
    df = fb.read_team_season_stats()
    df = _flatten_cols(df)

    if df is None or df.empty:
        return pd.DataFrame()

    df["source"] = "fbref"
    return df.reset_index(drop=True)


def extract_fbref_player_season_stats(leagues, seasons):
    seasons_fb = [_to_fbref_season(s) for s in seasons]

    fb = FBref(leagues=leagues, seasons=seasons_fb)
    df = fb.read_player_season_stats()
    df = _flatten_cols(df)

    if df is None or df.empty:
        return pd.DataFrame()

    df["source"] = "fbref"
    return df.reset_index(drop=True)