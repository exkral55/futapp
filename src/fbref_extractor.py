from soccerdata import FBref
import pandas as pd


def extract_fbref_team_season_stats(leagues, seasons):
    fb = FBref()
    all_tables = []

    for lg in leagues:
        for season in seasons:
            try:
                df = fb.read_team_season_stats(
                    league=lg,
                    season=season
                )
                df["source"] = "fbref"
                df["league_code"] = lg
                df["season_year"] = season
                all_tables.append(df)
                print(f"[OK] FBref TEAM stats {lg} {season}")
            except Exception as e:
                print(f"[ERR] FBref TEAM stats {lg} {season} → {e}")

    if not all_tables:
        return pd.DataFrame()

    return pd.concat(all_tables, ignore_index=True)


def extract_fbref_player_season_stats(leagues, seasons):
    fb = FBref()
    all_tables = []

    for lg in leagues:
        for season in seasons:
            try:
                df = fb.read_player_season_stats(
                    league=lg,
                    season=season
                )
                df["source"] = "fbref"
                df["league_code"] = lg
                df["season_year"] = season
                all_tables.append(df)
                print(f"[OK] FBref PLAYER stats {lg} {season}")
            except Exception as e:
                print(f"[ERR] FBref PLAYER stats {lg} {season} → {e}")

    if not all_tables:
        return pd.DataFrame()

    return pd.concat(all_tables, ignore_index=True)