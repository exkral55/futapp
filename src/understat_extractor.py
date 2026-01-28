from soccerdata import Understat
import pandas as pd


def extract_understat_player_season_stats(leagues, seasons):
    us = Understat()
    all_tables = []

    for lg in leagues:
        for season in seasons:
            try:
                df = us.read_player_season_stats(
                    league=lg,
                    season=season
                )
                df["source"] = "understat"
                df["league_code"] = lg
                df["season_year"] = season
                all_tables.append(df)
                print(f"[OK] Understat PLAYER stats {lg} {season}")
            except Exception as e:
                print(f"[ERR] Understat PLAYER stats {lg} {season} → {e}")

    if not all_tables:
        return pd.DataFrame()

    return pd.concat(all_tables, ignore_index=True)