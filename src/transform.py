import pandas as pd

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_int(x):
    try:
        if pd.isna(x): 
            return None
        return int(float(x))
    except:
        return None

def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except:
        return None

def build_teams_from_fbref(df_team_stats):
    # FBref team stats genelde "squad" veya "team" kolonuyla gelir
    team_col = pick_col(df_team_stats, ["squad", "team", "Squad"])
    if team_col is None or df_team_stats.empty:
        return pd.DataFrame(columns=["id","name","country"])

    teams = (
        df_team_stats[[team_col]]
        .drop_duplicates()
        .rename(columns={team_col: "name"})
        .reset_index(drop=True)
    )
    teams["id"] = teams["name"].apply(lambda x: "T_" + str(abs(hash(x))) )
    teams["country"] = ""  # şimdilik boş, ileride league->country ile dolduracağız
    return teams[["id","name","country"]]

def build_team_season_from_fbref(df_team_stats, teams_df, seasons_df, leagues_df):
    """
    team_season: team_id, season_id, points, rank
    Burada season_id'yi (league_code + season_year) üzerinden map edeceğiz.
    """
    if df_team_stats.empty:
        return pd.DataFrame(columns=["team_id","season_id","points","rank"])

    team_col = pick_col(df_team_stats, ["squad", "team", "Squad"])
    pts_col  = pick_col(df_team_stats, ["pts", "points", "Pts"])
    rk_col   = pick_col(df_team_stats, ["rk", "rank", "Rk"])
    season_col = pick_col(df_team_stats, ["season_year", "Season"])
    league_col = pick_col(df_team_stats, ["league_code", "Comp"])

    # team_id map
    team_id_map = dict(zip(teams_df["name"], teams_df["id"]))

    out = pd.DataFrame()
    out["team_id"] = df_team_stats[team_col].map(team_id_map)

    # season_id map: leagues_df + season_year -> seasons_df.id
    # leagues_df.id bizim canonical; ama df'de league_code (fbref code) var.
    # Şimdilik basit: seasons_df'de season_year var; league match'i sonraki adımda netleştiririz.
    # Bu adımda sadece season_year ile eşleştirip tek ligmiş gibi davranmayalım:
    # sezon_id yoksa boş bırakırız, sonra düzeltiriz.
    out["season_id"] = df_team_stats[season_col].apply(lambda y: "") if season_col else ""

    out["points"] = df_team_stats[pts_col].apply(safe_int) if pts_col else None
    out["rank"]   = df_team_stats[rk_col].apply(safe_int) if rk_col else None

    # boş team_id'leri at
    out = out.dropna(subset=["team_id"]).reset_index(drop=True)
    return out[["team_id","season_id","points","rank"]]

def build_players_from_fbref(df_player_stats):
    if df_player_stats.empty:
        return pd.DataFrame(columns=["id","name","birth_date","nationality","position"])

    name_col = pick_col(df_player_stats, ["player", "player_name", "Player"])
    nat_col  = pick_col(df_player_stats, ["nation", "nationality", "Nation"])
    pos_col  = pick_col(df_player_stats, ["pos", "position", "Pos"])
    birth_col = pick_col(df_player_stats, ["birth_date", "born", "Birth"])

    players = df_player_stats[[name_col]].drop_duplicates().rename(columns={name_col: "name"}).reset_index(drop=True)
    players["id"] = players["name"].apply(lambda x: "P_" + str(abs(hash(x))) )

    players["birth_date"] = df_player_stats.drop_duplicates(subset=[name_col])[birth_col].values if birth_col else ""
    players["nationality"] = df_player_stats.drop_duplicates(subset=[name_col])[nat_col].values if nat_col else ""
    players["position"] = df_player_stats.drop_duplicates(subset=[name_col])[pos_col].values if pos_col else ""

    return players[["id","name","birth_date","nationality","position"]]

def build_player_season_stats(df_fbref, df_understat, players_df, teams_df):
    """
    player_season_stats: player_id, team_id, season_id, minutes, goals, xg, assists
    Şimdilik season_id boş kalabilir; sonra lig+sezon mapping yapacağız.
    xg: Understat varsa doldur, yoksa FBref'ten yoksa boş
    """
    cols = ["player_id","team_id","season_id","minutes","goals","xg","assists"]
    if df_fbref.empty:
        return pd.DataFrame(columns=cols)

    name_col = pick_col(df_fbref, ["player", "player_name", "Player"])
    team_col = pick_col(df_fbref, ["squad", "team", "Squad"])
    min_col  = pick_col(df_fbref, ["minutes", "min", "Min"])
    g_col    = pick_col(df_fbref, ["goals", "gls", "Gls"])
    a_col    = pick_col(df_fbref, ["assists", "ast", "Ast"])
    season_col = pick_col(df_fbref, ["season_year", "Season"])

    player_id_map = dict(zip(players_df["name"], players_df["id"]))
    team_id_map   = dict(zip(teams_df["name"], teams_df["id"]))

    out = pd.DataFrame()
    out["player_id"] = df_fbref[name_col].map(player_id_map)
    out["team_id"]   = df_fbref[team_col].map(team_id_map)
    out["season_id"] = ""  # sonraki adımda dolduracağız

    out["minutes"] = df_fbref[min_col].apply(safe_int) if min_col else None
    out["goals"]   = df_fbref[g_col].apply(safe_int) if g_col else None
    out["assists"] = df_fbref[a_col].apply(safe_int) if a_col else None
    out["xg"]      = None

    # Understat xG ekle (player_std + season_year + league_code üzerinden daha sonra güçlü eşleme yapacağız)
    if df_understat is not None and not df_understat.empty and "player_std" in df_understat.columns and "player_std" in df_fbref.columns:
        # Basit join: player_std + season_year
        us_season_col = pick_col(df_understat, ["season_year", "season"])
        fb_season_col = season_col
        us_xg_col = pick_col(df_understat, ["xg", "xG"])

        if us_season_col and fb_season_col and us_xg_col:
            us_small = df_understat[["player_std", us_season_col, us_xg_col]].copy()
            us_small = us_small.rename(columns={us_season_col: "season_year", us_xg_col: "xg"})
            us_small["xg"] = us_small["xg"].apply(safe_float)

            fb_small = df_fbref[["player_std", fb_season_col]].copy()
            fb_small = fb_small.rename(columns={fb_season_col: "season_year"})

            # index align
            out = out.join(fb_small)
            out = out.merge(us_small, on=["player_std","season_year"], how="left", suffixes=("", "_us"))
            out["xg"] = out["xg"].fillna(out["xg_us"])
            out = out.drop(columns=["xg_us","season_year","player_std"], errors="ignore")

    out = out.dropna(subset=["player_id","team_id"]).reset_index(drop=True)
    return out[cols]