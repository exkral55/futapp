import pandas as pd

# 1) Forvetleri filtrele
def filtre_forvet(df):
    return df[df["position"] == "F"]

# 2) Forvet için basit skor hesapla
def forvet_skor(df):
    df = df.copy()
    df["skor"] = (
        df["goals"] * 4 +
        df["assists"] * 3 +
        df["xg"] * 2 +
        df["xa"] * 2 +
        df["shots"] * 0.3 +
        df["key_passes"] * 0.5
    )
    return df.sort_values("skor", ascending=False)
