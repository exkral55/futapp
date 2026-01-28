import pandas as pd
import soccerdata as sd

def cek_understat_lig_sezon(leagues: str, seasons: list[str]) -> pd.DataFrame:
    """Understat'tan oyuncu sezon istatistiklerini çeker ve DataFrame döner."""
    us = sd.Understat(leagues=leagues, seasons=seasons)
    df = us.read_player_season_stats()
    return df

def main():
    # ✅ Understat'ın desteklediği 5 büyük lig (1. lig)
    ligler = [
        "ENG-Premier League",
        "ESP-La Liga",
        "GER-Bundesliga",
        "ITA-Serie A",
        "FRA-Ligue 1",
    ]

    # ✅ Son 5 sezon (Understat tarafında yıl formatı kullanıyoruz)
    # 2019 = 2019-2020 sezonu gibi düşünebilirsin
    sezonlar = ["2019", "2020", "2021", "2022", "2023","2024","2025"]

    tum = []

    print("📥 Veri çekimi başlıyor...")
    for lig in ligler:
        print(f"\n--- {lig} ---")
        df = cek_understat_lig_sezon(lig, sezonlar)

        # Kaynak bilgisi ekleyelim (ileride filtre için çok önemli)
        df["source"] = "understat"
        df["league_name"] = lig

        print(f"✅ Satır sayısı: {len(df)}")
        tum.append(df)

    # Hepsini birleştir
    df_all = pd.concat(tum, ignore_index=True)

    # Basit temizlik (kolonlar varsa)
    keep_cols = [c for c in [
        "league_name", "season_id", "team_id", "player_id", "position",
        "matches", "minutes", "goals", "assists", "xg", "xa",
        "shots", "key_passes", "yellow_cards", "red_cards",
        "xg_chain", "xg_buildup", "source"
    ] if c in df_all.columns]
    df_all = df_all[keep_cols]

    out_csv = "players_2019-25.csv"
    df_all.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n🎉 BİTTİ!")
    print(f"✅ Toplam satır: {len(df_all)}")
    print(f"✅ CSV: {out_csv}")

    print("\n📌 Örnek (ilk 10):")
    print(df_all.head(10).to_string(index=False))

if __name__ == "__main__":
    main()