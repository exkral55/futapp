import pandas as pd
import soccerdata as sd
from src.scorer import filtre_forvet, forvet_skor

print(">>> main.py dosyası çalıştı")

def main():
    print("📊 Understat veri çekimi başlıyor...")

    # Understat bağlantısı
    us = sd.Understat(
        leagues="ENG-Premier League",
        seasons="2023"
    )

    # Oyuncu sezon verisini çek
    df_players = us.read_player_season_stats()

    print("\n=== OYUNCU SEZON İSTATİSTİKLERİ (ilk 10) ===")
    print(df_players.head(10).to_string(index=False))

    # CSV kaydet
    df_players.to_csv("players_ENG_PL_2023.csv", index=False, encoding="utf-8-sig")
    print("\n✅ players_ENG_PL_2023.csv kaydedildi!")

    # --- Forvet Analizi ---
    print("\n📌 Forvetler filtreleniyor...")
    forvetler = filtre_forvet(df_players)

    print("📌 Forvet skorları hesaplanıyor...")
    skorlu_forvetler = forvet_skor(forvetler)

    print("\n🏆 EN İYİ 10 FORVET:")
    print(
        skorlu_forvetler[
            ["player_id", "team_id", "position", "goals", "assists", "xg", "xa", "skor"]
        ].head(10).to_string(index=False)
    )

if __name__ == "__main__":
    print(">>> main() çağrılıyor...")
    main()