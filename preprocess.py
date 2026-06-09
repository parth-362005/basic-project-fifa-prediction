"""
preprocess.py
=============
Handles all data loading, cleaning, and feature engineering for the
FIFA World Cup 2026 Match Outcome Predictor.

Datasets expected (download instructions in README.md):
  data/results.csv       – historical international match results
  data/fifa_ranking.csv  – historical FIFA world rankings
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────────

def load_data(results_path="data/results.csv",
              ranking_path="data/fifa_ranking.csv"):
    """
    Load match results and FIFA ranking CSVs.
    Returns two DataFrames: (results, rankings).
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"'{results_path}' not found.\n"
            "Please follow the dataset download instructions in README.md."
        )
    if not os.path.exists(ranking_path):
        raise FileNotFoundError(
            f"'{ranking_path}' not found.\n"
            "Please follow the dataset download instructions in README.md."
        )

    results  = pd.read_csv(results_path, parse_dates=["date"])
    rankings = pd.read_csv(ranking_path, parse_dates=["rank_date"])

    print(f"[INFO] Loaded {len(results):,} match records.")
    print(f"[INFO] Loaded {len(rankings):,} ranking records.")
    return results, rankings


# ─────────────────────────────────────────────
# 2. CLEAN RESULTS
# ─────────────────────────────────────────────

def clean_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop rows with missing scores
    - Keep only matches from 1993 onwards (FIFA rankings era)
    - Add an 'outcome' column: 0 = Home Win, 1 = Draw, 2 = Away Win
    """
    df = results.dropna(subset=["home_score", "away_score"]).copy()
    df = df[df["date"] >= "1993-01-01"].reset_index(drop=True)

    # Encode outcome
    conditions = [
        df["home_score"] > df["away_score"],   # Home Win
        df["home_score"] == df["away_score"],  # Draw
        df["home_score"] < df["away_score"],   # Away Win
    ]
    df["outcome"] = np.select(conditions, [0, 1, 2])

    print(f"[INFO] After cleaning: {len(df):,} matches remain.")
    return df


# ─────────────────────────────────────────────
# 3. PREPARE RANKINGS LOOKUP
# ─────────────────────────────────────────────

def prepare_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise ranking columns so we have:
      rank_date | country_full | rank | total_points
    """
    # Different Kaggle ranking datasets use slightly different column names
    col_map = {}
    cols = rankings.columns.str.lower().tolist()

    # Map common variants
    for c in rankings.columns:
        cl = c.lower()
        if "country" in cl or "team" in cl:
            col_map[c] = "country_full"
        elif cl == "rank":
            col_map[c] = "rank"
        elif "point" in cl or "total" in cl:
            col_map[c] = "total_points"
        elif "date" in cl:
            col_map[c] = "rank_date"

    rankings = rankings.rename(columns=col_map)

    required = {"rank_date", "country_full", "rank"}
    missing  = required - set(rankings.columns)
    if missing:
        raise ValueError(f"Rankings CSV is missing columns: {missing}\n"
                         f"Found columns: {list(rankings.columns)}")

    if "total_points" not in rankings.columns:
        rankings["total_points"] = 0   # fallback

    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])
    rankings = rankings.sort_values("rank_date")
    return rankings[["rank_date", "country_full", "rank", "total_points"]]


# ─────────────────────────────────────────────
# 4. MERGE RANKINGS INTO MATCHES
# ─────────────────────────────────────────────

def get_ranking_on_date(team: str, match_date,
                        rankings: pd.DataFrame):
    """
    Return the most-recent FIFA rank & points for *team* 
    that was published on or before *match_date*.
    Falls back to 200 / 0 if no data found.
    """
    mask = (rankings["country_full"] == team) & \
           (rankings["rank_date"] <= match_date)
    subset = rankings.loc[mask]
    if subset.empty:
        return 200, 0   # default for unknown teams
    latest = subset.iloc[-1]
    return latest["rank"], latest["total_points"]


def merge_rankings(matches: pd.DataFrame,
                   rankings: pd.DataFrame) -> pd.DataFrame:
    """
    For every match add:
      home_rank, home_points, away_rank, away_points
    This step can take a minute for large datasets.
    """
    print("[INFO] Merging FIFA rankings into match data (may take ~60s)…")

    home_ranks, home_pts = [], []
    away_ranks, away_pts = [], []

    for _, row in matches.iterrows():
        hr, hp = get_ranking_on_date(row["home_team"], row["date"], rankings)
        ar, ap = get_ranking_on_date(row["away_team"], row["date"], rankings)
        home_ranks.append(hr); home_pts.append(hp)
        away_ranks.append(ar); away_pts.append(ap)

    matches = matches.copy()
    matches["home_rank"]   = home_ranks
    matches["home_points"] = home_pts
    matches["away_rank"]   = away_ranks
    matches["away_points"] = away_pts

    print("[INFO] Rankings merged successfully.")
    return matches


# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model-ready features:
      rank_diff      – home_rank minus away_rank  (negative = home team ranked higher)
      points_diff    – home_points minus away_points
      is_neutral     – 1 if match played on neutral ground, else 0
    """
    df = df.copy()
    df["rank_diff"]   = df["home_rank"]   - df["away_rank"]
    df["points_diff"] = df["home_points"] - df["away_points"]
    df["is_neutral"]  = df["neutral"].astype(int) if "neutral" in df.columns else 0
    return df


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(results_path="data/results.csv",
                 ranking_path="data/fifa_ranking.csv",
                 save_path="data/processed_matches.csv"):
    """
    Full preprocessing pipeline. Saves processed CSV and returns DataFrame.
    """
    results, rankings = load_data(results_path, ranking_path)
    results   = clean_results(results)
    rankings  = prepare_rankings(rankings)
    matches   = merge_rankings(results, rankings)
    matches   = engineer_features(matches)

    os.makedirs("data", exist_ok=True)
    matches.to_csv(save_path, index=False)
    print(f"[INFO] Processed data saved to '{save_path}'.")
    return matches


if __name__ == "__main__":
    run_pipeline()
