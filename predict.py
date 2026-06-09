"""
predict.py
==========
Prediction helper used by app.py.

Given two team names, looks up their most recent FIFA rankings,
builds the feature vector, and returns outcome probabilities.
"""

import os
import pickle
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
MODEL_PATH  = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
META_PATH   = "models/model_meta.pkl"
RANKING_CSV = "data/fifa_ranking.csv"
RESULTS_CSV = "data/results.csv"

# ─────────────────────────────────────────────
# LOAD ARTEFACTS (cached after first call)
# ─────────────────────────────────────────────

_model  = None
_scaler = None
_meta   = None
_rankings_df = None


def _load_artefacts():
    global _model, _scaler, _meta
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Trained model not found. Please run `python train_model.py` first."
            )
        with open(MODEL_PATH,  "rb") as f: _model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: _scaler = pickle.load(f)
        with open(META_PATH,   "rb") as f: _meta   = pickle.load(f)


def _load_rankings():
    """Load and normalise the FIFA ranking CSV (cached)."""
    global _rankings_df
    if _rankings_df is not None:
        return _rankings_df

    from preprocess import prepare_rankings
    rankings = pd.read_csv(RANKING_CSV, parse_dates=["rank_date"])
    _rankings_df = prepare_rankings(rankings)
    return _rankings_df


# ─────────────────────────────────────────────
# TEAM LISTS
# ─────────────────────────────────────────────

def get_all_teams():
    """
    Return a sorted list of all team names that appear in the
    results dataset — used to populate the Streamlit dropdowns.
    """
    if not os.path.exists(RESULTS_CSV):
        return []
    df = pd.read_csv(RESULTS_CSV)
    teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    return sorted(teams.tolist())


def get_latest_rank(team: str, rankings: pd.DataFrame):
    """
    Return the most recent rank & points for a team.
    Falls back to 200 / 0 if no data exists.
    """
    subset = rankings[rankings["country_full"] == team]
    if subset.empty:
        return 200, 0
    latest = subset.sort_values("rank_date").iloc[-1]
    return int(latest["rank"]), float(latest["total_points"])


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def predict_match(home_team: str, away_team: str, neutral: bool = False):
    """
    Predict the outcome probabilities for a single match.

    Parameters
    ----------
    home_team : str   – name of the home team
    away_team : str   – name of the away team
    neutral   : bool  – True if match is on a neutral venue

    Returns
    -------
    dict with keys:
      'home_team'  : str
      'away_team'  : str
      'model_name' : str
      'Home Win'   : float (probability 0–1)
      'Draw'       : float
      'Away Win'   : float
      'prediction' : str  – most-likely outcome label
      'home_rank'  : int
      'away_rank'  : int
    """
    _load_artefacts()
    rankings = _load_rankings()

    # ── Build feature vector ──
    home_rank, home_pts = get_latest_rank(home_team, rankings)
    away_rank, away_pts = get_latest_rank(away_team, rankings)

    rank_diff   = home_rank   - away_rank
    points_diff = home_pts    - away_pts
    is_neutral  = int(neutral)

    X_raw = np.array([[rank_diff, points_diff, is_neutral]], dtype=float)
    X     = _scaler.transform(X_raw)

    # ── Predict ──
    proba      = _model.predict_proba(X)[0]          # shape (3,)
    labels     = _meta["labels"]                     # {0: 'Home Win', …}
    class_order= _model.classes_                     # order of classes

    prob_dict = {labels[c]: float(proba[i]) for i, c in enumerate(class_order)}
    prediction = max(prob_dict, key=prob_dict.get)

    return {
        "home_team":  home_team,
        "away_team":  away_team,
        "model_name": _meta["best_model_name"],
        "Home Win":   prob_dict.get("Home Win", 0.0),
        "Draw":       prob_dict.get("Draw",     0.0),
        "Away Win":   prob_dict.get("Away Win", 0.0),
        "prediction": prediction,
        "home_rank":  home_rank,
        "away_rank":  away_rank,
    }


# ─────────────────────────────────────────────
# QUICK CLI TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    result = predict_match("Brazil", "Argentina")
    print("\n=== PREDICTION ===")
    for k, v in result.items():
        print(f"  {k:<14}: {v}")
