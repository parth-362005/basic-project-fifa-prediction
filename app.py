"""
app.py
======
Streamlit web application for the FIFA World Cup 2026 Match Outcome Predictor.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="FIFA WC 2026 Predictor",
    page_icon="⚽",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0a1628; }

    /* Cards */
    .pred-card {
        background: linear-gradient(135deg, #1a2a4a 0%, #0d1f3c 100%);
        border: 1px solid #2a4a7f;
        border-radius: 16px;
        padding: 24px 28px;
        margin: 12px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }

    /* Outcome badge */
    .outcome-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 24px;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    /* Probability bar label */
    .bar-label {
        color: #c8d8f0;
        font-size: 0.92rem;
        margin-bottom: 2px;
    }

    /* Section header */
    .section-hdr {
        color: #7aa7e0;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* Team names */
    .team-name {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 800;
    }

    /* Rank badge */
    .rank-badge {
        background: #1e3560;
        color: #7aa7e0;
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Divider */
    hr.styled { border: none; border-top: 1px solid #2a4a7f; margin: 18px 0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_predict_module():
    """Import predict module (cached so artefacts are loaded once)."""
    from predict import predict_match, get_all_teams
    return predict_match, get_all_teams


def model_ready() -> bool:
    return os.path.exists("models/best_model.pkl")


def render_prob_bar(label: str, prob: float, color: str):
    """Render a labelled progress bar for a probability value."""
    pct = round(prob * 100, 1)
    st.markdown(f'<p class="bar-label">{label}</p>', unsafe_allow_html=True)
    st.progress(prob)
    st.markdown(f"**{pct}%**")


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 10px 0 4px 0;">
    <span style="font-size:3rem;">⚽</span>
    <h1 style="color:#ffffff; margin:0; font-size:2rem; font-weight:900;
               letter-spacing:-0.01em;">
        FIFA World Cup 2026
    </h1>
    <p style="color:#7aa7e0; font-size:1.05rem; margin-top:4px;">
        Match Outcome Predictor — Powered by Machine Learning
    </p>
</div>
<hr class="styled">
""", unsafe_allow_html=True)


# ── Check model exists ───────────────────────────────────────────────────────
if not model_ready():
    st.error(
        "🚫 **Trained model not found.**\n\n"
        "Please run the following command in your terminal first:\n\n"
        "```bash\npython train_model.py\n```"
    )
    st.stop()


# ── Load predict helpers ─────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    predict_match, get_all_teams = load_predict_module()

teams = get_all_teams()
if not teams:
    st.error("Could not load team list. Make sure `data/results.csv` exists.")
    st.stop()

# Add popular WC 2026 teams at the top of the list for convenience
wc_favorites = [
    "Brazil", "Argentina", "France", "Spain", "England",
    "Germany", "Portugal", "Netherlands", "Belgium", "Uruguay",
    "Croatia", "Morocco", "Senegal", "Japan", "United States",
    "Mexico", "Canada", "Colombia", "Ecuador", "Australia",
]
# Keep only teams that actually exist in the dataset
wc_favorites = [t for t in wc_favorites if t in teams]
other_teams  = [t for t in teams if t not in wc_favorites]
team_options = wc_favorites + ["─── All Teams ───"] + other_teams


def clean_option(t):
    """Remove separator pseudo-option."""
    return t if t != "─── All Teams ───" else None


# ── Team Selection ────────────────────────────────────────────────────────────
st.markdown('<p class="section-hdr">Select Teams</p>', unsafe_allow_html=True)

col1, col_vs, col2 = st.columns([5, 1, 5])

with col1:
    home_sel = st.selectbox(
        "🏠 Home Team",
        team_options,
        index=team_options.index("Brazil") if "Brazil" in team_options else 0,
    )

with col_vs:
    st.markdown(
        "<div style='text-align:center; color:#7aa7e0; "
        "font-size:1.4rem; font-weight:900; padding-top:32px;'>VS</div>",
        unsafe_allow_html=True
    )

with col2:
    default_away = "Argentina" if "Argentina" in team_options else team_options[1]
    away_sel = st.selectbox(
        "✈️ Away Team",
        team_options,
        index=team_options.index(default_away),
    )

neutral = st.checkbox("⚖️ Neutral venue (e.g. World Cup group stage)", value=True)

st.markdown("")

# ── Validate selection ───────────────────────────────────────────────────────
home_team = clean_option(home_sel)
away_team = clean_option(away_sel)

if not home_team or not away_team:
    st.warning("Please select valid teams from the lists above.")
    st.stop()

if home_team == away_team:
    st.warning("Please select two **different** teams.")
    st.stop()


# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("🔮  Predict Match Outcome", use_container_width=True, type="primary"):
    with st.spinner("Running prediction…"):
        try:
            result = predict_match(home_team, away_team, neutral=neutral)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # ── Result Card ──────────────────────────────────────────────────────
    outcome = result["prediction"]
    outcome_colors = {
        "Home Win":  "#22c55e",   # green
        "Draw":      "#f59e0b",   # amber
        "Away Win":  "#ef4444",   # red
    }
    badge_color = outcome_colors.get(outcome, "#7aa7e0")

    st.markdown("---")
    st.markdown('<p class="section-hdr">Prediction Result</p>',
                unsafe_allow_html=True)

    # Team comparison header
    c1, c2, c3 = st.columns([4, 2, 4])
    with c1:
        st.markdown(
            f'<p class="team-name">{home_team}</p>'
            f'<span class="rank-badge">FIFA #{result["home_rank"]}</span>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div style="text-align:center; padding-top:10px;">'
            f'<span class="outcome-badge" style="background:{badge_color}20;'
            f'color:{badge_color}; border:2px solid {badge_color};">'
            f'{outcome}</span></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<p class="team-name" style="text-align:right">{away_team}</p>'
            f'<div style="text-align:right">'
            f'<span class="rank-badge">FIFA #{result["away_rank"]}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Probability bars
    st.markdown('<p class="section-hdr">Win Probabilities</p>',
                unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        hw = result["Home Win"]
        st.metric(label=f"🏠 {home_team} Win", value=f"{hw*100:.1f}%")
        st.progress(hw)

    with pc2:
        dr = result["Draw"]
        st.metric(label="🤝 Draw", value=f"{dr*100:.1f}%")
        st.progress(dr)

    with pc3:
        aw = result["Away Win"]
        st.metric(label=f"✈️ {away_team} Win", value=f"{aw*100:.1f}%")
        st.progress(aw)

    # Model used
    st.markdown(
        f'<p style="color:#4a6a9a; font-size:0.8rem; text-align:center; '
        f'margin-top:12px;">Model: {result["model_name"]}</p>',
        unsafe_allow_html=True,
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ About This App")
    st.markdown("""
This app uses **historical international football results** and 
**FIFA world rankings** to predict the outcome of any match.

**Three models are trained:**
- Logistic Regression
- Random Forest
- XGBoost

The best-performing model (by test accuracy) is automatically selected.

---
**Features used:**
- FIFA ranking difference
- FIFA points difference  
- Neutral venue flag

---
**Data Sources:**  
[Kaggle – International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

[Kaggle – FIFA World Rankings](https://www.kaggle.com/datasets/cashncarry/fifaworldranking)
""")

    # Show model comparison if meta exists
    if os.path.exists("models/model_meta.pkl"):
        with open("models/model_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        st.markdown("---")
        st.markdown("### 📊 Model Accuracies")
        for name, acc in meta.get("results", {}).items():
            best = "⭐ " if name == meta.get("best_model_name") else "  "
            st.markdown(f"{best}**{name}**: `{acc:.4f}`")

    st.markdown("---")
    st.markdown(
        '<p style="color:#4a6a9a; font-size:0.75rem;">'
        'Built with Python, scikit-learn, XGBoost & Streamlit</p>',
        unsafe_allow_html=True,
    )
