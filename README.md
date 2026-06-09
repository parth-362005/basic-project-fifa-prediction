# ⚽ FIFA World Cup 2026 — Match Outcome Predictor

A complete end-to-end machine-learning project that predicts football match
outcomes (**Home Win / Draw / Away Win**) using historical international
results and FIFA world rankings.

---

## 📁 Project Structure

```
fifa_predictor/
├── app.py              ← Streamlit web application
├── train_model.py      ← Model training + evaluation
├── preprocess.py       ← Data loading, cleaning, feature engineering
├── predict.py          ← Prediction helper (used by app.py)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
├── data/               ← (you create this) raw & processed CSVs
│   ├── results.csv
│   └── fifa_ranking.csv
└── models/             ← (auto-created) trained model artefacts
    ├── best_model.pkl
    ├── scaler.pkl
    └── model_meta.pkl
```

---

## 📥 Step 1 — Download the Datasets

You need **two free CSV files from Kaggle**.

### 1a. International Football Results
- URL: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
- Download the file called **`results.csv`**
- Save it as: `data/results.csv`

Expected columns:
```
date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
```

### 1b. FIFA World Rankings
- URL: https://www.kaggle.com/datasets/cashncarry/fifaworldranking
- Download the file called **`fifa_ranking-2024-07-18.csv`** (or similar)
- Save it as: `data/fifa_ranking.csv`

Expected columns (any naming variant is handled automatically):
```
rank_date, country_full, rank, total_points
```

> **Note:** Both datasets are free to download with a Kaggle account.
> Create a free account at https://www.kaggle.com if you don't have one.

Create the `data/` folder manually or let the scripts create it:
```bash
mkdir data
```

---

## ⚙️ Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.9 or higher.

---

## 🏋️ Step 3 — Train the Models

```bash
python train_model.py
```

This will:
1. Run the preprocessing pipeline (merge rankings into results, engineer features)
2. Train **Logistic Regression**, **Random Forest**, and **XGBoost**
3. Evaluate each with 5-fold cross-validation + held-out test set
4. Save the best model to `models/best_model.pkl`

Typical runtime: 2–5 minutes (most time is spent merging ~50,000 matches
with the ranking history).

---

## 🚀 Step 4 — Launch the Web App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

- Select a **Home Team** and **Away Team** from the dropdowns
- Tick "Neutral venue" for World Cup group-stage matches
- Click **Predict Match Outcome**
- See win probabilities for all three outcomes

---

## 🤖 How It Works

| Step | File | What happens |
|------|------|--------------|
| Data Loading | `preprocess.py` | Reads `results.csv` + `fifa_ranking.csv` |
| Cleaning | `preprocess.py` | Drops nulls, filters to 1993–present |
| Ranking Merge | `preprocess.py` | Attaches latest FIFA rank/points to each match |
| Feature Engineering | `preprocess.py` | Creates `rank_diff`, `points_diff`, `is_neutral` |
| Model Training | `train_model.py` | LR + RF + XGB, CV + test evaluation |
| Prediction | `predict.py` | Loads saved model, scales input, returns probabilities |
| Web UI | `app.py` | Streamlit front-end with interactive dropdowns |

### Features Used

| Feature | Description |
|---------|-------------|
| `rank_diff` | Home FIFA rank − Away FIFA rank (negative = home team ranked higher) |
| `points_diff` | Home FIFA points − Away FIFA points |
| `is_neutral` | 1 if match played on neutral ground, 0 otherwise |

### Target Variable

| Label | Meaning |
|-------|---------|
| `0` | Home Win |
| `1` | Draw |
| `2` | Away Win |

---

## 📊 Expected Results

Typical test-set accuracy on the full dataset (~50k matches, 1993–2024):

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~54–56% |
| Random Forest | ~55–57% |
| XGBoost | ~56–58% |

> Football is inherently unpredictable. An accuracy of 55–58% is realistic and
> in line with published academic work on this problem.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: data/results.csv` | Download the dataset (see Step 1) |
| `FileNotFoundError: models/best_model.pkl` | Run `python train_model.py` first |
| `ModuleNotFoundError: xgboost` | Run `pip install -r requirements.txt` |
| Rankings CSV column error | Check column names match `rank_date, country_full, rank, total_points` |
| Streamlit not found | Run `pip install streamlit` |

---

## 📌 Requirements

- Python 3.9+
- pandas, numpy, scikit-learn, xgboost, streamlit

All pinned in `requirements.txt`.
