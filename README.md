# 📦 Demand Forecasting AI — LightGBM Challenger

A Streamlit web app that trains a LightGBM demand forecasting model and compares it against your existing baseline forecasts. The AI only "wins" for items where it improves accuracy by **≥ 10%**.

---

## 🗂️ Files in this Repo

```
lgbm_app/
├── app.py                        ← Streamlit web app (main file)
├── train_lightgbm_colab.py       ← Original training script (run in Colab)
├── requirements.txt              ← Python dependencies
└── README.md                     ← This file
```

---

## 🚀 How to Deploy (Step-by-Step)

### Step 1 — Put the code on GitHub

1. Go to [github.com](https://github.com) and sign in (or create a free account)
2. Click the **+** button → **New repository**
3. Name it `demand-forecasting-ai` → click **Create repository**
4. Upload these files: `app.py`, `requirements.txt`, `README.md`, `train_lightgbm_colab.py`
   - Click **Add file → Upload files** and drag them in
   - Click **Commit changes**

---

### Step 2 — Deploy on Streamlit Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **New app**
4. Fill in:
   - **Repository:** `your-username/demand-forecasting-ai`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy!**
6. In ~2 minutes your app is live at a public URL like:
   `https://your-username-demand-forecasting-ai.streamlit.app`

---

### Step 3 — Use the App

1. Open the URL from Step 2
2. Go to the **Upload & Train** tab
3. Upload your three data files:
   - `train_daily.parquet`
   - `train_weekly.parquet`
   - `baseline_to_beat.parquet`
4. Click **🚀 Train & Evaluate**
5. View results in the **Results** and **Report** tabs
6. Download `phase4_eval.csv` and `phase4_report.md`

---

## 💻 Run Locally (optional)

If you want to run it on your own computer instead:

```bash
# 1. Install Python 3.10+ if you don't have it
# 2. Open terminal/command prompt and run:

pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## 📊 What the App Does

| Step | What happens |
|------|-------------|
| Upload | You provide 3 parquet files with features and baseline MAE values |
| Train | LightGBM trains 4 models (daily × 30d/90d, weekly × 30d/90d) with 5-fold cross-validation |
| Evaluate | Each item's ML prediction is compared to its held-out actual value |
| Decision | ML only "ships" for an item if it beats the baseline by ≥ 10% MAE |
| Output | `phase4_eval.csv` (all results) + `phase4_report.md` (summary) |

---

## ❓ FAQ

**Q: What are the 3 parquet files?**  
These come from Phase 3 of your data pipeline. If you don't have them, run the preprocessing steps first.

**Q: Training takes a long time?**  
Yes — LightGBM with 1200 estimators on large datasets can take 10–30 minutes. The progress bar will keep you updated.

**Q: Can I share the app URL with my boss?**  
Yes! Streamlit Cloud apps are publicly accessible. Anyone with the link can use it.

---

*Built with [Streamlit](https://streamlit.io) + [LightGBM](https://lightgbm.readthedocs.io)*
