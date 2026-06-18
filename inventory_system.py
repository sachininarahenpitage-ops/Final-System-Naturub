"""
Naturub Exports International (Pvt) Ltd
Combined Inventory Management System
Merges: real GRN/Issued/Requested data + LightGBM ML forecasting engine
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import os, sys, io, json, yaml

st.set_page_config(
    page_title="Naturub Inventory System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&family=IBM+Plex+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.block-container { padding-top: 1.5rem; }
.kpi { background:#0f172a; border:1px solid #1e3a5f; border-radius:10px; padding:1.2rem 0.6rem; text-align:center; margin-bottom:0.5rem; }
.kpi .label { color:#64748b; font-size:0.7rem; text-transform:uppercase; letter-spacing:2px; }
.kpi .value { color:#38bdf8; font-size:clamp(1.05rem, 1.5vw, 1.5rem); font-weight:700; font-family:'IBM Plex Mono'; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; display:block; line-height:1.3; }
.kpi .sub   { color:#94a3b8; font-size:0.75rem; }
.kpi-red .value  { color:#ef4444; }
.kpi-green .value { color:#22c55e; }
.kpi-amber .value { color:#f59e0b; }
.alert-red   { background:#1c0a0a; border-left:4px solid #ef4444; border-radius:0 8px 8px 0; padding:10px 14px; color:#fca5a5; font-size:13px; margin:4px 0; }
.alert-green { background:#052e16; border-left:4px solid #22c55e; border-radius:0 8px 8px 0; padding:10px 14px; color:#86efac; font-size:13px; margin:4px 0; }
.alert-amber { background:#1c1500; border-left:4px solid #f59e0b; border-radius:0 8px 8px 0; padding:10px 14px; color:#fcd34d; font-size:13px; margin:4px 0; }
.alert-blue  { background:#0c1445; border-left:4px solid #38bdf8; border-radius:0 8px 8px 0; padding:10px 14px; color:#bae6fd; font-size:13px; margin:4px 0; }
.gauge-red   { background:#1c0a0a; border:2px solid #ef4444; border-radius:12px; padding:1rem; text-align:center; }
.gauge-green { background:#052e16; border:2px solid #22c55e; border-radius:12px; padding:1rem; text-align:center; }
.gauge-amber { background:#1c1500; border:2px solid #f59e0b; border-radius:12px; padding:1rem; text-align:center; }
.section-title { color:#38bdf8; font-size:0.9rem; font-weight:600; text-transform:uppercase; letter-spacing:2px; margin:1.5rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Login ─────────────────────────────────────────────────────────────────────
def load_users():
    for p in [os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.txt"), "users.txt"]:
        if os.path.exists(p):
            users = {}
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    parts = line.split(",", 1)
                    if len(parts) == 2:
                        users[parts[0].strip()] = parts[1].strip()
            return users
    return {"admin": "admin123"}

def show_login():
    st.markdown("""
    <div style='text-align:center; padding-top:3rem;'>
        <span style='font-size:3rem;'>📦</span>
        <h1 style='color:#38bdf8; font-family:IBM Plex Mono; margin:0.5rem 0;'>Naturub Inventory System</h1>
        <p style='color:#64748b;'>Naturub Exports International (Pvt) Ltd</p>
    </div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔐 Sign In", use_container_width=True, type="primary"):
            users = load_users()
            if username in users and users[username] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Incorrect username or password.")
        st.markdown('<p style="text-align:center;color:#475569;font-size:0.75rem;margin-top:1rem;">Contact your administrator for access.</p>', unsafe_allow_html=True)

# ── Config (lead times) ───────────────────────────────────────────────────────
LEAD_TIMES = {
    "Cotton": 65, "Nylon": 80, "Polyester": 70,
    "Rubber": 90, "Sewing Thread": 100, "Spandex": 60, "_default": 80
}
SERVICE_LEVEL_Z = 1.65

def lead_time_for(material: str) -> int:
    for k, v in LEAD_TIMES.items():
        if k.lower() in str(material).lower():
            return v
    return LEAD_TIMES["_default"]

# ── ML Forecaster ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_artifacts():
    """Load LightGBM models and ship table if available."""
    try:
        import lightgbm as lgb
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
        if not os.path.exists(base):
            base = "artifacts"
        if not os.path.exists(base):
            return None

        models = {}
        for fname, key in [
            ("lgbm_daily_h30.txt",  ("daily",  30)),
            ("lgbm_daily_h90.txt",  ("daily",  90)),
            ("lgbm_weekly_h4.txt",  ("weekly", 30)),
            ("lgbm_weekly_h13.txt", ("weekly", 90)),
        ]:
            p = os.path.join(base, fname)
            if os.path.exists(p):
                models[key] = lgb.Booster(model_file=p)

        ship_path = os.path.join(base, "ship_table.csv")
        ship = pd.read_csv(ship_path) if os.path.exists(ship_path) else pd.DataFrame()
        item_master_path = os.path.join(base, "item_master.csv")
        item_master = pd.read_csv(item_master_path) if os.path.exists(item_master_path) else pd.DataFrame()

        return {"models": models, "ship": ship, "item_master": item_master}
    except Exception:
        return None

ML_FEATURES = ["lag_1","lag_7","lag_14","lag_30","lag_60","lag_90",
                "rmean_7","rstd_7","rmean_30","rstd_30","rmean_90","rstd_90",
                "dow","month","is_month_start","is_month_end","days_since_issue",
                "family","material","stock_type","reorder_qty","reorder_ratio"]

def ml_wins(item, grain, horizon, ship):
    if ship.empty: return False
    row = ship[(ship["item"]==item)&(ship["grain"]==grain)&(ship["horizon"]==horizon)]
    return bool(row["ml_wins"].values[0]) if len(row) else False

def build_features(issued_item: pd.DataFrame) -> pd.DataFrame | None:
    """Build ML feature row from issued history."""
    try:
        df = issued_item[["TRANSACTION_DATE","QTY"]].copy()
        df.columns = ["date","y"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").resample("D")["y"].sum().reset_index()
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) < 30: return None

        last = df.iloc[-1:]
        row = {}
        y = df["y"].values
        row["lag_1"]  = float(y[-1])
        row["lag_7"]  = float(y[-7])  if len(y)>=7  else 0.0
        row["lag_14"] = float(y[-14]) if len(y)>=14 else 0.0
        row["lag_30"] = float(y[-30]) if len(y)>=30 else 0.0
        row["lag_60"] = float(y[-60]) if len(y)>=60 else 0.0
        row["lag_90"] = float(y[-90]) if len(y)>=90 else 0.0
        row["rmean_7"]  = float(np.mean(y[-7:]))
        row["rstd_7"]   = float(np.std(y[-7:]))
        row["rmean_30"] = float(np.mean(y[-30:]))
        row["rstd_30"]  = float(np.std(y[-30:]))
        row["rmean_90"] = float(np.mean(y[-90:])) if len(y)>=90 else float(np.mean(y))
        row["rstd_90"]  = float(np.std(y[-90:]))  if len(y)>=90 else float(np.std(y))
        d = df["date"].iloc[-1]
        row["dow"]            = int(d.dayofweek)
        row["month"]          = int(d.month)
        row["is_month_start"] = int(d.is_month_start)
        row["is_month_end"]   = int(d.is_month_end)
        row["days_since_issue"] = 0
        row["family"]       = "UNKNOWN"
        row["material"]     = "Polyester"
        row["stock_type"]   = "YARN"
        row["reorder_qty"]  = 0.0
        row["reorder_ratio"]= 0.0
        return pd.DataFrame([row])
    except Exception:
        return None

def get_forecast(item_code, issued_item, horizon_days, ml_artifacts):
    """Get demand forecast — ML if it wins, else moving average baseline."""
    try:
        df = issued_item[["TRANSACTION_DATE","QTY"]].copy()
        df.columns = ["date","y"]
        df["date"] = pd.to_datetime(df["date"])
        daily = df.set_index("date").resample("D")["y"].sum()
        avg_daily = float(daily.tail(90).mean())
        baseline = avg_daily * horizon_days

        if ml_artifacts and ml_artifacts["models"]:
            grain = "daily"
            key = (grain, horizon_days)
            if key in ml_artifacts["models"]:
                wins = ml_wins(item_code, grain, horizon_days, ml_artifacts["ship"])
                if wins:
                    feat = build_features(issued_item)
                    if feat is not None:
                        pred = float(ml_artifacts["models"][key].predict(feat)[0])
                        pred = max(0.0, pred)
                        return pred, "ML 🤖", avg_daily
        return baseline, "Baseline 📊", avg_daily
    except Exception:
        avg_daily = float(issued_item["QTY"].mean()) if not issued_item.empty else 0
        return avg_daily * horizon_days, "Baseline 📊", avg_daily

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    grn    = pd.read_parquet(os.path.join(base, "GRN_Dataset.parquet"))
    issued = pd.read_parquet(os.path.join(base, "Issued_Dataset.parquet"))
    req    = pd.read_parquet(os.path.join(base, "Requested_Data.parquet"))
    for df in [grn, issued]:
        df["ITEM_CODE"]        = df["ITEM_CODE"].astype(str).str.strip()
        df["ITEM_DESCRIPTION"] = df["ITEM_DESCRIPTION"].astype(str).str.strip()
        df["QTY"]              = pd.to_numeric(df["QTY"], errors="coerce").fillna(0)
        df["BALANCE_QTY"]      = pd.to_numeric(df["BALANCE_QTY"], errors="coerce")
        df["REORDER_QTY"]      = pd.to_numeric(df["REORDER_QTY"], errors="coerce").fillna(0)
        df["SIH"]              = pd.to_numeric(df["SIH"], errors="coerce").fillna(0)
        df["TRANSACTION_DATE"] = pd.to_datetime(df["TRANSACTION_DATE"], errors="coerce")
    return grn, issued, req

def get_item_info(grn, issued, item_code):
    g = grn[grn["ITEM_CODE"] == item_code]
    i = issued[issued["ITEM_CODE"] == item_code]
    if g.empty and i.empty: return None
    desc       = g["ITEM_DESCRIPTION"].iloc[0] if not g.empty else i["ITEM_DESCRIPTION"].iloc[0]
    unit       = g["MEASUER_UNIT"].iloc[0]      if not g.empty else i["MEASUER_UNIT"].iloc[0]
    reorder    = float(g["REORDER_QTY"].iloc[0] if not g.empty else i["REORDER_QTY"].iloc[0])
    sih        = float(g["SIH"].iloc[-1]         if not g.empty else i["SIH"].iloc[-1])
    material   = g["STOCK_TYPE"].iloc[0]         if not g.empty else "Polyester"
    lt         = lead_time_for(str(material))
    return {"desc":desc,"unit":unit,"reorder_qty":reorder,"sih":sih,
            "lead_time":lt,"material":material,"grn":g,"issued":i}

def stock_status_gauge(sih, reorder_qty):
    if reorder_qty <= 0:
        pct = 100
        status, color, css = "OK", "#22c55e", "gauge-green"
    else:
        pct = (sih / reorder_qty) * 100
        if sih <= reorder_qty:
            status, color, css = "🔴 RED", "#ef4444", "gauge-red"
        elif sih <= reorder_qty * 1.2:
            status, color, css = "🟡 AMBER", "#f59e0b", "gauge-amber"
        else:
            status, color, css = "🟢 GREEN", "#22c55e", "gauge-green"
    return status, color, css, pct

def monthly_movement(grn_item, issued_item):
    grn_m = (grn_item.set_index("TRANSACTION_DATE")["QTY"]
             .resample("ME").sum().reset_index()
             .rename(columns={"TRANSACTION_DATE":"month","QTY":"received"}))
    iss_m = (issued_item.set_index("TRANSACTION_DATE")["QTY"]
             .resample("ME").sum().reset_index()
             .rename(columns={"TRANSACTION_DATE":"month","QTY":"issued"}))
    m = pd.merge(grn_m, iss_m, on="month", how="outer").fillna(0).sort_values("month")
    return m

# ── Main App ──────────────────────────────────────────────────────────────────
def show_app():
    ml_artifacts = load_ml_artifacts()
    ml_available = ml_artifacts is not None and bool(ml_artifacts.get("models"))

    with st.sidebar:
        st.markdown("## 📦 Naturub Inventory")
        st.markdown("**Naturub Exports International**")
        st.markdown("---")
        st.markdown(f"👤 **{st.session_state.get('username','')}**")
        if ml_available:
            st.markdown("🤖 **ML Engine:** Active")
        else:
            st.markdown("📊 **ML Engine:** Baseline mode")
        page = st.radio("Navigation", [
            "🔍 Item Lookup & Forecast",
            "📊 Stock Movement",
            "⚠️ Reorder Alerts",
            "🛒 Order Evaluator",
            "📅 Order Date Planner",
            "📈 Reorder Optimizer",
        ])
        st.markdown("---")
        if st.button("🚪 Log Out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    with st.spinner("Loading data..."):
        grn, issued, req = load_data()

    item_list = sorted(set(grn["ITEM_CODE"].unique()) | set(issued["ITEM_CODE"].unique()))

    # ── PAGE 1: Item Lookup & Forecast ────────────────────────────────────────
    if page == "🔍 Item Lookup & Forecast":
        st.markdown("# 🔍 Item Lookup & Demand Forecast")
        st.markdown("Search by item code or job number — view stock status, health gauge, and AI demand forecast.")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            search_code = st.text_input("Enter Item Code", placeholder="e.g. YPOL0004").strip().upper()
        with col2:
            search_job = st.text_input("Enter Job Number", placeholder="e.g. 22-Y-27238").strip()

        if search_job:
            grn_job    = grn[grn["JOB_NO"].astype(str).str.contains(search_job, na=False)]
            issued_job = issued[issued["JOB_NO"].astype(str).str.contains(search_job, na=False)]
            items_in_job = set(grn_job["ITEM_CODE"].tolist()) | set(issued_job["ITEM_CODE"].tolist())
            if items_in_job:
                st.success(f"Found {len(items_in_job)} item(s) for job {search_job}")
                search_code = st.selectbox("Select item from job", sorted(items_in_job))
            else:
                st.warning(f"No records found for job number: {search_job}")

        horizon = st.radio("Forecast horizon", [30, 90], horizontal=True, format_func=lambda x: f"{x} days")

        if search_code:
            info = get_item_info(grn, issued, search_code)
            if not info:
                st.error(f"Item code **{search_code}** not found.")
            else:
                st.markdown(f"### {search_code} — {info['desc']}")
                st.markdown(f"*{info['material']} | Unit: {info['unit']} | Lead Time: {info['lead_time']} days*")
                st.markdown("")

                # ── Stock Health Gauge ────────────────────────────────────────
                status, color, css, pct = stock_status_gauge(info["sih"], info["reorder_qty"])
                col_g, col_kpi = st.columns([1, 2])
                with col_g:
                    st.markdown(f"""
                    <div class="{css}" style="padding:1.5rem; text-align:center;">
                        <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:2px;">Stock Health</div>
                        <div style="font-size:2.5rem;font-weight:700;color:{color};font-family:IBM Plex Mono;">{status}</div>
                        <div style="font-size:0.85rem;color:#94a3b8;">Stock is {pct:.1f}% of reorder point</div>
                        <div style="background:#1e293b;border-radius:6px;height:10px;margin-top:10px;">
                            <div style="background:{color};width:{min(pct,100):.0f}%;height:10px;border-radius:6px;"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                with col_kpi:
                    c1,c2,c3,c4 = st.columns(4)
                    for col, val, label, sub in [
                        (c1, f"{info['sih']:,.1f}", "Stock in Hand", info['unit']),
                        (c2, f"{info['reorder_qty']:,.1f}", "Reorder Point", info['unit']),
                        (c3, f"{info['lead_time']} days", "Lead Time", "delivery"),
                        (c4, len(info['grn']), "GRN Transactions", "total"),
                    ]:
                        col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{val}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

                st.markdown("")

                # ── ML / Baseline Forecast ────────────────────────────────────
                st.markdown('<p class="section-title">📊 Demand Forecast</p>', unsafe_allow_html=True)
                if not info["issued"].empty:
                    forecast_val, method, avg_daily = get_forecast(
                        search_code, info["issued"], horizon, ml_artifacts)

                    # Safety stock & reorder point calculation
                    daily_std = float(info["issued"].set_index("TRANSACTION_DATE")["QTY"]
                                      .resample("D").sum().tail(180).std())
                    safety_stock = SERVICE_LEVEL_Z * daily_std * np.sqrt(info["lead_time"])
                    rop = (avg_daily * info["lead_time"]) + safety_stock

                    # Stock projection
                    days_to_rop    = max(0, (info["sih"] - rop) / avg_daily) if avg_daily > 0 else 999
                    days_to_zero   = max(0, info["sih"] / avg_daily)          if avg_daily > 0 else 999
                    rop_date       = (datetime.today() + timedelta(days=int(days_to_rop))).strftime("%d %b %Y")
                    stockout_date  = (datetime.today() + timedelta(days=int(days_to_zero))).strftime("%d %b %Y")

                    fc1,fc2,fc3,fc4 = st.columns(4)
                    fc1.markdown(f'<div class="kpi"><div class="label">Forecast ({horizon}d)</div><div class="value">{forecast_val:,.0f}</div><div class="sub">{info["unit"]} — {method}</div></div>', unsafe_allow_html=True)
                    fc2.markdown(f'<div class="kpi"><div class="label">Avg Daily Usage</div><div class="value">{avg_daily:,.1f}</div><div class="sub">{info["unit"]}/day</div></div>', unsafe_allow_html=True)
                    fc3.markdown(f'<div class="kpi"><div class="label">Model Reorder Point</div><div class="value">{rop:,.0f}</div><div class="sub">{info["unit"]}</div></div>', unsafe_allow_html=True)
                    fc4.markdown(f'<div class="kpi"><div class="label">Safety Stock</div><div class="value">{safety_stock:,.0f}</div><div class="sub">{info["unit"]}</div></div>', unsafe_allow_html=True)

                    st.markdown("")
                    # Projection alerts
                    if days_to_rop <= horizon:
                        st.markdown(f'<div class="alert-amber">⚠️ Stock will reach reorder point in <strong>{days_to_rop:.0f} days</strong> (around {rop_date}). Place order soon!</div>', unsafe_allow_html=True)
                    if days_to_zero <= horizon:
                        st.markdown(f'<div class="alert-red">🚨 Stock will run out in <strong>{days_to_zero:.0f} days</strong> (around {stockout_date}). Urgent action required!</div>', unsafe_allow_html=True)
                    if days_to_rop > horizon and days_to_zero > horizon:
                        st.markdown(f'<div class="alert-green">✅ Stock sufficient for the next {horizon} days. Reorder point reached around {rop_date}.</div>', unsafe_allow_html=True)

                # Recent transactions
                st.markdown('<p class="section-title">Recent Transactions</p>', unsafe_allow_html=True)
                t1, t2 = st.tabs(["📥 Recent Receipts (GRN)", "📤 Recent Issues"])
                with t1:
                    st.dataframe(info['grn'].sort_values("TRANSACTION_DATE", ascending=False)
                                 .head(10)[["TRANSACTION_DATE","QTY","GRN_NO","JOB_NO","COMPANY"]]
                                 .rename(columns={"TRANSACTION_DATE":"Date","QTY":"Qty","GRN_NO":"GRN No","JOB_NO":"Job No","COMPANY":"Supplier"}),
                                 use_container_width=True, hide_index=True)
                with t2:
                    st.dataframe(info['issued'].sort_values("TRANSACTION_DATE", ascending=False)
                                 .head(10)[["TRANSACTION_DATE","QTY","JOB_NO","ISSUE_BY"]]
                                 .rename(columns={"TRANSACTION_DATE":"Date","QTY":"Qty","JOB_NO":"Job No","ISSUE_BY":"Issued By"}),
                                 use_container_width=True, hide_index=True)

    # ── PAGE 2: Stock Movement ────────────────────────────────────────────────
    elif page == "📊 Stock Movement":
        st.markdown("# 📊 Stock Movement & Historical Patterns")
        st.markdown("---")
        item_code = st.selectbox("Select Item", item_list)
        info = get_item_info(grn, issued, item_code)
        if info:
            st.markdown(f"**{info['desc']}** | Unit: {info['unit']}")
            m = monthly_movement(info['grn'], info['issued'])
            if not m.empty:
                chart_type = st.radio("Chart Type", ["Bar Chart","Line Chart","Both"], horizontal=True)
                if chart_type in ["Bar Chart","Both"]:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=m["month"],y=m["received"],name="Received",marker_color="#38bdf8",opacity=0.8))
                    fig.add_trace(go.Bar(x=m["month"],y=m["issued"],name="Issued",marker_color="#f97316",opacity=0.8))
                    if info['reorder_qty']>0:
                        fig.add_hline(y=info['reorder_qty'],line_dash="dash",line_color="#ef4444",annotation_text="Reorder Point")
                    fig.update_layout(barmode="group",template="plotly_dark",title="Monthly Stock Movement — Bar",
                                      xaxis_title="Month",yaxis_title=f"Qty ({info['unit']})",height=400,
                                      legend=dict(orientation="h",y=1.1))
                    st.plotly_chart(fig,use_container_width=True)
                if chart_type in ["Line Chart","Both"]:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=m["month"],y=m["received"],name="Received",
                                              mode="lines+markers",line=dict(color="#38bdf8",width=2)))
                    fig2.add_trace(go.Scatter(x=m["month"],y=m["issued"],name="Issued",
                                              mode="lines+markers",line=dict(color="#f97316",width=2)))
                    if info['reorder_qty']>0:
                        fig2.add_hline(y=info['reorder_qty'],line_dash="dash",line_color="#ef4444",annotation_text="Reorder Point")
                    fig2.update_layout(template="plotly_dark",title="Monthly Stock Movement — Line",
                                       xaxis_title="Month",yaxis_title=f"Qty ({info['unit']})",height=400,
                                       legend=dict(orientation="h",y=1.1))
                    st.plotly_chart(fig2,use_container_width=True)

                # Cumulative trend
                m["net"] = m["received"] - m["issued"]
                m["cumulative"] = m["net"].cumsum()
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=m["month"],y=m["cumulative"],mode="lines+markers",
                                          name="Cumulative Net",line=dict(color="#22c55e",width=2)))
                if info['reorder_qty']>0:
                    below = m[m["cumulative"]<info['reorder_qty']]
                    if not below.empty:
                        fig3.add_trace(go.Scatter(x=below["month"],y=below["cumulative"],mode="markers",
                                                  name="Below Reorder",marker=dict(color="#ef4444",size=10,symbol="x")))
                fig3.update_layout(template="plotly_dark",title="Cumulative Stock Trend",height=350)
                st.plotly_chart(fig3,use_container_width=True)

                # ── Historical Patterns ───────────────────────────────────────
                st.markdown("---")
                st.markdown("## 📈 Historical Patterns")
                iss = info['issued'].copy()
                grn_i = info['grn'].copy()
                iss["year"]  = iss["TRANSACTION_DATE"].dt.year
                iss["month_num"] = iss["TRANSACTION_DATE"].dt.month
                iss["month_name"] = iss["TRANSACTION_DATE"].dt.strftime("%b")
                grn_i["year"] = grn_i["TRANSACTION_DATE"].dt.year

                tab_yr, tab_mo, tab_hm = st.tabs(["📅 Yearly","📆 Monthly","🗓️ Heatmap"])

                with tab_yr:
                    yr_iss = iss.groupby("year")["QTY"].sum().reset_index().rename(columns={"QTY":"Issued"})
                    yr_grn = grn_i.groupby("year")["QTY"].sum().reset_index().rename(columns={"QTY":"Received"})
                    yr = pd.merge(yr_grn, yr_iss, on="year", how="outer").fillna(0).sort_values("year")
                    yr["year"] = yr["year"].astype(str)
                    fig_yr = go.Figure()
                    fig_yr.add_trace(go.Bar(x=yr["year"],y=yr["Received"],name="Received",marker_color="#38bdf8",opacity=0.85))
                    fig_yr.add_trace(go.Bar(x=yr["year"],y=yr["Issued"],name="Issued",marker_color="#f97316",opacity=0.85))
                    fig_yr.add_trace(go.Scatter(x=yr["year"],y=yr["Issued"],name="Issued Trend",
                                                mode="lines+markers",line=dict(color="#fbbf24",width=2,dash="dot")))
                    fig_yr.update_layout(barmode="group",template="plotly_dark",title="Yearly Issued vs Received",height=420)
                    st.plotly_chart(fig_yr,use_container_width=True)
                    yr["Net"] = yr["Received"]-yr["Issued"]
                    st.dataframe(yr.rename(columns={"year":"Year","Received":f"Received ({info['unit']})",
                                                    "Issued":f"Issued ({info['unit']})","Net":f"Net ({info['unit']})"})
                                 .round(1),use_container_width=True,hide_index=True)

                with tab_mo:
                    mo_iss = iss.groupby(["month_num","month_name"])["QTY"].sum().reset_index().rename(columns={"QTY":"Issued"})
                    mo_grn = grn_i.copy()
                    mo_grn["month_num"]  = mo_grn["TRANSACTION_DATE"].dt.month
                    mo_grn["month_name"] = mo_grn["TRANSACTION_DATE"].dt.strftime("%b")
                    mo_grn2 = mo_grn.groupby(["month_num","month_name"])["QTY"].sum().reset_index().rename(columns={"QTY":"Received"})
                    mo = pd.merge(mo_grn2,mo_iss,on=["month_num","month_name"],how="outer").fillna(0).sort_values("month_num")
                    fig_mo = go.Figure()
                    fig_mo.add_trace(go.Bar(x=mo["month_name"],y=mo["Received"],name="Received",marker_color="#38bdf8",opacity=0.85))
                    fig_mo.add_trace(go.Bar(x=mo["month_name"],y=mo["Issued"],name="Issued",marker_color="#f97316",opacity=0.85))
                    fig_mo.add_trace(go.Scatter(x=mo["month_name"],y=mo["Issued"],name="Issued Trend",
                                                mode="lines+markers",line=dict(color="#fbbf24",width=2,dash="dot")))
                    fig_mo.update_layout(barmode="group",template="plotly_dark",title="Monthly Pattern (All Years Combined)",height=420)
                    st.plotly_chart(fig_mo,use_container_width=True)
                    if not mo.empty:
                        peak = mo.loc[mo["Issued"].idxmax(),"month_name"]
                        low  = mo.loc[mo["Issued"].idxmin(),"month_name"]
                        c1,c2 = st.columns(2)
                        c1.markdown(f'<div class="alert-amber">📈 Peak usage month: <strong>{peak}</strong></div>',unsafe_allow_html=True)
                        c2.markdown(f'<div class="alert-blue">📉 Lowest usage month: <strong>{low}</strong></div>',unsafe_allow_html=True)

                with tab_hm:
                    hm = iss.groupby(["year","month_num"])["QTY"].sum().reset_index()
                    hm_pivot = hm.pivot(index="year",columns="month_num",values="QTY").fillna(0)
                    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                    hm_pivot.columns = [month_names[c-1] for c in hm_pivot.columns]
                    fig_hm = go.Figure(go.Heatmap(z=hm_pivot.values,x=hm_pivot.columns.tolist(),
                                                   y=hm_pivot.index.astype(str).tolist(),colorscale="Blues",
                                                   text=hm_pivot.values.round(0),texttemplate="%{text:,.0f}",
                                                   hovertemplate="Year: %{y}<br>Month: %{x}<br>Issued: %{z:,.1f}<extra></extra>"))
                    fig_hm.update_layout(template="plotly_dark",title="Issued Quantity Heatmap — Year × Month",height=400)
                    st.plotly_chart(fig_hm,use_container_width=True)
                    st.markdown('<div class="alert-blue">ℹ️ Darker = higher quantity issued in that month/year.</div>',unsafe_allow_html=True)

    # ── PAGE 3: Reorder Alerts ────────────────────────────────────────────────
    elif page == "⚠️ Reorder Alerts":
        st.markdown("# ⚠️ Reorder Alerts")
        st.markdown("All items currently at or below their reorder point.")
        st.markdown("---")
        with st.spinner("Scanning all items..."):
            latest = (grn.sort_values("TRANSACTION_DATE").groupby("ITEM_CODE").last().reset_index()
                      [["ITEM_CODE","ITEM_DESCRIPTION","SIH","REORDER_QTY","STOCK_TYPE","MEASUER_UNIT"]])
            latest["SIH"]         = pd.to_numeric(latest["SIH"],errors="coerce").fillna(0)
            latest["REORDER_QTY"] = pd.to_numeric(latest["REORDER_QTY"],errors="coerce").fillna(0)
            alerts = latest[(latest["REORDER_QTY"]>0)&(latest["SIH"]<=latest["REORDER_QTY"])].copy()
            alerts["Stock Gap"] = (alerts["REORDER_QTY"]-alerts["SIH"]).round(2)
            alerts["Status"] = alerts.apply(lambda r: "🔴 RED" if r["SIH"]<=r["REORDER_QTY"] else "🟡 AMBER", axis=1)
            alerts = alerts.sort_values("Stock Gap",ascending=False)

        c1,c2,c3 = st.columns(3)
        c1.markdown(f'<div class="kpi kpi-red"><div class="label">Items Below Reorder</div><div class="value">{len(alerts)}</div><div class="sub">require attention</div></div>',unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi"><div class="label">Total Items</div><div class="value">{len(latest)}</div><div class="sub">tracked</div></div>',unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi kpi-amber"><div class="label">Alert Rate</div><div class="value">{len(alerts)/max(len(latest),1)*100:.1f}%</div><div class="sub">at risk</div></div>',unsafe_allow_html=True)

        st.markdown("")
        type_filter = st.selectbox("Filter by Stock Type",["All"]+sorted(latest["STOCK_TYPE"].dropna().unique().tolist()))
        disp = alerts if type_filter=="All" else alerts[alerts["STOCK_TYPE"]==type_filter]
        st.dataframe(disp[["ITEM_CODE","ITEM_DESCRIPTION","STOCK_TYPE","MEASUER_UNIT","SIH","REORDER_QTY","Stock Gap","Status"]]
                     .rename(columns={"ITEM_CODE":"Code","ITEM_DESCRIPTION":"Description","STOCK_TYPE":"Type",
                                      "MEASUER_UNIT":"Unit","SIH":"Stock in Hand","REORDER_QTY":"Reorder Point","Stock Gap":"Gap"}),
                     use_container_width=True,hide_index=True)
        st.download_button("⬇️ Download Alert List",disp.to_csv(index=False).encode(),"reorder_alerts.csv","text/csv")

    # ── PAGE 4: Order Evaluator ───────────────────────────────────────────────
    elif page == "🛒 Order Evaluator":
        st.markdown("# 🛒 Order Evaluator")
        st.markdown("Check if current stock can fulfill an incoming order.")
        st.markdown("---")
        col1,col2 = st.columns(2)
        with col1:
            item_code  = st.selectbox("Select Item",item_list)
            order_qty  = st.number_input("Order Requires (Qty)",min_value=0.0,step=10.0,value=1000.0)
        with col2:
            order_date = st.date_input("Required By Date",value=datetime.today()+timedelta(days=30))

        if st.button("🔍 Evaluate Order",type="primary"):
            info = get_item_info(grn,issued,item_code)
            if info:
                available = info['sih']
                shortfall = max(0,order_qty-available)
                sufficient = available >= order_qty
                c1,c2,c3 = st.columns(3)
                c1.markdown(f'<div class="kpi"><div class="label">Available Stock</div><div class="value">{available:,.1f}</div><div class="sub">{info["unit"]}</div></div>',unsafe_allow_html=True)
                c2.markdown(f'<div class="kpi"><div class="label">Order Required</div><div class="value">{order_qty:,.1f}</div><div class="sub">{info["unit"]}</div></div>',unsafe_allow_html=True)
                c3.markdown(f'<div class="{"kpi kpi-red" if shortfall>0 else "kpi kpi-green"}"><div class="label">Shortfall</div><div class="value">{shortfall:,.1f}</div><div class="sub">{info["unit"]}</div></div>',unsafe_allow_html=True)
                st.markdown("")
                if sufficient:
                    st.markdown(f'<div class="alert-green">✅ SUFFICIENT — Stock ({available:,.1f}) covers the order ({order_qty:,.1f} {info["unit"]}). Order can proceed.</div>',unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-red">❌ INSUFFICIENT — Shortfall of {shortfall:,.1f} {info["unit"]}. Purchase order needed.</div>',unsafe_allow_html=True)
                    days_left = (pd.Timestamp(order_date)-pd.Timestamp.today()).days
                    order_by  = pd.Timestamp(order_date)-timedelta(days=info["lead_time"])
                    if days_left < info["lead_time"]:
                        st.markdown(f'<div class="alert-amber">⚠️ URGENT — Only {days_left} days to required date but lead time is {info["lead_time"]} days!</div>',unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="alert-blue">📅 Place order by: <strong>{order_by.strftime("%d %B %Y")}</strong></div>',unsafe_allow_html=True)

    # ── PAGE 5: Order Date Planner ────────────────────────────────────────────
    elif page == "📅 Order Date Planner":
        st.markdown("# 📅 Order Date Planner")
        st.markdown("Calculate the latest date to place an order for on-time delivery.")
        st.markdown("---")
        col1,col2 = st.columns(2)
        with col1:
            item_code     = st.selectbox("Select Item",item_list)
            required_date = st.date_input("Material Required By",value=datetime.today()+timedelta(days=60))
        with col2:
            custom_lead   = st.number_input("Override Lead Time (days, 0 = system value)",min_value=0,value=0)

        if st.button("📅 Calculate Order Date",type="primary"):
            info = get_item_info(grn,issued,item_code)
            if info:
                lt       = custom_lead if custom_lead>0 else info["lead_time"]
                order_by = pd.Timestamp(required_date)-timedelta(days=lt)
                days_left= (order_by-pd.Timestamp.today()).days
                c1,c2,c3 = st.columns(3)
                c1.markdown(f'<div class="kpi"><div class="label">Required By</div><div class="value">{pd.Timestamp(required_date).strftime("%d %b")}</div><div class="sub">{pd.Timestamp(required_date).strftime("%Y")}</div></div>',unsafe_allow_html=True)
                c2.markdown(f'<div class="kpi"><div class="label">Lead Time</div><div class="value">{lt} days</div><div class="sub">supplier</div></div>',unsafe_allow_html=True)
                c3.markdown(f'<div class="kpi {"kpi-red" if days_left<0 else "kpi-green"}"><div class="label">Order By</div><div class="value">{order_by.strftime("%d %b")}</div><div class="sub">{order_by.strftime("%Y")}</div></div>',unsafe_allow_html=True)
                st.markdown("")
                if days_left < 0:
                    st.markdown(f'<div class="alert-red">🚨 OVERDUE — Order should have been placed {abs(days_left)} days ago! Expedite immediately.</div>',unsafe_allow_html=True)
                elif days_left <= 7:
                    st.markdown(f'<div class="alert-amber">⚠️ URGENT — {days_left} days left. Order by <strong>{order_by.strftime("%d %B %Y")}</strong>.</div>',unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-green">✅ Recommended order date: <strong>{order_by.strftime("%d %B %Y")}</strong> — {days_left} days from today.</div>',unsafe_allow_html=True)
                st.markdown(f'<div class="alert-blue">📦 <strong>{info["desc"][:80]}</strong><br>Required by {pd.Timestamp(required_date).strftime("%d %B %Y")} → Order by <strong>{order_by.strftime("%d %B %Y")}</strong></div>',unsafe_allow_html=True)

    # ── PAGE 6: Reorder Optimizer ─────────────────────────────────────────────
    elif page == "📈 Reorder Optimizer":
        st.markdown("# 📈 Reorder Level Optimizer")
        st.markdown("Compare system-recommended reorder levels vs company-defined levels.")
        st.markdown("---")
        item_code = st.selectbox("Select Item",item_list)
        info = get_item_info(grn,issued,item_code)
        if info and not info["issued"].empty:
            daily_usage = info["issued"].set_index("TRANSACTION_DATE")["QTY"].resample("D").sum()
            avg_daily   = float(daily_usage.mean())
            daily_std   = float(daily_usage.tail(180).std())
            safety      = SERVICE_LEVEL_Z * daily_std * np.sqrt(info["lead_time"])
            rec_reorder = round((avg_daily * info["lead_time"]) + safety, 2)
            current     = info["reorder_qty"]

            st.markdown(f"**{info['desc']}** | Lead Time: {info['lead_time']} days | Service Level: 95%")
            c1,c2,c3 = st.columns(3)
            c1.markdown(f'<div class="kpi"><div class="label">Avg Daily Usage</div><div class="value">{avg_daily:,.1f}</div><div class="sub">{info["unit"]}/day</div></div>',unsafe_allow_html=True)
            c2.markdown(f'<div class="kpi kpi-green"><div class="label">Recommended Reorder</div><div class="value">{rec_reorder:,.0f}</div><div class="sub">{info["unit"]}</div></div>',unsafe_allow_html=True)
            c3.markdown(f'<div class="kpi kpi-amber"><div class="label">Current Reorder Point</div><div class="value">{current:,.0f}</div><div class="sub">{info["unit"]}</div></div>',unsafe_allow_html=True)
            st.markdown("")
            diff = rec_reorder - current
            if abs(diff)<0.01:
                st.markdown('<div class="alert-green">✅ Current reorder level matches the recommendation perfectly.</div>',unsafe_allow_html=True)
            elif diff>0:
                st.markdown(f'<div class="alert-amber">⚠️ Recommendation is {diff:,.0f} higher than current. Consider increasing to reduce stockout risk.</div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-blue">ℹ️ Current reorder point is {abs(diff):,.0f} higher than recommendation — conservative buffer in place.</div>',unsafe_allow_html=True)

            fig = go.Figure(go.Bar(x=["Current Reorder","Recommended","Stock in Hand"],
                                   y=[current,rec_reorder,info["sih"]],
                                   marker_color=["#f97316","#38bdf8","#22c55e"],
                                   text=[f"{current:,.0f}",f"{rec_reorder:,.0f}",f"{info['sih']:,.0f}"],
                                   textposition="outside"))
            fig.update_layout(template="plotly_dark",title="Reorder Level Comparison",
                              yaxis_title=f"Qty ({info['unit']})",height=380)
            st.plotly_chart(fig,use_container_width=True)

            mo_usage = daily_usage.resample("ME").sum().reset_index()
            fig2 = px.line(mo_usage,x="TRANSACTION_DATE",y="QTY",template="plotly_dark",
                           title="Monthly Consumption Trend",
                           labels={"TRANSACTION_DATE":"Month","QTY":f"Qty ({info['unit']})"})
            fig2.add_hline(y=rec_reorder,line_dash="dash",line_color="#38bdf8",annotation_text="Recommended")
            if current>0:
                fig2.add_hline(y=current,line_dash="dot",line_color="#f97316",annotation_text="Current")
            st.plotly_chart(fig2,use_container_width=True)

# ── Entry ─────────────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if not st.session_state["logged_in"]:
    show_login()
else:
    show_app()
