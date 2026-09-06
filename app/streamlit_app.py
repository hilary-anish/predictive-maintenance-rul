"""
Predictive Maintenance KPI Dashboard.

Shows fleet health, maintenance priorities, model performance,
drift status, and retrain history. Designed for non-technical
stakeholders (maintenance managers, operations leads).

Why Streamlit?
- Python-native (no JS/HTML needed)
- Interactive widgets with zero frontend code
- Easy to deploy on Hugging Face Spaces (free)
- Recruiters can see a live demo without setup

Why Plotly for charts?
- Interactive (hover, zoom, pan) vs matplotlib's static images
- Better looking out of the box
- Consistent styling with minimal code
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────
# Must be the first Streamlit command in the script.
# layout="wide" uses the full browser width instead of a narrow column.
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 Predictive Maintenance Dashboard")
st.caption("Fleet health monitoring • RUL predictions • Drift detection • Retraining history")

# ── Data source toggle ─────────────────────────────────────────────
# Two modes: API (live) or Local (from files).
# Local mode lets the dashboard work without the API running —
# important for Hugging Face Spaces deployment and recruiter demos.
data_source = st.sidebar.radio(
    "Data source",
    ["Local (from files)", "API (live)"],
    help="Local mode works without the API server running.",
)

API_URL = None
if data_source == "API (live)":
    API_URL = st.sidebar.text_input("API endpoint", "http://localhost:8000")


def load_fleet_data():
    """
    Load fleet predictions from API or local files.

    Returns a DataFrame with columns:
    unit, rul_pred, lower, upper
    """
    if API_URL:
        import requests
        try:
            resp = requests.get(f"{API_URL}/predict/fleet", timeout=10)
            resp.raise_for_status()
            fleet = resp.json()
            return pd.DataFrame({
                "unit": fleet["units"],
                "rul_pred": fleet["rul_predictions"],
                "lower": fleet["confidence_lower"],
                "upper": fleet["confidence_upper"],
            }), fleet.get("kpis", {})
        except Exception as e:
            st.warning(f"API error: {e}. Falling back to local data.")

    # Local mode: load from processed data and run model
    import joblib
    import torch

    from pdm.config import DEVICE, FEATURE_SENSORS, PROC
    from pdm.models.lstm import RULLSTM

    data_path = PROC / "FD001.npz"
    if not data_path.exists():
        st.error("No processed data found. Run `python -m pdm.data.build_dataset` first.")
        return None, None

    d = np.load(data_path)
    Xte = torch.tensor(d["Xte"], dtype=torch.float32).to(DEVICE)

    # Load model
    model = RULLSTM(len(FEATURE_SENSORS)).to(DEVICE)
    model_path = PROC / "lstm.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        preds = model(Xte).cpu().numpy()

    # Conformal intervals
    conformal_path = PROC / "conformal.joblib"
    if conformal_path.exists():
        cp = joblib.load(conformal_path)
        intervals = cp.predict(preds)
        lower = intervals["lower"]
        upper = intervals["upper"]
    else:
        lower = np.maximum(preds - 20, 0)
        upper = preds + 20

    n = len(preds)
    preds_arr = np.array(preds)
    critical = int((preds_arr < 30).sum())
    warning = int(((preds_arr >= 30) & (preds_arr < 80)).sum())
    healthy = int((preds_arr >= 80).sum())

    kpis = {
        "total_units": n,
        "critical": critical,
        "warning": warning,
        "healthy": healthy,
        "avg_rul": float(preds_arr.mean()),
        "min_rul": float(preds_arr.min()),
    }

    df = pd.DataFrame({
        "unit": list(range(1, n + 1)),
        "rul_pred": preds.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
    })

    return df, kpis


# ── Load data ──────────────────────────────────────────────────────
df_fleet, kpis = load_fleet_data()

# ── Tabs ───────────────────────────────────────────────────────────
tab_fleet, tab_model, tab_drift, tab_retrain = st.tabs([
    "🏭 Fleet Health", "📊 Model Performance",
    "📈 Drift Monitor", "🔄 Retrain History"
])

# ── Tab 1: Fleet Health KPIs ──────────────────────────────────────
with tab_fleet:
    if df_fleet is not None and kpis:
        # KPI cards in a row
        # Why 4 columns? Each KPI gets equal visual weight.
        # Streamlit's st.metric renders as a card with big number + label.
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔴 Critical (RUL < 30)", kpis.get("critical", 0))
        col2.metric("🟡 Warning (30-80)", kpis.get("warning", 0))
        col3.metric("🟢 Healthy (80+)", kpis.get("healthy", 0))
        col4.metric("📊 Avg Fleet RUL", f"{kpis.get('avg_rul', 0):.0f} cycles")

        st.divider()

        # Fleet RUL distribution histogram
        # Why histogram? Shows the shape of fleet health at a glance.
        # The vertical red line marks the critical threshold.
        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            st.subheader("Fleet RUL Distribution")
            fig = px.histogram(
                df_fleet, x="rul_pred", nbins=20,
                labels={"rul_pred": "Predicted RUL (cycles)", "count": "Number of Units"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig.add_vline(x=30, line_dash="dash", line_color="red",
                          annotation_text="Critical threshold")
            fig.add_vline(x=80, line_dash="dash", line_color="orange",
                          annotation_text="Warning threshold")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Priority table: units closest to failure
        # Why show top 10? Maintenance managers need actionable priorities,
        # not a full dump of 100 units.
        with col_table:
            st.subheader("🚨 Maintenance Priority Queue")
            df_priority = df_fleet.nsmallest(10, "rul_pred").copy()
            df_priority["status"] = df_priority["rul_pred"].apply(
                lambda x: "🔴 Critical" if x < 15
                else "🟠 Warning" if x < 30
                else "🟡 Monitor"
            )
            st.dataframe(
                df_priority[["unit", "rul_pred", "lower", "upper", "status"]]
                .rename(columns={
                    "unit": "Unit ID",
                    "rul_pred": "Predicted RUL",
                    "lower": "Lower (90%)",
                    "upper": "Upper (90%)",
                    "status": "Status",
                }),
                use_container_width=True, hide_index=True,
            )

        # Scatter plot: prediction vs confidence interval
        st.subheader("Unit-Level Predictions with Confidence Intervals")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_fleet["unit"], y=df_fleet["rul_pred"],
            mode="markers", name="Prediction",
            marker=dict(size=6, color="#1f77b4"),
        ))
        fig2.add_trace(go.Scatter(
            x=df_fleet["unit"], y=df_fleet["upper"],
            mode="lines", name="Upper bound (90%)",
            line=dict(width=0), showlegend=False,
        ))
        fig2.add_trace(go.Scatter(
            x=df_fleet["unit"], y=df_fleet["lower"],
            mode="lines", name="90% Interval",
            fill="tonexty", fillcolor="rgba(31,119,180,0.15)",
            line=dict(width=0),
        ))
        fig2.add_hline(y=30, line_dash="dash", line_color="red",
                       annotation_text="Critical")
        fig2.update_layout(
            xaxis_title="Unit ID", yaxis_title="RUL (cycles)",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("No fleet data available. Check data source settings.")

# ── Tab 2: Model Performance ─────────────────────────────────────
with tab_model:
    st.subheader("Architecture Benchmark")

    # Try to load from MLflow
    try:
        from pdm.serving.registry import get_all_model_metrics
        metrics = get_all_model_metrics()
        if metrics:
            df_models = pd.DataFrame(metrics)[["model", "rmse", "mae", "phm_score"]]
            df_models.columns = ["Model", "RMSE", "MAE", "PHM Score"]
            df_models = df_models.round(2)
            st.dataframe(df_models, use_container_width=True, hide_index=True)

            # Bar chart comparison
            fig = px.bar(
                df_models, x="Model", y="RMSE",
                color="Model", title="RMSE Comparison",
                color_discrete_sequence=["#2ecc71", "#3498db", "#e74c3c"],
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model metrics found in MLflow.")
    except Exception as e:
        st.warning(f"Could not load MLflow metrics: {e}")
        # Fallback: manual entry
        st.info("Run model training scripts to populate this table.")

    # Calibration metrics
    st.subheader("Uncertainty Calibration")
    conformal_path = Path("data/processed/conformal.joblib")
    if conformal_path.exists():
        import joblib
        cp = joblib.load(conformal_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Target Coverage", "90%")
        col2.metric("Conformal Quantile (q̂)", f"{cp.q_hat:.1f} cycles")
        col3.metric("Interval Width", f"±{cp.q_hat:.1f} cycles")
    else:
        st.info("Run `python -m pdm.models.calibrate` to generate calibration data.")

# ── Tab 3: Drift Monitor ─────────────────────────────────────────
with tab_drift:
    st.subheader("Data Drift Detection")

    # Show latest drift report if available
    report_path = Path("data/processed/drift_report.html")
    if report_path.exists():
        # Show summary from retrain log
        log_path = Path("data/processed/retrain_log.jsonl")
        if log_path.exists():
            entries = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
            drift_entries = [e for e in entries if "drift" in e]

            if drift_entries:
                latest = drift_entries[-1]["drift"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Dataset Drift", "⚠ YES" if latest["dataset_drift"] else "✅ No")
                col2.metric("Features Drifted",
                            f"{latest['n_drifted_features']}/{latest['total_features']}")
                col3.metric("Drift Ratio", f"{latest['drift_ratio']:.1%}")

        st.subheader("Latest Evidently Report")
        st.components.v1.html(
            report_path.read_text(), height=600, scrolling=True
        )
    else:
        st.info(
            "No drift report yet. Run `python -m pdm.monitoring.drift` "
            "to generate one."
        )

# ── Tab 4: Retrain History ────────────────────────────────────────
with tab_retrain:
    st.subheader("Automated Retraining History")

    log_path = Path("data/processed/retrain_log.jsonl")
    if log_path.exists():
        entries = [json.loads(line) for line in log_path.read_text().strip().split("\n")]

        if entries:
            # Summary metrics
            total_runs = len(entries)
            promotions = sum(1 for e in entries if e.get("decision") == "promote")
            keeps = sum(1 for e in entries if e.get("decision") == "keep_champion")
            skips = sum(1 for e in entries if e.get("action") == "no_retrain")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Pipeline Runs", total_runs)
            col2.metric("Promotions", promotions)
            col3.metric("Kept Champion", keeps)
            col4.metric("Skipped (no drift)", skips)

            st.divider()

            # Expandable history cards (most recent first)
            for entry in reversed(entries[-10:]):
                decision = entry.get("decision", entry.get("action", "unknown"))
                timestamp = entry.get("timestamp", "N/A")

                # Color-code the decision
                if decision == "promote":
                    icon = "🟢"
                elif decision == "keep_champion":
                    icon = "🟡"
                else:
                    icon = "⚪"

                with st.expander(f"{icon} {timestamp} — {decision.upper()}"):
                    st.write(f"**Reason:** {entry.get('reason', 'N/A')}")

                    if entry.get("champion"):
                        st.write(f"**Champion RMSE:** {entry['champion'].get('rmse', 'N/A')}")
                    if entry.get("challenger"):
                        st.write(f"**Challenger RMSE:** {entry['challenger'].get('rmse', 'N/A')}")
                    if entry.get("drift"):
                        drift = entry["drift"]
                        st.write(
                            f"**Drift:** {drift.get('n_drifted_features', '?')}/"
                            f"{drift.get('total_features', '?')} features "
                            f"({drift.get('drift_ratio', 0):.1%})"
                        )
    else:
        st.info(
            "No retraining events yet. "
            "Run `python -m pdm.pipeline.retrain --force` to trigger one."
        )

# ── Sidebar info ───────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "End-to-end predictive maintenance system. "
    "LSTM + TCN models with calibrated conformal prediction intervals, "
    "Evidently drift monitoring, automated retraining, "
    "deployed on Kubernetes."
)
st.sidebar.markdown(
    "[GitHub](https://github.com/hilary-anish/predictive-maintenance-rul)"
)
