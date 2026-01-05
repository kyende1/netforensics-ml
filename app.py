# app.py — Encrypted Traffic Forensics — ML Dashboard
# Compatible with your train_cicids.py outputs (CICIDS2017 + ISCX2016)

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, auc
)

# -------------------------- CONFIG --------------------------
PAGE_TITLE = "🔎 Encrypted Traffic Forensics — ML Dashboard"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

ROOT = Path(r"C:\Users\OCHA\Desktop\IUBH\10. Project - Computer Science Project\netforensics")

DATASETS = {
    "CICIDS2017": {
        "data_dir": ROOT / "data_cicids2017",
        "models_dir": ROOT / "models_cicids2017",
        "reports_dir": ROOT / "reports_cicids2017",
        "rf_metrics": "rf_cicids_metrics.txt",
        "sgd_metrics": "sgd_cicids_metrics.txt",
        "xgb_metrics": "xgb_cicids_metrics.txt",   # NEW: XGBoost report
        "rf_model": "rf.joblib",
        "sgd_model": "sgd.joblib",
        "xgb_model": "xgb.joblib",                  # NEW: XGBoost model
        "scaler": "scaler.joblib",
    },
    "ISCXVPN2016": {
        "data_dir": ROOT / "data_iscx2016",
        "models_dir": ROOT / "models_iscx2016",
        "reports_dir": ROOT / "reports_iscx2016",
        "rf_metrics": "rf_iscx_metrics.txt",
        "sgd_metrics": "sgd_iscx_metrics.txt",
        "xgb_metrics": "xgb_iscx_metrics.txt",     # NEW
        "rf_model": "rf.joblib",
        "sgd_model": "sgd.joblib",
        "xgb_model": "xgb.joblib",                 # NEW
        "scaler": "scaler.joblib",
    },
}

# Light CSS polish
st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 1.4rem; }
code, pre { font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------- HELPERS & CACHE -------------------------------
def _short_path(p: Path) -> str:
    """Obfuscate absolute paths for display if user chooses to hide paths."""
    parts = p.parts
    return f"{parts[0]}\\…\\{parts[-2]}\\{parts[-1]}" if len(parts) > 3 else str(p)

@st.cache_data(show_spinner=False)
def load_splits(data_dir: Path):
    """Load held-out test split produced by train_cicids.py."""
    Xte_p = data_dir / "x_test.csv"
    yte_p = data_dir / "y_test.csv"
    if not Xte_p.exists() or not yte_p.exists():
        raise FileNotFoundError(f"Missing test split in {data_dir} (need x_test.csv and y_test.csv).")
    Xte = pd.read_csv(Xte_p)
    yt = pd.read_csv(yte_p)
    y_col = "label" if "label" in yt.columns else yt.columns[0]
    yte = yt[y_col].astype(int).values
    return Xte, yte

@st.cache_resource(show_spinner=False)
def load_models(models_dir: Path, cfg: dict):
    """Load trained models + optional scaler."""
    models = {}
    rf_path = models_dir / cfg["rf_model"]
    sgd_path = models_dir / cfg["sgd_model"]
    xgb_path = models_dir / cfg.get("xgb_model", "xgb.joblib")   # NEW
    scaler_path = models_dir / cfg["scaler"]

    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    if rf_path.exists():
        models["RandomForest"] = joblib.load(rf_path)

    if sgd_path.exists():
        try:
            models["SGDClassifier"] = joblib.load(sgd_path)
        except Exception:
            pass

    # NEW: XGBoost model
    if xgb_path.exists():
        try:
            models["XGBoost"] = joblib.load(xgb_path)
        except Exception:
            pass

    return models, scaler

def metrics_from_preds(y_true, y_pred, proba=None):
    """Compute standard metrics and confusion matrix with fixed labels [0,1]."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc_v = roc_auc_score(y_true, proba) if proba is not None else None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return acc, p, r, f1, auc_v, cm

def plotly_confusion_matrix(cm, title="Confusion Matrix", normalize=False):
    """Interactive confusion matrix (counts or row-normalized)."""
    if normalize:
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1
        cm = cm / denom
        ztext = np.round(cm, 2).astype(str)
        colors = px.colors.sequential.Teal
    else:
        ztext = cm.astype(str)
        colors = px.colors.sequential.Blues
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred 0", "Pred 1"],
            y=["True 0", "True 1"],
            text=ztext,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale=colors,
            showscale=True,
        )
    )
    fig.update_layout(title=title, margin=dict(l=40, r=20, t=60, b=40), height=420)
    return fig

def plotly_roc(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}"))
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash"))
    )
    fig.update_layout(
        title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=420, margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig

def plotly_pr(y_true, proba):
    prec, rec, _ = precision_recall_curve(y_true, proba)
    pr_auc = auc(rec, prec)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"AP = {pr_auc:.3f}"))
    fig.update_layout(
        title="Precision–Recall Curve", xaxis_title="Recall", yaxis_title="Precision",
        height=420, margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig

def plotly_feature_importance(model, columns: List[str], top_k=20):
    """Top-K feature importances for RandomForest-like models."""
    if not hasattr(model, "feature_importances_"):
        return None
    imp = model.feature_importances_
    df = (
        pd.DataFrame({"feature": columns, "importance": imp})
        .sort_values("importance", ascending=True)
        .tail(top_k)
    )
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {top_k} Feature Importances",
        height=520,
        color="importance",
        color_continuous_scale="viridis",
    )
    fig.update_layout(margin=dict(l=80, r=20, t=60, b=40))
    return fig

def align_to_training_features(df: pd.DataFrame, train_cols: List[str]) -> pd.DataFrame:
    """
    Align an uploaded CSV to the training feature set:
    - Keep only numeric columns.
    - Add any missing training columns with zeros.
    - Order columns exactly as during training.
    """
    df = df.select_dtypes(include=[np.number]).copy()
    for c in train_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[train_cols]

# ------------------------------ MAIN APP ------------------------------
def main():
    # Header/logo
    from PIL import Image
    logo_path = ROOT / "assets" / "logo.png"
    if logo_path.exists():
        c1, c2 = st.columns([1, 8])
        c1.image(str(logo_path), use_container_width=True)
        c2.title("Encrypted Traffic Forensics — ML Dashboard")
    else:
        st.title(PAGE_TITLE)
    st.caption("Metadata-only ML for encrypted traffic. Evaluate, explain, predict.")

    # Sidebar controls
    st.sidebar.header("Controls")
    ds_name = st.sidebar.selectbox("Experiment", list(DATASETS.keys()))
    cfg = DATASETS[ds_name]
    thr = st.sidebar.slider("Decision threshold", 0.00, 1.00, 0.50, 0.01)
    hide_paths = st.sidebar.toggle("Hide absolute paths in UI", value=True)

    col_sb1, col_sb2, col_sb3 = st.sidebar.columns(3)
    do_train_cic = col_sb1.button("🚀 Train CICIDS (fast)")
    do_train_both = col_sb2.button("🚀 Train CIC+ISCX")
    do_refresh = col_sb3.button("🔁 Refresh")

    if do_train_cic or do_train_both:
        st.sidebar.info("Starting training…")
        args = ["--fast"]
        if do_train_both:
            args.append("--train-iscx")
        proc = subprocess.run(
            [sys.executable, "train_cicids.py", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        st.session_state["last_train_stdout"] = proc.stdout[-5000:]
        st.session_state["last_train_stderr"] = proc.stderr[-5000:]
        st.session_state["last_train_rc"] = proc.returncode
        st.rerun()

    if do_refresh:
        st.rerun()

    # Paths summary (masked if chosen)
    data_dir, models_dir, reports_dir = cfg["data_dir"], cfg["models_dir"], cfg["reports_dir"]
    show_data = _short_path(data_dir) if hide_paths else str(data_dir)
    show_models = _short_path(models_dir) if hide_paths else str(models_dir)
    show_reports = _short_path(reports_dir) if hide_paths else str(reports_dir)

    cA, cB, cC = st.columns(3)
    cA.info(f"**Data:** {show_data}")
    cB.info(f"**Models:** {show_models}")
    cC.info(f"**Reports:** {show_reports}")

    # Load artifacts
    try:
        Xte, yte = load_splits(data_dir)
    except Exception as e:
        st.error(f"Could not load test split: {e}")
        st.stop()

    models, scaler = load_models(models_dir, cfg)
    if not models:
        st.warning("No trained models found. Train first, then click Refresh.")
        st.stop()

    # Tabs (NEW: Live Simulation)
    tab_overview, tab_eval, tab_explain, tab_predict, tab_live, tab_reports = st.tabs(
        ["Overview", "Evaluate", "Explain", "Predict", "Live Simulation", "Reports"]
    )

    # ---------------- Overview ----------------
    with tab_overview:
        st.subheader(f"Experiment: {ds_name}")
        st.write(
            """
**Goal:** Metadata-only classification of encrypted traffic (no payload decryption).  
**Models:** RandomForest, a linear SVM surrogate via SGDClassifier, and XGBoost (Phase 3).  
**Data:** CICIDS2017; optional second dataset ISCX VPN/NonVPN (2016).
            """
        )
        if "last_train_rc" in st.session_state:
            st.markdown("### Last training run")
            st.write("Status:", "✅ Success" if st.session_state["last_train_rc"] == 0 else "❌ Error")
            with st.expander("Show stdout"):
                st.code(st.session_state.get("last_train_stdout", "") or "(empty)")
            if st.session_state.get("last_train_stderr"):
                with st.expander("Show stderr"):
                    st.code(st.session_state["last_train_stderr"])

        # Show cross-eval report if available (CICIDS page only)
        if ds_name == "CICIDS2017":
            cross_p = (ROOT / "reports_cicids2017" / "rf_cicids_on_iscx_metrics.txt")
            if cross_p.exists():
                st.markdown("### Cross-evaluation (CICIDS RF → ISCX test)")
                st.code(cross_p.read_text(encoding="utf-8", errors="ignore"))

    # ---------------- Evaluate ----------------
    with tab_eval:
        st.subheader("Model evaluation on held-out test set")
        model_choice = st.radio("Choose model", list(models.keys()), horizontal=True)
        model = models[model_choice]

        # Predict with optional threshold for probabilistic models
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xte)[:, 1]
            yhat = (proba >= thr).astype(int)
        else:
            proba = None
            yhat = model.predict(Xte)

        acc, p, r, f1, auc_v, cm = metrics_from_preds(yte, yhat, proba)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Precision", f"{p:.3f}")
        m3.metric("Recall", f"{r:.3f}")
        m4.metric("F1", f"{f1:.3f}")
        m5.metric("ROC-AUC", "n/a" if auc_v is None else f"{auc_v:.3f}")

        c1, c2 = st.columns(2)
        c1.plotly_chart(
            plotly_confusion_matrix(cm, "Confusion Matrix (Counts)"),
            use_container_width=True,
        )
        c2.plotly_chart(
            plotly_confusion_matrix(cm, "Confusion Matrix (Normalized)", normalize=True),
            use_container_width=True,
        )

        if proba is not None:
            r1, r2 = st.columns(2)
            r1.plotly_chart(plotly_roc(yte, proba), use_container_width=True)
            r2.plotly_chart(plotly_pr(yte, proba), use_container_width=True)

    # ---------------- Explain ----------------
    with tab_explain:
        st.subheader("Explainability")
        topk = st.slider("Top features to show", 5, 40, 20, 1)
        # Prefer RF or any model with feature_importances_
        expl_model = None
        if "RandomForest" in models:
            expl_model = models["RandomForest"]
        elif "XGBoost" in models:
            expl_model = models["XGBoost"]
        if expl_model:
            fig_imp = plotly_feature_importance(expl_model, Xte.columns.tolist(), top_k=topk)
            if fig_imp:
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("No feature_importances_ available.")
        else:
            st.info("No tree-based model loaded for feature importance.")

    # ---------------- Predict (batch CSV) ----------------
    with tab_predict:
        st.subheader("Try your own CSV (same schema as training)")
        up = st.file_uploader("Upload CSV of flows", type=["csv"])
        if up:
            try:
                df = pd.read_csv(up)
                df.columns = [c.strip() for c in df.columns]

                # Load training feature order from x_train.csv
                train_cols_path = cfg["data_dir"] / "x_train.csv"
                if not train_cols_path.exists():
                    st.error("x_train.csv not found — please (re)train first.")
                else:
                    train_cols = pd.read_csv(train_cols_path, nrows=1).columns.tolist()
                    # Remove extra ID-like columns if present
                    for c in ("Flow ID", "Src IP", "Dst IP", "Timestamp"):
                        if c in df.columns:
                            df.drop(columns=c, inplace=True)
                    # Align and scale
                    num = align_to_training_features(df, train_cols)
                    if scaler is not None:
                        num = pd.DataFrame(scaler.transform(num), columns=num.columns)

                    mdl_name = st.selectbox("Model to use", list(models.keys()))
                    mdl = models[mdl_name]
                    if hasattr(mdl, "predict_proba"):
                        proba_u = mdl.predict_proba(num)[:, 1]
                        preds_u = (proba_u >= thr).astype(int)
                        out = pd.DataFrame({"proba": proba_u, "pred": preds_u})
                    else:
                        preds_u = mdl.predict(num)
                        out = pd.DataFrame({"pred": preds_u})

                    st.success(f"Predicted {len(out)} rows")
                    st.dataframe(out.head(200), use_container_width=True)
                    st.download_button("Download predictions CSV", out.to_csv(index=False), "predictions.csv")
            except Exception as e:
                st.error(f"Could not score the uploaded file: {e}")

    # ---------------- Live Simulation (Phase 3) ----------------
    with tab_live:
        st.subheader("Live Traffic Simulation & SIEM Export")

        sim_source = st.radio(
            "Simulation data source",
            ["Use current test set", "Upload custom CSV"],
            horizontal=True,
        )

        # Prepare simulation data
        sim_df = None
        sim_y = None

        if sim_source == "Use current test set":
            sim_df = Xte.copy()
            sim_y = yte
        else:
            up_live = st.file_uploader("Upload CSV for simulation", type=["csv"], key="live_csv")
            if up_live:
                try:
                    df_live = pd.read_csv(up_live)
                    df_live.columns = [c.strip() for c in df_live.columns]
                    train_cols_path = cfg["data_dir"] / "x_train.csv"
                    if train_cols_path.exists():
                        train_cols = pd.read_csv(train_cols_path, nrows=1).columns.tolist()
                        for c in ("Flow ID", "Src IP", "Dst IP", "Timestamp"):
                            if c in df_live.columns:
                                df_live.drop(columns=c, inplace=True)
                        num_live = align_to_training_features(df_live, train_cols)
                        if scaler is not None:
                            num_live = pd.DataFrame(scaler.transform(num_live), columns=num_live.columns)
                        sim_df = num_live
                        sim_y = None  # no ground truth
                    else:
                        st.error("x_train.csv not found — please (re)train first.")
                except Exception as e:
                    st.error(f"Could not prepare simulation data: {e}")

        if sim_df is not None:
            max_flows = int(min(len(sim_df), 2000))
            n_flows = st.slider("Number of flows to simulate", 10, max_flows, min(100, max_flows), step=10)
            model_live_choice = st.selectbox("Model for simulation", list(models.keys()), key="live_model")

            # Simulated rate is conceptual (for discussion in report)
            rate = st.slider("Simulated rate (flows per second)", 1, 100, 10)

            if st.button("Run simulation"):
                mdl = models[model_live_choice]
                sim_subset = sim_df.iloc[:n_flows].copy()
                if hasattr(mdl, "predict_proba"):
                    scores = mdl.predict_proba(sim_subset)[:, 1]
                    preds = (scores >= thr).astype(int)
                else:
                    scores = None
                    preds = mdl.predict(sim_subset)

                sim_results = pd.DataFrame({
                    "flow_index": np.arange(len(sim_subset)),
                    "pred": preds,
                })
                if scores is not None:
                    sim_results["score"] = scores
                if sim_y is not None:
                    sim_results["true_label"] = sim_y[:len(sim_subset)]

                # Mark alerts (predicted malicious)
                sim_results["is_alert"] = sim_results["pred"] == 1

                st.markdown("### Simulation summary")
                total_alerts = int(sim_results["is_alert"].sum())
                colS1, colS2, colS3 = st.columns(3)
                colS1.metric("Simulated flows", len(sim_results))
                colS2.metric("Alerts (pred=1)", total_alerts)
                colS3.metric("Simulated duration (sec)", f"{len(sim_results) / rate:.1f}")

                st.markdown("### Sample of simulated stream")
                st.dataframe(sim_results.head(50), use_container_width=True)

                # ---------------- SIEM / IDS-style alert export ----------------# ---------------- SIEM / IDS-style alert export ----------------
                st.markdown("### Alert export (for SIEM / IDS integration)")
                alerts = sim_results[sim_results["is_alert"]].copy()
                if not alerts.empty:
                    # Build minimal JSONL-style alert stream
                    now_iso = pd.Timestamp.utcnow().isoformat()
                    alerts["timestamp"] = now_iso
                    alerts["dataset"] = ds_name
                    alerts["model"] = model_live_choice
                    alerts["threshold"] = thr

                    # Reorder columns to be SIEM-friendly
                    cols_order = [
                        "timestamp",
                        "dataset",
                        "model",
                        "threshold",
                        "flow_index",
                        "score",
                        "pred",
                        "true_label",
                    ]
                    # Keep only columns that actually exist
                    cols_order = [c for c in cols_order if c in alerts.columns]
                    alerts = alerts[cols_order]

                    # ✅ Convert each record dict to a JSON line
                    records = alerts.to_dict(orient="records")
                    jsonl_lines = [json.dumps(rec, default=str) for rec in records]
                    jsonl = "\n".join(jsonl_lines)

                    st.write("These alerts can be forwarded to a SIEM (e.g., Splunk / ELK) as JSON lines.")
                    st.dataframe(alerts.head(50), use_container_width=True)
                    st.download_button(
                        "Download alerts as JSONL",
                        jsonl,
                        file_name="alerts.jsonl",
                        mime="application/json",
                    )
                else:
                    st.info("No alerts (pred=1) in this simulation at the current threshold.")

    # ---------------- Reports ----------------
    with tab_reports:
        st.subheader("Saved reports")
        rf_metrics_p = reports_dir / cfg["rf_metrics"]
        sgd_metrics_p = reports_dir / cfg["sgd_metrics"]
        xgb_metrics_p = reports_dir / cfg.get("xgb_metrics", "xgb_metrics.txt")  # NEW

        colR1, colR2, colR3 = st.columns(3)

        if rf_metrics_p.exists():
            txt = rf_metrics_p.read_text(encoding="utf-8", errors="ignore")
            colR1.markdown("**RandomForest metrics**")
            colR1.code(txt)
            colR1.download_button("Download RF metrics", txt, file_name=rf_metrics_p.name)
        else:
            colR1.info("RF metrics not found yet.")

        if sgd_metrics_p.exists():
            txt2 = sgd_metrics_p.read_text(encoding="utf-8", errors="ignore")
            colR2.markdown("**SGDClassifier metrics**")
            colR2.code(txt2)
            colR2.download_button("Download SGD metrics", txt2, file_name=sgd_metrics_p.name)
        else:
            colR2.info("SGD metrics not found or SGD not trained.")

        if xgb_metrics_p.exists():
            txt3 = xgb_metrics_p.read_text(encoding="utf-8", errors="ignore")
            colR3.markdown("**XGBoost metrics (Phase 3)**")
            colR3.code(txt3)
            colR3.download_button("Download XGBoost metrics", txt3, file_name=xgb_metrics_p.name)
        else:
            colR3.info("XGBoost metrics not found (or xgboost not installed).")

# Run the app
try:
    main()
except Exception as e:
    st.exception(e)
