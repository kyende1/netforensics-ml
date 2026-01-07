"""
My project dashboard for encrypted traffic classification.
Made for my IUBH computer science project.
"""

import json
import subprocess
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc

# Basic setup
st.set_page_config(page_title="Traffic Forensics Dashboard", layout="wide")

# My local path - adjust if needed
ROOT = Path(r"C:\Users\OCHA\Desktop\IUBH\10. Project - Computer Science Project\netforensics")

# Where to find everything
DATASETS = {
    "CICIDS2017": {
        "data": ROOT / "data_cicids2017",
        "models": ROOT / "models_cicids2017",
        "reports": ROOT / "reports_cicids2017",
    },
    "ISCXVPN2016": {
        "data": ROOT / "data_iscx2016",
        "models": ROOT / "models_iscx2016",
        "reports": ROOT / "reports_iscx2016",
    },
}

# Helper functions
def shorten_path(p):
    """Don't show the whole long path in UI"""
    parts = str(p).split("\\")
    if len(parts) > 4:
        return f"{parts[0]}\\...\\{parts[-2]}\\{parts[-1]}"
    return str(p)

def load_test_files(data_dir):
    """Load the test data that was saved during training"""
    try:
        X = pd.read_csv(data_dir / "x_test.csv")
        y_df = pd.read_csv(data_dir / "y_test.csv")
        y = y_df["label"].values if "label" in y_df.columns else y_df.iloc[:, 0].values
        return X, y.astype(int)
    except Exception as e:
        st.error(f"Can't load test data: {e}")
        return None, None

def load_my_models(models_dir):
    """Load trained models from disk"""
    models = {}
    scaler = None
    
    # Try to load scaler first
    scaler_path = models_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    
    # Load RandomForest if exists
    rf_path = models_dir / "rf.joblib"
    if rf_path.exists():
        models["Random Forest"] = joblib.load(rf_path)
    
    # Load SGD if exists
    sgd_path = models_dir / "sgd.joblib"
    if sgd_path.exists():
        try:
            models["SGD Classifier"] = joblib.load(sgd_path)
        except:
            pass
    
    # Try XGBoost - installed it for comparison
    xgb_path = models_dir / "xgb.joblib"
    if xgb_path.exists():
        try:
            models["XGBoost"] = joblib.load(xgb_path)
        except:
            st.warning("XGBoost model exists but can't load it")
    
    return models, scaler

def make_confusion_matrix_plot(cm, title, normalize=False):
    """Create confusion matrix visualization"""
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        z = np.round(cm_norm, 2)
        colors = "Greens"
    else:
        z = cm
        colors = "Blues"
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Benign", "Attack"],
        y=["Benign", "Attack"],
        text=z,
        texttemplate="%{text}",
        colorscale=colors
    ))
    fig.update_layout(title=title, height=350)
    return fig

def main():
    st.title("Traffic Classification Dashboard")
    st.write("For my IUBH project - classifying encrypted traffic using metadata")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    dataset = st.sidebar.selectbox("Choose dataset", list(DATASETS.keys()))
    cfg = DATASETS[dataset]
    
    # Threshold slider
    threshold = st.sidebar.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Buttons to run training
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Train CICIDS"):
        with st.sidebar:
            st.info("Training CICIDS...")
            result = subprocess.run([sys.executable, "train_cicids.py", "--fast"], 
                                   cwd=str(ROOT), capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Done!")
            else:
                st.error("Training failed")
        st.rerun()
    
    if col2.button("Train Both"):
        with st.sidebar:
            st.info("Training both datasets...")
            result = subprocess.run([sys.executable, "train_cicids.py", "--fast", "--train-iscx"],
                                   cwd=str(ROOT), capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Done!")
            else:
                st.error("Training failed")
        st.rerun()
    
    # Show paths
    st.sidebar.write("---")
    hide_paths = st.sidebar.checkbox("Hide long paths", True)
    
    if hide_paths:
        st.info(f"**Data:** {shorten_path(cfg['data'])}")
        st.info(f"**Models:** {shorten_path(cfg['models'])}")
        st.info(f"**Reports:** {shorten_path(cfg['reports'])}")
    
    # Load data and models
    X_test, y_test = load_test_files(cfg["data"])
    if X_test is None:
        st.error("No test data found. Please train models first.")
        return
    
    models, scaler = load_my_models(cfg["models"])
    if not models:
        st.warning("No trained models found. Click 'Train CICIDS' first.")
        return
    
    # Tabs
    tabs = st.tabs(["Overview", "Test Results", "Feature Importance", "Predict", "Simulation", "Reports"])
    
    with tabs[0]:
        st.write(f"### Dataset: {dataset}")
        st.write("""
        This dashboard shows results from my machine learning models for encrypted traffic classification.
        
        **Models available:**
        - Random Forest
        - SGD Classifier (like linear SVM)
        - XGBoost (added for comparison)
        
        The models only use flow metadata (no packet contents).
        """)
    
    with tabs[1]:
        st.write("### Model Performance on Test Set")
        model_choice = st.selectbox("Select model", list(models.keys()))
        model = models[model_choice]
        
        # Make predictions
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs >= threshold).astype(int)
        else:
            probs = None
            preds = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        
        if probs is not None:
            try:
                auc_score = roc_auc_score(y_test, probs)
            except:
                auc_score = None
        else:
            auc_score = None
        
        cm = confusion_matrix(y_test, preds)
        
        # Show metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")
        m4.metric("F1 Score", f"{f1:.3f}")
        m5.metric("AUC", f"{auc_score:.3f}" if auc_score else "N/A")
        
        # Show confusion matrices
        col1, col2 = st.columns(2)
        col1.plotly_chart(make_confusion_matrix_plot(cm, "Counts"), use_container_width=True)
        col2.plotly_chart(make_confusion_matrix_plot(cm, "Normalized", normalize=True), use_container_width=True)
        
        # ROC curve if we have probabilities
        if probs is not None and auc_score is not None:
            fpr, tpr, _ = roc_curve(y_test, probs)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {auc_score:.3f}'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.write("### Feature Importance")
        # Only tree models have feature importances
        tree_models = [name for name in models.keys() if name in ["Random Forest", "XGBoost"]]
        if tree_models:
            model_for_features = st.selectbox("Choose model for features", tree_models)
            model = models[model_for_features]
            
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                top_n = st.slider("Show top N features", 10, 30, 15)
                
                # Create dataframe and sort
                feat_df = pd.DataFrame({
                    'feature': X_test.columns,
                    'importance': importances
                }).sort_values('importance', ascending=True).tail(top_n)
                
                # Plot
                fig = px.bar(feat_df, x='importance', y='feature', orientation='h', 
                           title=f'Top {top_n} Important Features')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("This model doesn't have feature importances")
        else:
            st.info("No tree-based models loaded")
    
    with tabs[3]:
        st.write("### Predict on New Data")
        uploaded = st.file_uploader("Upload CSV file with traffic features", type=["csv"])
        
        if uploaded:
            try:
                # Read the file
                df = pd.read_csv(uploaded)
                
                # Remove columns we don't need for prediction
                cols_to_drop = []
                for col in ["Flow ID", "Src IP", "Dst IP", "Timestamp"]:
                    if col in df.columns:
                        cols_to_drop.append(col)
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                
                # Get training columns to match format
                try:
                    train_cols = pd.read_csv(cfg["data"] / "x_train.csv", nrows=0).columns.tolist()
                except:
                    st.error("Can't find training data format. Train models first.")
                    return
                
                # Make sure uploaded data has same columns
                for col in train_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                
                # Keep only columns in training order
                df = df[train_cols]
                
                # Scale if we have a scaler
                if scaler is not None:
                    df_scaled = pd.DataFrame(scaler.transform(df), columns=train_cols)
                else:
                    df_scaled = df
                
                # Choose model
                model_choice = st.selectbox("Model to use", list(models.keys()), key="predict_model")
                model = models[model_choice]
                
                # Make predictions
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(df_scaled)[:, 1]
                    preds = (probs >= threshold).astype(int)
                    results = pd.DataFrame({
                        'Probability': probs,
                        'Prediction': preds,
                        'Label': ['Benign' if p == 0 else 'Attack' for p in preds]
                    })
                else:
                    preds = model.predict(df_scaled)
                    results = pd.DataFrame({
                        'Prediction': preds,
                        'Label': ['Benign' if p == 0 else 'Attack' for p in preds]
                    })
                
                st.success(f"Made {len(results)} predictions")
                st.dataframe(results.head(50))
                
                # Download button
                csv = results.to_csv(index=False)
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
                
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
    
    with tabs[4]:
        st.write("### Traffic Simulation")
        st.write("Simulate a stream of traffic and generate alerts")
        
        # Choose data source
        use_test_data = st.radio("Use:", ["Test set data", "Upload my own"], horizontal=True)
        
        if use_test_data == "Test set data":
            sim_data = X_test.copy()
            sim_labels = y_test
        else:
            sim_file = st.file_uploader("Upload simulation data", type=["csv"], key="sim_upload")
            if sim_file:
                try:
                    sim_df = pd.read_csv(sim_file)
                    # Match training format
                    train_cols = pd.read_csv(cfg["data"] / "x_train.csv", nrows=0).columns.tolist()
                    for col in ["Flow ID", "Src IP", "Dst IP", "Timestamp"]:
                        if col in sim_df.columns:
                            sim_df = sim_df.drop(columns=col)
                    
                    for col in train_cols:
                        if col not in sim_df.columns:
                            sim_df[col] = 0.0
                    
                    sim_data = sim_df[train_cols]
                    if scaler:
                        sim_data = pd.DataFrame(scaler.transform(sim_data), columns=train_cols)
                    sim_labels = None
                except Exception as e:
                    st.error(f"Error loading simulation data: {e}")
                    sim_data = None
            else:
                sim_data = None
        
        if sim_data is not None:
            # Simulation settings
            n_flows = st.slider("Number of flows", 10, min(1000, len(sim_data)), 100)
            flow_rate = st.slider("Flows per second", 1, 50, 10)
            sim_model = st.selectbox("Model for simulation", list(models.keys()), key="sim_model")
            
            if st.button("Run Simulation"):
                # Get subset
                subset = sim_data.iloc[:n_flows]
                model = models[sim_model]
                
                # Predict
                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(subset)[:, 1]
                    preds = (scores >= threshold).astype(int)
                else:
                    scores = None
                    preds = model.predict(subset)
                
                # Create results
                results = pd.DataFrame({
                    'flow_index': range(n_flows),
                    'prediction': preds
                })
                
                if scores is not None:
                    results['score'] = scores
                
                if sim_labels is not None:
                    results['true_label'] = sim_labels[:n_flows]
                
                # Find alerts
                alerts = results[results['prediction'] == 1]
                
                # Show summary
                st.write("### Simulation Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Flows", n_flows)
                col2.metric("Alerts", len(alerts))
                col3.metric("Duration", f"{n_flows/flow_rate:.1f} seconds")
                
                # Show some data
                st.write("First 20 flows:")
                st.dataframe(results.head(20))
                
                # Export alerts if any
                if not alerts.empty:
                    st.write("### Alerts for SIEM")
                    alerts_export = alerts.copy()
                    alerts_export['timestamp'] = pd.Timestamp.now().isoformat()
                    alerts_export['model'] = sim_model
                    alerts_export['threshold'] = threshold
                    
                    # Create JSONL
                    json_lines = []
                    for _, row in alerts_export.iterrows():
                        json_lines.append(json.dumps(row.to_dict(), default=str))
                    
                    st.download_button("Download Alerts as JSONL", 
                                      "\n".join(json_lines),
                                      "alerts.jsonl",
                                      "application/json")
                else:
                    st.info("No alerts at current threshold")
    
    with tabs[5]:
        st.write("### Training Reports")
        
        # Try to load report files
        rf_report = cfg["reports"] / "rf_metrics.txt"
        sgd_report = cfg["reports"] / "sgd_metrics.txt"
        xgb_report = cfg["reports"] / "xgb_metrics.txt"
        
        col1, col2, col3 = st.columns(3)
        
        if rf_report.exists():
            with open(rf_report, 'r', encoding='utf-8') as f:
                rf_text = f.read()
            col1.text_area("Random Forest", rf_text, height=300)
            col1.download_button("Download", rf_text, "rf_report.txt")
        else:
            col1.info("No RF report yet")
        
        if sgd_report.exists():
            with open(sgd_report, 'r', encoding='utf-8') as f:
                sgd_text = f.read()
            col2.text_area("SGD Classifier", sgd_text, height=300)
            col2.download_button("Download", sgd_text, "sgd_report.txt")
        else:
            col2.info("No SGD report yet")
        
        if xgb_report.exists():
            with open(xgb_report, 'r', encoding='utf-8') as f:
                xgb_text = f.read()
            col3.text_area("XGBoost", xgb_text, height=300)
            col3.download_button("Download", xgb_text, "xgb_report.txt")
        else:
            col3.info("No XGBoost report yet")

if __name__ == "__main__":
    main()