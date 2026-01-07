"""
Train models for my encrypted traffic project.
Reads CICIDS2017 and ISCXVPN2016, trains models, saves everything.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not installed, skipping that model")

# Setup paths for my project
ROOT = Path(r"C:\Users\OCHA\Desktop\IUBH\10. Project - Computer Science Project\netforensics")

# Input data locations
CICIDS_RAW = ROOT / "cicids2017_raw"
ISCX_RAW = ROOT / "iscx2016_raw"

# Output folders
MODELS_CIC = ROOT / "models_cicids2017"
REPORTS_CIC = ROOT / "reports_cicids2017"
DATA_CIC = ROOT / "data_cicids2017"

MODELS_ISCX = ROOT / "models_iscx2016"
REPORTS_ISCX = ROOT / "reports_iscx2016"
DATA_ISCX = ROOT / "data_iscx2016"

# Create folders if needed
for folder in [MODELS_CIC, REPORTS_CIC, DATA_CIC, MODELS_ISCX, REPORTS_ISCX, DATA_ISCX]:
    folder.mkdir(parents=True, exist_ok=True)

# Settings
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
SCALE_DATA = True

# Fast mode settings (for testing)
FAST_MODE = True
MAX_FILES = 3  # Only use first 3 CICIDS files in fast mode
CHUNK_READING = True
CHUNK_SIZE = 200000
DOWNSAMPLE = 0.3  # Use 30% of data in fast mode

# Columns to remove (metadata not features)
METADATA_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]

def find_label_column(df):
    """Figure out which column has the labels"""
    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in ["label", "class", "attack", "category", "traffictype"]:
            return col
    return None

def convert_to_binary_labels(series):
    """Convert string labels to binary (0=normal, 1=attack/vpn)"""
    labels = series.astype(str).str.strip().str.upper()
    
    # Handle CICIDS labels
    if any("BENIGN" in str(x) for x in labels.unique()[:10]):
        return (labels != "BENIGN").astype(int)
    
    # Handle ISCX VPN labels
    if any("VPN" in str(x) for x in labels.unique()[:10]):
        return labels.str.contains("VPN").astype(int)
    
    # If already 0/1 or similar
    unique_vals = labels.unique()
    if set(unique_vals).issubset({"0", "1", "0.0", "1.0"}):
        return labels.astype(int)
    
    # Default: assume 1 is positive class
    return (labels == "1").astype(int)

def load_cicids_data(data_folder):
    """Load CICIDS CSV files"""
    print(f"Loading CICIDS from {data_folder}")
    
    # Find CSV files
    csv_files = sorted(data_folder.glob("*.csv"))
    # Skip any merged/clean files
    csv_files = [f for f in csv_files if "clean" not in f.name.lower()]
    
    if FAST_MODE:
        csv_files = csv_files[:MAX_FILES]
    
    if not csv_files:
        print("No CSV files found!")
        return None
    
    all_data = []
    
    for csv_file in csv_files:
        print(f"  Reading {csv_file.name}")
        
        if CHUNK_READING:
            # Read in chunks to save memory
            for chunk in pd.read_csv(csv_file, chunksize=CHUNK_SIZE, low_memory=True):
                # Clean up column names
                chunk.columns = [c.strip() for c in chunk.columns]
                
                # Remove metadata columns
                for meta_col in METADATA_COLS:
                    if meta_col in chunk.columns:
                        chunk = chunk.drop(columns=[meta_col])
                
                # Find label column
                label_col = find_label_column(chunk)
                if label_col is None:
                    print(f"    Skipping chunk - no label found")
                    continue
                
                # Convert labels to binary
                labels = convert_to_binary_labels(chunk[label_col])
                
                # Remove label column if not named "label"
                if label_col != "label":
                    chunk = chunk.drop(columns=[label_col])
                
                # Keep only numeric columns
                numeric_data = chunk.select_dtypes(include=[np.number]).copy()
                if numeric_data.empty:
                    continue
                
                # Add labels and source info
                numeric_data["label"] = labels.values
                numeric_data["source_file"] = csv_file.name
                
                all_data.append(numeric_data)
        else:
            # Read entire file
            df = pd.read_csv(csv_file, low_memory=True)
            df.columns = [c.strip() for c in df.columns]
            
            # Remove metadata
            for meta_col in METADATA_COLS:
                if meta_col in df.columns:
                    df = df.drop(columns=[meta_col])
            
            # Find labels
            label_col = find_label_column(df)
            if label_col is None:
                print(f"  Skipping {csv_file.name} - no labels")
                continue
            
            labels = convert_to_binary_labels(df[label_col])
            if label_col != "label":
                df = df.drop(columns=[label_col])
            
            numeric_data = df.select_dtypes(include=[np.number]).copy()
            if numeric_data.empty:
                continue
            
            numeric_data["label"] = labels.values
            numeric_data["source_file"] = csv_file.name
            all_data.append(numeric_data)
    
    if not all_data:
        print("No data loaded!")
        return None
    
    # Combine all chunks
    combined = pd.concat(all_data, ignore_index=True)
    
    # Clean up
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna()
    
    # Downsample if in fast mode
    if FAST_MODE and DOWNSAMPLE < 1.0:
        combined = combined.groupby("label", group_keys=False).apply(
            lambda x: x.sample(frac=DOWNSAMPLE, random_state=RANDOM_SEED)
        )
    
    print(f"  Loaded {len(combined)} rows, {combined.shape[1]} columns")
    return combined

def load_iscx_data(csv_path):
    """Load ISCX VPN dataset"""
    print(f"Loading ISCX from {csv_path}")
    
    if not csv_path.exists():
        print("  File not found!")
        return None
    
    try:
        df = pd.read_csv(csv_path, low_memory=True)
        df.columns = [c.strip() for c in df.columns]
        
        # Remove metadata
        for meta_col in METADATA_COLS:
            if meta_col in df.columns:
                df = df.drop(columns=[meta_col])
        
        # Find label column
        label_col = find_label_column(df)
        if label_col is None:
            print("  No label column found!")
            return None
        
        # Convert to binary (VPN=1, NonVPN=0)
        labels = convert_to_binary_labels(df[label_col])
        if label_col != "label":
            df = df.drop(columns=[label_col])
        
        # Keep numeric
        numeric_data = df.select_dtypes(include=[np.number]).copy()
        numeric_data["label"] = labels.values
        numeric_data["source_file"] = csv_path.name
        
        # Clean
        numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if FAST_MODE and DOWNSAMPLE < 1.0:
            numeric_data = numeric_data.groupby("label", group_keys=False).apply(
                lambda x: x.sample(frac=DOWNSAMPLE, random_state=RANDOM_SEED)
            )
        
        print(f"  Loaded {len(numeric_data)} rows")
        return numeric_data
    
    except Exception as e:
        print(f"  Error loading ISCX: {e}")
        return None

def split_dataset(df):
    """Split data into train/val/test"""
    if "label" not in df.columns:
        print("No label column!")
        return None
    
    y = df["label"].astype(int)
    X = df.drop(columns=["label", "source_file"], errors="ignore")
    
    # First split: test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    
    # Second split: validation from remaining
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=RANDOM_SEED
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, X_test.copy()

def save_model_results(model, X_test, y_test, model_name, reports_dir):
    """Evaluate model and save results to file"""
    predictions = model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary", zero_division=0)
    
    try:
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probas)
        else:
            auc = "n/a"
    except:
        auc = "n/a"
    
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, digits=3)
    
    # Format results
    results_text = f"""Model: {model_name}
Accuracy: {acc:.4f}
Precision: {prec:.4f}
Recall: {rec:.4f}
F1 Score: {f1:.4f}
AUC: {auc}

Confusion Matrix:
{cm}

Classification Report:
{report}
"""
    
    # Save to file
    output_file = reports_dir / f"{model_name}_metrics.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(results_text)
    
    print(f"  Saved {model_name} results to {output_file}")
    print(results_text)
    
    return acc, f1

def train_on_dataset(df, dataset_name):
    """Train models on a dataset and save everything"""
    print(f"\n=== Training on {dataset_name} ===")
    
    # Set output paths based on dataset
    if "cicids" in dataset_name.lower():
        models_dir = MODELS_CIC
        reports_dir = REPORTS_CIC
        data_dir = DATA_CIC
    else:
        models_dir = MODELS_ISCX
        reports_dir = REPORTS_ISCX
        data_dir = DATA_ISCX
    
    # Split data
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, X_test_raw_keep = split_dataset(df)
    
    if X_train_raw is None:
        print("Failed to split data!")
        return None
    
    # Scale data
    scaler = None
    if SCALE_DATA:
        print("  Scaling features...")
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)
        X_val = pd.DataFrame(scaler.transform(X_val_raw), columns=X_val_raw.columns)
        X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns)
        
        # Save scaler
        scaler_path = models_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"  Saved scaler to {scaler_path}")
    else:
        X_train, X_val, X_test = X_train_raw, X_val_raw, X_test_raw
    
    # Save splits for dashboard
    print("  Saving data splits...")
    X_train.to_csv(data_dir / "x_train.csv", index=False)
    pd.Series(y_train, name="label").to_csv(data_dir / "y_train.csv", index=False)
    X_val.to_csv(data_dir / "x_val.csv", index=False)
    pd.Series(y_val, name="label").to_csv(data_dir / "y_val.csv", index=False)
    X_test.to_csv(data_dir / "x_test.csv", index=False)
    pd.Series(y_test, name="label").to_csv(data_dir / "y_test.csv", index=False)
    X_test_raw_keep.to_csv(data_dir / "X_test_raw.csv", index=False)
    
    # Train Random Forest
    print("  Training Random Forest...")
    rf_params = {
        "n_estimators": [200] if FAST_MODE else [200, 400],
        "max_depth": [None, 30]
    }
    
    rf = GridSearchCV(
        RandomForestClassifier(n_jobs=-1, random_state=RANDOM_SEED, class_weight="balanced"),
        param_grid=rf_params,
        scoring="f1",
        cv=3,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    joblib.dump(rf.best_estimator_, models_dir / "rf.joblib")
    rf_acc, rf_f1 = save_model_results(rf.best_estimator_, X_test, y_test, f"rf_{dataset_name}", reports_dir)
    
    # Train SGD Classifier (if we have both classes)
    if len(np.unique(y_train)) >= 2:
        print("  Training SGD Classifier...")
        sgd_params = {"alpha": [0.0001, 0.001]}
        
        sgd = GridSearchCV(
            SGDClassifier(loss="hinge", class_weight="balanced", random_state=RANDOM_SEED),
            param_grid=sgd_params,
            scoring="f1",
            cv=3,
            n_jobs=-1
        )
        
        sgd.fit(X_train, y_train)
        joblib.dump(sgd.best_estimator_, models_dir / "sgd.joblib")
        sgd_acc, sgd_f1 = save_model_results(sgd.best_estimator_, X_test, y_test, f"sgd_{dataset_name}", reports_dir)
    else:
        print("  Skipping SGD - only one class in training data")
    
    # Train XGBoost if available
    if XGBOOST_AVAILABLE:
        print("  Training XGBoost...")
        xgb_params = {
            "n_estimators": [200],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1]
        }
        
        xgb = GridSearchCV(
            XGBClassifier(n_jobs=-1, random_state=RANDOM_SEED, eval_metric="logloss"),
            param_grid=xgb_params,
            scoring="f1",
            cv=3,
            n_jobs=-1
        )
        
        xgb.fit(X_train, y_train)
        joblib.dump(xgb.best_estimator_, models_dir / "xgb.joblib")
        xgb_acc, xgb_f1 = save_model_results(xgb.best_estimator_, X_test, y_test, f"xgb_{dataset_name}", reports_dir)
    else:
        print("  Skipping XGBoost - not installed")
    
    print(f"  Done with {dataset_name}!")
    return scaler

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train models for traffic classification")
    parser.add_argument("--fast", action="store_true", help="Use fast mode (subset of data)")
    parser.add_argument("--train-iscx", action="store_true", help="Also train on ISCX dataset")
    
    args = parser.parse_args()
    
    # Update global fast mode if specified
    global FAST_MODE
    if args.fast:
        FAST_MODE = True
    
    print("Starting model training...")
    print(f"Fast mode: {FAST_MODE}")
    print(f"Train ISCX: {args.train_iscx}")
    
    # ===== Train on CICIDS =====
    print("\n" + "="*50)
    cicids_data = load_cicids_data(CICIDS_RAW)
    
    if cicids_data is None:
        print("Failed to load CICIDS data!")
        return
    
    print(f"CICIDS class balance: {cicids_data['label'].value_counts().to_dict()}")
    
    cicids_scaler = train_on_dataset(cicids_data, "cicids")
    
    # ===== Train on ISCX (optional) =====
    if args.train_iscx:
        print("\n" + "="*50)
        iscx_file = ISCX_RAW / "ISCXVPN2016.csv"
        
        if iscx_file.exists():
            iscx_data = load_iscx_data(iscx_file)
            if iscx_data is not None:
                print(f"ISCX class balance: {iscx_data['label'].value_counts().to_dict()}")
                train_on_dataset(iscx_data, "iscx")
            else:
                print("Failed to load ISCX data")
        else:
            print(f"ISCX file not found: {iscx_file}")
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Models saved to: {MODELS_CIC} and {MODELS_ISCX}")
    print(f"Reports saved to: {REPORTS_CIC} and {REPORTS_ISCX}")
    print(f"Data splits saved to: {DATA_CIC} and {DATA_ISCX}")

if __name__ == "__main__":
    main()