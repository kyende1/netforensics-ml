# train_cicids.py
# Trains CICIDS2017 and ISCX2016 pipelines, saves models/splits/reports,
# and cross-evaluates the CICIDS RF on ISCX using a feature-name normalizer.

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)

# Optional: XGBoost (extra model for Phase 3)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warn] xgboost is not installed; XGBoost model will be skipped.")

# --------------------------- ROOTS (your structure) ---------------------------
ROOT = Path(r"C:\Users\OCHA\Desktop\IUBH\10. Project - Computer Science Project\netforensics")

# Inputs
RAW_CICIDS = ROOT / "cicids2017_raw"           # daily CICIDS2017 CSVs
RAW_ISCX   = ROOT / "iscx2016_raw"             # contains ISCXVPN2016.csv

# CICIDS outputs
MODELS_CIC = ROOT / "models_cicids2017";    MODELS_CIC.mkdir(parents=True, exist_ok=True)
REPORTS_CIC= ROOT / "reports_cicids2017";   REPORTS_CIC.mkdir(parents=True, exist_ok=True)
DATA_CIC   = ROOT / "data_cicids2017";      DATA_CIC.mkdir(parents=True, exist_ok=True)

# ISCX outputs
MODELS_ISCX = ROOT / "models_iscx2016";     MODELS_ISCX.mkdir(parents=True, exist_ok=True)
REPORTS_ISCX= ROOT / "reports_iscx2016";    REPORTS_ISCX.mkdir(parents=True, exist_ok=True)
DATA_ISCX   = ROOT / "data_iscx2016";       DATA_ISCX.mkdir(parents=True, exist_ok=True)

# --------------------------- SETTINGS ---------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
SCALE        = True

# Fast mode defaults (overridden by --fast)
FAST_MODE       = True
LIMIT_FILES     = 3            # only first N CICIDS files (None for all)
USE_CHUNKS      = True         # read CSVs in chunks to save RAM
CHUNK_SIZE      = 200_000
DTYPE_FLOAT     = "float32"    # shrink numeric dtypes
DOWNSAMPLE_FRAC = 0.30         # per-class fraction (None to disable)

ID_COLS = ("Flow ID", "Src IP", "Dst IP", "Timestamp")

# --------------------------- UTILS ---------------------------
def _save_splits(Xtr, ytr, Xva, yva, Xte, yte, out_dir: Path, Xte_raw: pd.DataFrame | None = None):
    """Save scaled splits for the app; optionally also save raw X_test for cross-eval."""
    # Lowercase filenames to match app.py expectations
    pd.DataFrame(Xtr, columns=Xtr.columns).to_csv(out_dir / "x_train.csv", index=False)
    pd.Series(ytr, name="label").to_csv(out_dir / "y_train.csv", index=False)
    pd.DataFrame(Xva, columns=Xva.columns).to_csv(out_dir / "x_val.csv", index=False)
    pd.Series(yva, name="label").to_csv(out_dir / "y_val.csv", index=False)
    pd.DataFrame(Xte, columns=Xte.columns).to_csv(out_dir / "x_test.csv", index=False)
    pd.Series(yte, name="label").to_csv(out_dir / "y_test.csv", index=False)
    if Xte_raw is not None:
        # keep the original, unscaled features for cross-eval alignment
        pd.DataFrame(Xte_raw, columns=Xte_raw.columns).to_csv(out_dir / "X_test_raw.csv", index=False)

def _eval_model(model, X, y, name: str, reports_dir: Path):
    yhat = model.predict(X)
    acc = accuracy_score(y, yhat)
    p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    try:
        proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
    except Exception:
        auc = None
    cm  = confusion_matrix(y, yhat)
    rep = classification_report(y, yhat, digits=3)
    txt = f"""{name}
ACC={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}  AUC={auc if auc is not None else 'n/a'}
Confusion Matrix:
{cm}
Report:
{rep}
"""
    (reports_dir / f"{name}_metrics.txt").write_text(txt, encoding="utf-8")
    print(txt)

def _split_rowwise(df: pd.DataFrame):
    """Row-wise stratified split (works even with a single source file)."""
    y = df["label"].astype(int)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        df, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, stratify=y_tmp, random_state=RANDOM_STATE
    )
    def _xy(d: pd.DataFrame):
        yy = d["label"].astype(int).values
        XX = d.drop(columns=["label", "__source_file__"], errors="ignore")
        return XX, yy
    return _xy(X_train), _xy(X_val), _xy(X_test), X_test.drop(columns=["label","__source_file__"], errors="ignore")

# --------------------------- CICIDS LOADING ---------------------------
def _list_cicids_files(folder: Path, limit: int | None) -> list[Path]:
    files = sorted(folder.glob("*.csv"))
    # ignore any ad-hoc "clean" merged file; use official daily CSVs
    files = [f for f in files if "clean" not in f.name.lower()]
    if limit: files = files[:limit]
    return files

def _find_cicids_label(df: pd.DataFrame) -> str | None:
    lower = {c.strip().lower(): c for c in df.columns}
    for k in ("label", "class", "attack", "category"):
        if k in lower:
            return lower[k]
    return None

def _label_cicids_binary(series: pd.Series) -> pd.Series:
    vals = series.astype(str).str.strip().str.upper()
    return (vals != "BENIGN").astype(np.int8)

def read_cicids_all(folder: Path) -> pd.DataFrame:
    files = _list_cicids_files(folder, LIMIT_FILES if FAST_MODE else None)
    if not files:
        raise SystemExit(f"No CICIDS CSV files found in {folder}.")

    frames = []
    for csv in files:
        if USE_CHUNKS:
            for chunk in pd.read_csv(csv, engine=None, low_memory=True, chunksize=CHUNK_SIZE):
                chunk.columns = [c.strip() for c in chunk.columns]
                for c in ID_COLS:
                    if c in chunk.columns: chunk.drop(columns=c, inplace=True)
                lab = _find_cicids_label(chunk)
                if lab is None:
                    print(f"[skip] {csv.name}: no label in this chunk."); continue
                y = _label_cicids_binary(chunk[lab])
                if lab != "label": chunk.drop(columns=[lab], inplace=True)
                chunk["label"] = y
                chunk["__source_file__"] = csv.name
                num = chunk.select_dtypes(include=[np.number]).copy()
                if DTYPE_FLOAT:
                    for col in num.columns:
                        if col != "label": num[col] = num[col].astype(DTYPE_FLOAT, errors="ignore")
                num["__source_file__"] = chunk["__source_file__"].values
                frames.append(num)
        else:
            df = pd.read_csv(csv, engine="pyarrow", low_memory=True)
            df.columns = [c.strip() for c in df.columns]
            for c in ID_COLS:
                if c in df.columns: df.drop(columns=c, inplace=True)
            lab = _find_cicids_label(df)
            if lab is None:
                print(f"[skip] {csv.name}: no label; skipping file."); continue
            y = _label_cicids_binary(df[lab])
            if lab != "label": df.drop(columns=[lab], inplace=True)
            df["label"] = y
            df["__source_file__"] = csv.name
            num = df.select_dtypes(include=[np.number]).copy()
            if DTYPE_FLOAT:
                for col in num.columns:
                    if col != "label": num[col] = num[col].astype(DTYPE_FLOAT, errors="ignore")
            num["__source_file__"] = df["__source_file__"].values
            frames.append(num)

    if not frames:
        raise SystemExit("No usable CICIDS rows loaded (no label columns found).")

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.replace([np.inf, -np.inf], np.nan).dropna()
    nunique = all_df.nunique()
    all_df = all_df[nunique[nunique > 1].index]

    # fast-mode downsample per class
    if FAST_MODE and DOWNSAMPLE_FRAC and 0 < DOWNSAMPLE_FRAC < 1 and "label" in all_df.columns:
        all_df = (
            all_df.groupby("label", group_keys=False)
                  .apply(lambda d: d.sample(frac=DOWNSAMPLE_FRAC, random_state=RANDOM_STATE))
        )
    return all_df

# --------------------------- ISCX LOADING ---------------------------
def _detect_iscx_label_column(df: pd.DataFrame) -> str | None:
    lower = {c.strip().lower(): c for c in df.columns}
    for k in ("label", "class", "vpn", "category", "app", "application", "class1"):
        if k in lower:
            return lower[k]
    return None

def _label_iscx_binary(series: pd.Series) -> pd.Series:
    """
    Map ISCX labels to binary: 1 = VPN, 0 = NonVPN.
    Accept strings like "VPN", "NonVPN", "NO-VPN", numeric 0/1, or application names:
    treat any label containing the substring 'VPN' as VPN.
    """
    s = series.astype(str).str.upper().str.replace("-", "", regex=False).str.replace(" ", "", regex=False)
    if set(s.unique()) <= {"0", "1"}:
        return s.astype(int).astype(np.int8)
    return s.str.contains("VPN").astype(np.int8)

def read_iscx_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(f"ISCX file not found: {csv_path}")
    frames = []
    if USE_CHUNKS:
        for chunk in pd.read_csv(csv_path, engine=None, low_memory=True, chunksize=CHUNK_SIZE):
            chunk.columns = [c.strip() for c in chunk.columns]
            for c in ID_COLS:
                if c in chunk.columns: chunk.drop(columns=c, inplace=True)
            lab = _detect_iscx_label_column(chunk)
            if lab is None:
                print("[skip] ISCX chunk: no label; skipping."); continue
            y = _label_iscx_binary(chunk[lab])
            if lab != "label": chunk.drop(columns=[lab], inplace=True)
            chunk["label"] = y
            chunk["__source_file__"] = "ISCXVPN2016.csv"
            num = chunk.select_dtypes(include=[np.number]).copy()
            if DTYPE_FLOAT:
                for col in num.columns:
                    if col != "label": num[col] = num[col].astype(DTYPE_FLOAT, errors="ignore")
            num["__source_file__"] = chunk["__source_file__"].values
            frames.append(num)
    else:
        df = pd.read_csv(csv_path, engine="pyarrow", low_memory=True)
        df.columns = [c.strip() for c in df.columns]
        for c in ID_COLS:
            if c in df.columns: df.drop(columns=c, inplace=True)
        lab = _detect_iscx_label_column(df)
        if lab is None:
            raise SystemExit("ISCX file has no recognizable VPN/NonVPN label column.")
        y = _label_iscx_binary(df[lab])
        if lab != "label": df.drop(columns=[lab], inplace=True)
        df["label"] = y
        df["__source_file__"] = "ISCXVPN2016.csv"
        num = df.select_dtypes(include=[np.number]).copy()
        if DTYPE_FLOAT:
            for col in num.columns:
                if col != "label": num[col] = num[col].astype(DTYPE_FLOAT, errors="ignore")
        num["__source_file__"] = df["__source_file__"].values
        frames.append(num)

    if not frames:
        raise SystemExit("No usable ISCX rows loaded (no label detected).")

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.replace([np.inf, -np.inf], np.nan).dropna()
    nunique = all_df.nunique()
    all_df = all_df[nunique[nunique > 1].index]
    return all_df

# --------------------------- NAME NORMALIZER ---------------------------
def _canon(name: str) -> str:
    """Canonicalize a feature name: lowercase, drop non-alnum (incl. spaces/_/-)."""
    s = "".join(ch for ch in name.lower() if ch.isalnum())
    return s

def _build_canonical_index(columns: list[str]) -> dict[str, str]:
    """
    Build map from canonical name -> ORIGINAL name.
    If duplicates map to the first occurrence.
    """
    idx: dict[str, str] = {}
    for col in columns:
        c = _canon(col)
        if c and c not in idx:
            idx[c] = col
    return idx

def align_like(
    X_new: pd.DataFrame,
    train_columns: list[str],
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Return a DataFrame whose columns and order exactly match train_columns.
    - Try to match by canonical name.
    - For missing columns, fill with scaler.mean_ if provided, else 0.
    - Extra columns in X_new are dropped.
    Also returns a small stats dict.
    """
    # indexes for matching
    train_canon = _build_canonical_index(train_columns)
    new_canon = _build_canonical_index(list(X_new.columns))

    # build aligned frame
    aligned = {}
    matched = 0
    missing = []
    for i, col in enumerate(train_columns):
        c = _canon(col)
        if c in new_canon:
            src = new_canon[c]
            aligned[col] = X_new[src].values
            matched += 1
        else:
            # fill with training mean => standardized value 0 after transform
            if scaler is not None and hasattr(scaler, "mean_") and len(scaler.mean_) == len(train_columns):
                aligned[col] = np.full((len(X_new),), scaler.mean_[i])
            else:
                aligned[col] = np.zeros((len(X_new),), dtype=float)
            missing.append(col)

    X_aligned = pd.DataFrame(aligned, columns=train_columns)
    stats = {"matched": matched, "missing": len(missing), "missing_list": missing[:20]}
    return X_aligned, stats

# --------------------------- TRAIN BLOCK ---------------------------
def train_block(df: pd.DataFrame, out_models: Path, out_reports: Path, out_data: Path, tag: str):
    # split (and keep raw X_test for cross-eval)
    (Xtr_raw, ytr), (Xva_raw, yva), (Xte_raw, yte), Xte_raw_only = _split_rowwise(df)

    # scale
    scaler_path = out_models / "scaler.joblib"
    if SCALE:
        scaler = StandardScaler().fit(Xtr_raw)
        Xtr = pd.DataFrame(scaler.transform(Xtr_raw), columns=Xtr_raw.columns)
        Xva = pd.DataFrame(scaler.transform(Xva_raw), columns=Xva_raw.columns)
        Xte = pd.DataFrame(scaler.transform(Xte_raw), columns=Xte_raw.columns)
        joblib.dump(scaler, scaler_path)
        print(f"[{tag}] Saved scaler: {scaler_path}")
    else:
        scaler = None
        Xtr, Xva, Xte = Xtr_raw, Xva_raw, Xte_raw

    # save splits (and raw X_test)
    _save_splits(Xtr, ytr, Xva, yva, Xte, yte, out_data, Xte_raw=Xte_raw_only)

    # RF
    print(f"[{tag}] Training RandomForest…")
    rf = GridSearchCV(
        RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"),
        param_grid={
            "n_estimators": [200] if FAST_MODE else [200, 400],
            "max_depth": [None, 30] if FAST_MODE else [None, 20, 40]
        },
        scoring="f1", cv=3, n_jobs=-1
    ).fit(Xtr, ytr)
    joblib.dump(rf.best_estimator_, out_models / "rf.joblib")
    _eval_model(rf.best_estimator_, Xte, yte, f"rf_{tag}", out_reports)

    # Linear SVM surrogate (SGD) if both classes present
    if len(np.unique(ytr)) >= 2:
        print(f"[{tag}] Training linear SVM surrogate (SGDClassifier)…")
        sgd = GridSearchCV(
            SGDClassifier(loss="hinge", class_weight="balanced", random_state=RANDOM_STATE),
            param_grid={"alpha": [0.0001, 0.001] if FAST_MODE else [1e-4, 5e-4, 1e-3]},
            scoring="f1", cv=3, n_jobs=-1
        ).fit(Xtr, ytr)
        joblib.dump(sgd.best_estimator_, out_models / "sgd.joblib")
        _eval_model(sgd.best_estimator_, Xte, yte, f"sgd_{tag}", out_reports)
    else:
        print(f"[{tag}] Warning: training set has one class — skipping SGDClassifier.")

    # XGBoost (Phase 3 extension)
    if HAS_XGB:
        print(f"[{tag}] Training XGBoost…")
        xgb = GridSearchCV(
            XGBClassifier(
                n_jobs=-1,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                tree_method="hist"
            ),
            param_grid={
                "n_estimators": [200] if FAST_MODE else [200, 400],
                "max_depth": [4, 6],
                "learning_rate": [0.05, 0.1],
            },
            scoring="f1",
            cv=3,
            n_jobs=-1
        ).fit(Xtr, ytr)
        joblib.dump(xgb.best_estimator_, out_models / "xgb.joblib")
        _eval_model(xgb.best_estimator_, Xte, yte, f"xgb_{tag}", out_reports)
    else:
        print(f"[{tag}] Skipping XGBoost (xgboost not installed).")

    return scaler_path

# --------------------------- CROSS EVAL (CICIDS RF on ISCX) ---------------------------
def cross_eval_cic_to_iscx(models_cic: Path, scaler_path: Path, reports_cic: Path):
    rf_path = models_cic / "rf.joblib"
    if not rf_path.exists():
        print("[cross] CICIDS RF not found; skipping cross-evaluation.")
        return
    if not scaler_path.exists():
        print("[cross] CICIDS scaler not found; skipping (model was likely trained unscaled).")
        return

    # Load CICIDS model + scaler and its training columns
    rf = joblib.load(rf_path)
    scaler: StandardScaler = joblib.load(scaler_path)
    if hasattr(scaler, "feature_names_in_"):
        train_cols = list(scaler.feature_names_in_)
    else:
        # fallback
        train_cols = list(pd.read_csv(DATA_CIC / "x_train.csv", nrows=1).columns)

    # Prefer raw ISCX test (unscaled)
    Xi_raw_p = DATA_ISCX / "X_test_raw.csv"
    yi_p     = DATA_ISCX / "y_test.csv"
    if not (Xi_raw_p.exists() and yi_p.exists()):
        print("[cross] ISCX raw test not found; try building it by running with --train-iscx first.")
        return

    Xi_raw = pd.read_csv(Xi_raw_p)
    yi_df  = pd.read_csv(yi_p)
    y_col = "label" if "label" in yi_df.columns else yi_df.columns[0]
    yi = yi_df[y_col].astype(int).values

    # Align ISCX columns to CICIDS training columns by canonical name
    Xi_aligned, stats = align_like(Xi_raw, train_cols, scaler=scaler)
    print(f"[cross] aligned columns: matched={stats['matched']} missing={stats['missing']}")
    if stats["matched"] < max(5, int(0.15 * len(train_cols))):
        print("[cross] Too few matched features; skipping cross-eval to avoid misleading results.")
        return

    # Apply CICIDS scaler and evaluate
    Xi_scaled = pd.DataFrame(scaler.transform(Xi_aligned), columns=train_cols)
    _eval_model(rf, Xi_scaled, yi, "rf_cicids_on_iscx", reports_cic)

# --------------------------- MAIN ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Fast mode: subset + chunked reading + smaller grids")
    parser.add_argument("--train-iscx", action="store_true", help="Also train ISCX2016 (ISCXVPN2016.csv)")
    args = parser.parse_args()

    global FAST_MODE
    if args.fast:
        FAST_MODE = True
    print(f"[CONFIG] FAST_MODE={FAST_MODE}  LIMIT_FILES={LIMIT_FILES}  USE_CHUNKS={USE_CHUNKS}")

    # 1) CICIDS
    print("[1/4] Loading CICIDS2017…")
    cic = read_cicids_all(RAW_CICIDS)
    cls = cic["label"].value_counts().to_dict()
    print(f"[info] CICIDS class balance: {cls}")
    print(f"Loaded rows: {len(cic)}  cols: {cic.shape[1]}")

    # 2) Train CICIDS
    print("[2/4] Training CICIDS2017 models…")
    scaler_cic_path = train_block(cic, MODELS_CIC, REPORTS_CIC, DATA_CIC, tag="cicids")

    # 3) ISCX (optional)
    if args.train_iscx:
        print("[3/4] Loading + training ISCX2016…")
        iscx_csv = RAW_ISCX / "ISCXVPN2016.csv"
        if not iscx_csv.exists():
            print(f"[warn] {iscx_csv} not found. Skipping ISCX training.")
        else:
            iscx = read_iscx_csv(iscx_csv)
            cls_i = iscx["label"].value_counts().to_dict()
            print(f"[info] ISCX rows: {len(iscx)}  cols: {iscx.shape[1]}  class balance: {cls_i}")
            train_block(iscx, MODELS_ISCX, REPORTS_ISCX, DATA_ISCX, tag="iscx")

    # 4) Cross-evaluate CICIDS RF on ISCX (if ISCX raw test exists)
    # print("[4/4] Cross-evaluating CICIDS RF on ISCX (if splits exist)…")
    # cross_eval_cic_to_iscx(MODELS_CIC, scaler_cic_path, REPORTS_CIC)

if __name__ == "__main__":
    main()
