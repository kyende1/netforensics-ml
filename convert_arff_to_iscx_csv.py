from pathlib import Path
import sys
import pandas as pd
import numpy as np

# -------- CONFIG --------
ROOT = Path(r"C:\Users\OCHA\Desktop\IUBH\10. Project - Computer Science Project\netforensics")
RAW  = ROOT / "iscx2016_raw"  # put the two ARFFs in here
OUT  = RAW / "ISCXVPN2016.csv"

ARFF_VPN    = RAW / "TimeBasedFeatures-Dataset-15s-VPN.arff"
ARFF_NONVPN = RAW / "TimeBasedFeatures-Dataset-15s-NO-VPN.arff"
# ------------------------

def _strip_leading_noise(text: str) -> str:
    """Remove leading blank/comment lines before @RELATION."""
    lines = text.splitlines()
    out = []
    seen_relation = False
    for ln in lines:
        if not seen_relation:
            s = ln.strip()
            if not s or s.startswith("%"):  # ARFF comments start with %
                continue
            # first meaningful line should be @RELATION
            if s.upper().startswith("@RELATION"):
                seen_relation = True
                out.append(ln)
            else:
                # If we hit non-empty non-comment non-@RELATION, keep it anyway
                # (some files have extra headers); liac-arff may still fail.
                out.append(ln)
        else:
            out.append(ln)
    return "\n".join(out)

def load_arff(path: Path) -> pd.DataFrame:
    """Load ARFF with multiple strategies: liac-arff (utf-8-sig), fallback to SciPy."""
    if not path.exists():
        raise FileNotFoundError(path)

    # Try liac-arff (best with utf-8-sig to strip BOM)
    try:
        import arff  # liac-arff
        raw = path.read_text(encoding="utf-8-sig", errors="ignore")
        raw = _strip_leading_noise(raw)
        obj = arff.loads(raw)
        cols = [a[0] for a in obj["attributes"]]
        df = pd.DataFrame(obj["data"], columns=cols)
        return df
    except Exception as e_liac:
        # Fallback: SciPy
        try:
            from scipy.io import arff as scipy_arff
        except Exception as e_import:
            raise ImportError(
                "ARFF reader not available. Install one of:\n"
                "  pip install liac-arff\n  or\n"
                "  pip install scipy"
            ) from e_import

        # SciPy can read directly from file path; it returns structured arrays
        try:
            data, meta = scipy_arff.loadarff(str(path))
            df = pd.DataFrame(data)
            # decode bytes columns
            for c in df.columns:
                if df[c].dtype == object and len(df[c]) and isinstance(df[c].iloc[0], (bytes, bytearray)):
                    df[c] = df[c].str.decode("utf-8", errors="ignore")
            return df
        except Exception as e_scipy:
            raise RuntimeError(f"Failed to parse ARFF with both liac-arff and SciPy: {path}\n"
                               f"liac-arff error: {e_liac}\nscipy error: {e_scipy}") from e_scipy

def main():
    # Load both ARFFs
    print(f"[read] {ARFF_VPN}")
    df_vpn = load_arff(ARFF_VPN)
    print(f"[read] {ARFF_NONVPN}")
    df_non = load_arff(ARFF_NONVPN)

    # Keep only numeric features; add binary label
    num_vpn = df_vpn.select_dtypes(include=[np.number]).copy()
    num_non = df_non.select_dtypes(include=[np.number]).copy()

    if num_vpn.empty or num_non.empty:
        print("[warn] One of the ARFFs has no numeric columns after parsing. "
              "Inspect file headers/attributes.")
    num_vpn["label"] = 1
    num_non["label"] = 0

    out = pd.concat([num_vpn, num_non], ignore_index=True)
    out["__source_file__"] = "ARFF15s"

    # Clean infinities/NaN, drop constant cols
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    nunique = out.nunique()
    out = out[nunique[nunique > 1].index]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[ok] Wrote {len(out):,} rows to {OUT}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
