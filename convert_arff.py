import numpy as np  # Explicitly added NumPy import
from scipy.io import arff
import pandas as pd
from pathlib import Path
import glob

# Define paths
root = Path(r"C:\Users\OCHA\Desktop\IUBH\10. Project - Computer Science Project\netforensics")
input_folder = root / "iscx_arff"  # Folder with ISCXVPN2016 .arff files
output_csv = root / "cicids2017_raw" / "ISCXVPN2016.csv"
output_csv.parent.mkdir(parents=True, exist_ok=True)

# Find all ARFF files
arff_files = glob.glob(str(input_folder / "*.arff"))
if not arff_files:
    raise FileNotFoundError(f"No ARFF files found in {input_folder}")

# Convert and merge all ARFF files
frames = []
for file in arff_files:
    try:
        data, meta = arff.loadarff(file)
        df = pd.DataFrame(data)
        # Decode byte strings and handle empty values
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.decode('utf-8')
                # Replace empty strings with NaN before numeric conversion
                df[col] = df[col].replace('', np.nan)
        # Check for potential label columns (expanded list based on filename hints)
        label_col = next((col for col in ['Label', 'Class', 'TrafficType', 'Category', 'Type'] if col in df.columns), None)
        if label_col:
            if df[label_col].dtype == object:
                df['label'] = df[label_col].str.upper().replace({'VPN': 1, 'NO-VPN': 0, 'NONVPN': 0}, inplace=False)
            else:
                df['label'] = df[label_col].map({1: 1, 0: 0})  # Handle numeric labels
            if label_col != 'label':
                df.drop(columns=[label_col], inplace=True)
        else:
            print(f"Warning: No recognizable label column ('Label', 'Class', 'TrafficType', 'Category', 'Type') in {file}. Skipping label conversion.")
        frames.append(df)
    except Exception as e:
        print(f"Error processing {file}: {e}")
        continue

merged_df = pd.concat(frames, ignore_index=True)
# Clean data with NumPy
merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()
drop_cols = {'Flow ID', 'Src IP', 'Dst IP', 'Timestamp'}
merged_df = merged_df.drop(columns=drop_cols, errors='ignore')

# Save the merged CSV
merged_df.to_csv(output_csv, index=False)
print(f"Merged CSV saved: {output_csv} ({len(merged_df)} rows, {merged_df.shape[1]} columns)")