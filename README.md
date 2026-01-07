# Supervised Detection of Malicious Flows from Encrypted Traffic Metadata
# DLMCSPCSP01 – Computer Science Project Portfolio

Author: Thomas Kyende  
Matriculation Number: 321146438  
Institution: IU International University of Applied Sciences  

## Project Overview
This project is about building a machine learning system that detects malicious network flows using just the metadata from encrypted traffic—no need for deep packet inspection or decrypting anything. It's privacy-friendly and helps with digital investigations in cybersecurity.  

I developed it in three phases:  
- **Phase 1**: Focused on data preprocessing and feature engineering.  
- **Phase 2**: Built the supervised ML models, evaluated them, and created a dashboard.  
- **Phase 3**: Added some enhancements based on professor feedback, like including XGBoost for comparison, a live traffic simulation for testing dynamic scenarios, and exporting alerts in a format that works with SIEM systems.  

## Key Features
- Privacy-preserving: Only uses metadata, no payload snooping.  
- Models: Random Forest, SGDClassifier (like a linear SVM), and XGBoost (added in Phase 3).  
- Tested generalization by applying CICIDS2017 models to ISCXVPN2016 data.  
- Interactive dashboard using Streamlit for easy evaluation and predictions.  
- Simulation mode to mimic live traffic and generate alerts.  
- Exports alerts as JSONL, which can plug into SIEM/IDS tools.  
- Training scripts that are modular and easy to run.  

## System Architecture
The setup has two main parts:  
1. **Training Layer**:  
   - Loads and cleans data.  
   - Trains the models.  
   - Does cross-dataset checks.  
   - Saves models using joblib.  

2. **Deployment Layer**:  
   - Loads the saved models.  
   - Runs predictions.  
   - Handles simulations and alert exports.  
   - Dashboard for interacting with everything.  

## Project Structure
.
├── app.py                     # The Streamlit dashboard app
├── train_cicids.py            # Script for training and evaluating models
├── convert_arff.py            # Helper to convert ARFF files (for ISCX)
├── convert_arff_to_iscx_csv.py # Another converter for ISCX data
├── tools/                     # Any extra utilities (if needed)
├── assets/                    # Images or figures for the dashboard
├── models_cicids2017/         # Where CICIDS models get saved
├── models_iscx2016/           # Where ISCX models get saved
├── requirements.txt           # List of Python packages needed
└── README.md                  # This file
(Note: I didn't include trained models in the submission—they get generated when you run the training script. Same for large datasets.)

## Requirements
- Python 3.9 or higher  
- Install packages with: `pip install -r requirements.txt`  
- If XGBoost gives trouble (it did on my machine once), try `pip install xgboost` separately.  

## Running the Dashboard
Just run: `streamlit run app.py`  
It'll open in your browser (usually at http://localhost:8501).  

From there, you can:  
- Pick a dataset and model.  
- See performance metrics on test data.  
- Compare models (especially with XGBoost now).  
- Simulate traffic flows and export alerts.  

## Model Training
To train models yourself: `python train_cicids.py`  
Add `--train-iscx` if you want to include ISCX too.  
Use `--fast` for quicker runs (smaller grids, sampling).  

This will:  
- Load the datasets (assuming they're in the right folders).  
- Train RF, SGD, and XGBoost.  
- Save models to the models_ folders.  
- Create reports with metrics.  

## Datasets
Uses:  
- CICIDS2017 (main one).  
- ISCXVPN2016 (for validation).  

Datasets aren't in the ZIP because they're huge—download them from the sources mentioned in the report. Once downloaded, put them in cicids2017_raw/ and iscx2016_raw/. The converters handle ARFF to CSV if needed.

## SIEM / IDS Alert Export
In the dashboard's simulation tab, it generates alerts and exports them as JSONL. This format works with tools like:  
- Elastic SIEM  
- Splunk  
- Wazuh  

Each alert has:  
- Timestamp  
- Prediction (malicious or not)  
- Score/confidence  
- Some flow details  

I added this in Phase 3 to show how it could integrate into real systems.

## Evaluation Summary
Detailed in the report, but quick highlights:  
- Models perform well (high F1/AUC).  
- Cross-dataset works okay, shows robustness.  
- Simulation tests dynamic stuff successfully.  
- Overall, it's feasible for real use.  

## Ethical & Privacy Considerations
Important: This doesn't look at packet contents, just metadata. So it keeps user privacy intact while still detecting threats. I made sure to follow ethical guidelines in the design.

## Future Work
Some ideas for next steps:  
- Add real-time streaming with Kafka or something.  
- Deploy to cloud for scalability.  
- Auto-retrain if data drifts.  
- Maybe try lightweight neural nets.  

## Documentation
Everything's explained in the academic report submitted with this. It covers methodology, results, and how it all ties together.

## Final Note
This was a fun project—started with basic analysis and built up to something deployable. It balances tech, ethics, and practicality for cybersecurity.

