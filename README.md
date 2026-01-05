# Supervised Detection of Malicious Flows from Encrypted Traffic Metadata

**Author:** Thomas Kyende  
**Matriculation Number:** 321146438  
**Institution:** IU International University of Applied Sciences  

## Project Overview
This project implements a privacy-preserving supervised machine learning system for detecting malicious network flows using encrypted traffic metadata only, without Deep Packet Inspection. The system was developed in three iterative phases:

### Phase 1: 
Data preprocessing and feature engineering framework

### Phase 2
Supervised ML modelling, evaluation and dashboard implementation

### Phase 3
Operational maturity enhancements:
- Added XGBoost model  
- Added live traffic simulation / real-time inference capability  
- Added SIEM-compatible JSONL alert export  

## Key Features
- Privacy-preserving classification  
- Models: Random Forest, SGDClassifier (Linear SVM), and XGBoost  
- Cross-dataset generalisation (CICIDS2017 → ISCXVPN2016)  
- Interactive Streamlit dashboard  
- Live traffic simulation  
- SIEM / IDS alert export (JSONL)  
- Modular training & evaluation scripts  

## System Architecture
The solution follows a two-layer pipeline.

### 1 Training Layer
- Data ingestion  
- Cleaning & preprocessing  
- Model training  
- Cross-dataset validation  
- Model persistence (joblib)

### 2 Deployment Layer
- Loads trained models  
- Runs inference  
- Live simulation  
- Alert export  
- Dashboard interaction
  
## Project Structure
- ├── app.py
- ├── train_cicids.py
- ├── convert_arff.py
- ├── convert_arff_to_iscx_csv.py
- ├── tools/
- ├── assets/
- ├── models_cicids2017/
- ├── models_iscx2016/
- ├── requirements.txt
- └── README.md
> Large datasets are not included due to size limits.

## Requirements
Python 3.9+
Install dependencies:
pip install -r requirements.txt
If XGBoost fails: pip install xgboost

## Running the Dashboard
streamlit run app.py
Open: http://localhost:8501

## Dashboard Capabilities
- Load trained models  
- Evaluate test data  
- Compare performance  
- Simulate live traffic  
- Export SIEM alerts in JSONL

## Training Models
python train_cicids.py
Outputs saved to:
- models_cicids2017/
- models_iscx2016/

## Datasets
- CICIDS2017  
- ISCXVPN2016  

> Publicly available and referenced in report.

## SIEM Integration
Exports JSONL alerts compatible with:
- Elastic SIEM  
- Splunk  
- Wazuh  

Each alert includes:
- Timestamp  
- Classification result  
- Confidence score  
- Flow attributes  

## Evaluation Summary
- Strong performance
- Cross-dataset robustness
- Live validation successful
- Deployment feasible
  
## Ethics & Privacy
- No packet payload inspection  
- Metadata only  
- Privacy-preserving approach  

## Future Work
- Kafka / Spark streaming  
- Cloud deployment  
- Model retraining automation  
- Lightweight deep learning  

## Documentation
Full methodology and evaluation are discussed in the academic report.


