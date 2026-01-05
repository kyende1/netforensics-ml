Supervised Detection of Malicious Flows from Encrypted Traffic Metadata
DLMCSPCSP01 – Computer Science Project Portfolio

Author: Thomas Kyende
Matriculation Number: 321146438
Institution: IU International University of Applied Sciences

Project Overview
This project implements a privacy-preserving supervised machine learning system for detecting malicious network flows using encrypted traffic metadata only, without Deep Packet Inspection. The system was developed in three iterative phases:

Phase 1: Data preprocessing and feature engineering framework
Phase 2: Supervised ML modelling, evaluation and dashboard implementation
Phase 3: Operational maturity enhancements:
    - Added XGBoost model
    - Added live traffic simulation / real-time inference capability
    - Added SIEM-compatible JSONL alert export

Key Features
    - Privacy-preserving classification
    - Models: Random Forest, SGDClassifier (Linear SVM), and XGBoost
    - Cross-dataset generalisation (CICIDS2017 → ISCXVPN2016)
    - Interactive visual dashboard (Streamlit)
    - Live traffic simulation mode
    - SIEM / IDS alert export (JSONL format)
    - Modular training and evaluation scripts

System Architecture
The solution follows a two-layer pipeline:
    1. Training Layer
        - Data ingestion
        - Cleaning & preprocessing
        - Model training
        - Cross-dataset validation
        - Model persistence (joblib)

    2. Deployment Layer
        - Loads trained models
        - Runs inference
        - Live simulation
        - Alert export
        - Dashboard interaction

Project Structure
                .
        ├── app.py                     # Streamlit dashboard
        ├── train_cicids.py            # Training + evaluation pipeline
        ├── convert_arff.py            # Dataset conversion helper
        ├── convert_arff_to_iscx_csv.py
        ├── tools/                     # Utilities (if applicable)
        ├── assets/                    # UI / figures / small assets
        ├── models_cicids2017/         # (optional) Trained models
        ├── models_iscx2016/           # (optional) Trained models
        ├── requirements.txt
        └── README.md
Large datasets are intentionally NOT included in submission due to size.

Requirements
    - Python 3.9+
    - pip
        - Install dependencies: pip install -r requirements.txt
        - If XGBoost installation fails on some systems: pip install xgboost

Running the Dashboard
Start the application: streamlit run app.py
Then open the browser link (usually): http://localhost:8501

Dashboard Capabilities
    - Load trained models
    - Evaluate test data
    - Compare model performance
    - Simulate live traffic
    - Export alerts in JSONL

Model Training
If you want to train your own models: python train_cicids.py

This script:
    - Loads dataset (if available)
    - Trains Random Forest, SGD, XGBoost
    - Saves trained models
    - Generates evaluation reports
    Models will be saved under: models_cicids2017/
                                models_iscx2016/
Datasets
This system works with:
- CICIDS2017
- ISCXVPN2016
Datasets are publicly available and referenced in the accompanying academic report. They are not included in submission due to size.

Due to submission storage limits, datasets are:
- Not included in ZIP
- Fully referenced in project report
- Publicly available for download
Once downloaded, update dataset paths in training scripts if needed.

SIEM / IDS Alert Export
    - The system supports JSONL alert export compatible with:
    - Elastic SIEM
    - Splunk
    - Wazuh
    - Other SOC tools

    Each alert includes:
    - Timestamp
    - Classification result
    - Confidence score
    - Flow attributes

Evaluation Summary
    - Key evaluation outcomes (detailed in report):
    - Strong performance across ML models
    - Demonstrated cross-dataset robustness
    - Live inference validation successful
    - Practical deployment feasibility shown

Ethical & Privacy Considerations
This project:
    - Does NOT inspect packet payloads
    - Uses metadata only
    - Preserves user privacy

Aligns with responsible cybersecurity engineering practices
Future Work
    - Suggested extensions:
    - Full real-time streaming (Kafka / Spark)
    - Cloud deployment
    - Automated retraining / drift detection
    - Lightweight deep learning deployment

Documentation
Full methodology, evaluation and deployment discussion are included in the final academic report submitted with this project.

Final Note
This project demonstrates a progression from analytical research to deployment-oriented cybersecurity engineering, balancing technical rigour, ethical awareness, and practical applicability.Supervised Detection of Malicious Flows from Encrypted Traffic Metadata
DLMCSPCSP01 – Computer Science Project Portfolio

Author: Thomas Kyende
Matriculation Number: 321146438
Institution: IU International University of Applied Sciences

Project Overview
This project implements a privacy-preserving supervised machine learning system for detecting malicious network flows using encrypted traffic metadata only, without Deep Packet Inspection. The system was developed in three iterative phases:

Phase 1: Data preprocessing and feature engineering framework
Phase 2: Supervised ML modelling, evaluation and dashboard implementation
Phase 3: Operational maturity enhancements:
    - Added XGBoost model
    - Added live traffic simulation / real-time inference capability
    - Added SIEM-compatible JSONL alert export

Key Features
    - Privacy-preserving classification
    - Models: Random Forest, SGDClassifier (Linear SVM), and XGBoost
    - Cross-dataset generalisation (CICIDS2017 → ISCXVPN2016)
    - Interactive visual dashboard (Streamlit)
    - Live traffic simulation mode
    - SIEM / IDS alert export (JSONL format)
    - Modular training and evaluation scripts

System Architecture
The solution follows a two-layer pipeline:
    1. Training Layer
        - Data ingestion
        - Cleaning & preprocessing
        - Model training
        - Cross-dataset validation
        - Model persistence (joblib)

    2. Deployment Layer
        - Loads trained models
        - Runs inference
        - Live simulation
        - Alert export
        - Dashboard interaction

Project Structure
                .
        ├── app.py                     # Streamlit dashboard
        ├── train_cicids.py            # Training + evaluation pipeline
        ├── convert_arff.py            # Dataset conversion helper
        ├── convert_arff_to_iscx_csv.py
        ├── tools/                     # Utilities (if applicable)
        ├── assets/                    # UI / figures / small assets
        ├── models_cicids2017/         # (optional) Trained models
        ├── models_iscx2016/           # (optional) Trained models
        ├── requirements.txt
        └── README.md
Large datasets are intentionally NOT included in submission due to size.

Requirements
    - Python 3.9+
    - pip
        - Install dependencies: pip install -r requirements.txt
        - If XGBoost installation fails on some systems: pip install xgboost

Running the Dashboard
Start the application: streamlit run app.py
Then open the browser link (usually): http://localhost:8501

Dashboard Capabilities
    - Load trained models
    - Evaluate test data
    - Compare model performance
    - Simulate live traffic
    - Export alerts in JSONL

Model Training
If you want to train your own models: python train_cicids.py

This script:
    - Loads dataset (if available)
    - Trains Random Forest, SGD, XGBoost
    - Saves trained models
    - Generates evaluation reports
    Models will be saved under: models_cicids2017/
                                models_iscx2016/
Datasets
This system works with:
- CICIDS2017
- ISCXVPN2016
Datasets are publicly available and referenced in the accompanying academic report. They are not included in submission due to size.

Due to submission storage limits, datasets are:
- Not included in ZIP
- Fully referenced in project report
- Publicly available for download
Once downloaded, update dataset paths in training scripts if needed.

SIEM / IDS Alert Export
    - The system supports JSONL alert export compatible with:
    - Elastic SIEM
    - Splunk
    - Wazuh
    - Other SOC tools

    Each alert includes:
    - Timestamp
    - Classification result
    - Confidence score
    - Flow attributes

Evaluation Summary
    - Key evaluation outcomes (detailed in report):
    - Strong performance across ML models
    - Demonstrated cross-dataset robustness
    - Live inference validation successful
    - Practical deployment feasibility shown

Ethical & Privacy Considerations
This project:
    - Does NOT inspect packet payloads
    - Uses metadata only
    - Preserves user privacy

Aligns with responsible cybersecurity engineering practices
Future Work
    - Suggested extensions:
    - Full real-time streaming (Kafka / Spark)
    - Cloud deployment
    - Automated retraining / drift detection
    - Lightweight deep learning deployment

Documentation
Full methodology, evaluation and deployment discussion are included in the final academic report submitted with this project.

Final Note
This project demonstrates a progression from analytical research to deployment-oriented cybersecurity engineering, balancing technical rigour, ethical awareness, and practical applicability.
