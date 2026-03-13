import pandas as pd
import numpy as np
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))
from ml_service import predict_seizure

data_path = "backend/data.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    feature_cols = [f"X{i}" for i in range(1, 179)]
    
    # 1. Test Seizure (y=1)
    seizure_sample = df[df['y'] == 1].iloc[0]
    eeg_values_s = seizure_sample[feature_cols].values.tolist()
    print(f"\n--- Testing Seizure Sample (y=1) ---")
    result_s = predict_seizure(eeg_values_s)
    print("Prediction:", result_s['prediction'])
    print("Confidence:", result_s['confidence'])
    print("Seizure Prob:", result_s.get('seizure_probability'))
    
    # 2. Test Non-Seizure (y=2 or 3 or 4 or 5)
    non_seizure_sample = df[df['y'] != 1].iloc[0]
    eeg_values_ns = non_seizure_sample[feature_cols].values.tolist()
    print(f"\n--- Testing Non-Seizure Sample (y={non_seizure_sample['y']}) ---")
    result_ns = predict_seizure(eeg_values_ns)
    print("Prediction:", result_ns['prediction'])
    print("Confidence:", result_ns['confidence'])
    print("Seizure Prob:", result_ns.get('seizure_probability'))

else:
    print("data.csv not found")
