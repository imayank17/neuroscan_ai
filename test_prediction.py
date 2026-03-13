import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "backend"))
from ml_service import predict_seizure

data_path = "backend/data.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    # Find a seizure sample (y=1)
    seizure_sample = df[df['y'] == 1].iloc[0]
    feature_cols = [f"X{i}" for i in range(1, 179)]
    eeg_values = seizure_sample[feature_cols].values.tolist()
    
    # Directly try to load model to see the error
    try:
        from tensorflow.keras.models import load_model
        model_path = os.path.join(os.getcwd(), "backend", "Epilepsy.h5")
        print(f"Directly loading from: {model_path}")
        # Try loading without compilation to see if it's a serialization issue
        model = load_model(model_path, compile=False)
        print("Model loaded successfully with compile=False")
        print("Model Input:", model.input_shape)
    except Exception as e:
        print(f"DIRECT LOAD ERROR: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nTesting with sample labeled y={seizure_sample['y']}")
    # Fallback to the real ml_service logic
    try:
        result = predict_seizure(eeg_values)
        print("Prediction Result:", result['prediction'])
        print("Demo Mode:", result.get('demo_mode'))
    except Exception as e:
        print(f"PREDICT ERROR: {e}")
else:
    print("data.csv not found")
