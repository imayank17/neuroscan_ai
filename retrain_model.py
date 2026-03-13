import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Config
DATA_PATH = "backend/data.csv"
MODEL_PATH = "backend/Epilepsy.h5"

def train_model():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Feature columns (X1 to X178)
    X = df.drop(['Unnamed: 0', 'y'], axis=1).values
    # Labels (1-5) -> 0-4 for categorical
    y = df['y'].values - 1
    
    print(f"Original X shape: {X.shape}")
    
    # Preprocessing as in ml_service (subsample every 4th point)
    X_subsampled = X[:, ::4]
    
    # APPLY NORMALIZATION (Crucial Fix)
    # Per-sample normalization matches ml_service.py logic
    X_normalized = []
    for sample in X_subsampled:
        mean = sample.mean()
        std = sample.std()
        if std == 0: std = 1.0
        X_normalized.append((sample - mean) / std)
    
    X_final = np.array(X_normalized)
    
    # Reshape for LSTM: (samples, time_steps, features)
    X_final = X_final.reshape(-1, X_final.shape[1], 1)
    
    # One-hot encode labels
    y_categorical = to_categorical(y, num_classes=5)
    
    # Build Model (Balanced LSTM)
    model = Sequential([
        LSTM(64, input_shape=(X_final.shape[1], 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training model (with normalization)...")
    model.fit(X_final, y_categorical, epochs=15, batch_size=32, validation_split=0.1, verbose=1)
    
    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Model saved successfully!")

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        train_model()
    else:
        print("data.csv not found at backend/data.csv")
