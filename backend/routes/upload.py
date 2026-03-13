import os
import csv
import io
import json
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from database import get_db
from auth import get_current_user
from ml_service import predict_seizure
from image_processor import extract_signal_from_image
from logger import app_logger

router = APIRouter(prefix="/api/upload", tags=["Upload & Analysis"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def parse_csv_eeg(content: bytes) -> list:
    """Parse CSV file and extract EEG values (X1-X178)."""
    text = content.decode("utf-8")
    df = pd.read_csv(io.StringIO(text))

    # Try to find the 178 feature columns
    feature_cols = [c for c in df.columns if c.startswith("X") and c[1:].isdigit()]

    if len(feature_cols) >= 178:
        feature_cols = sorted(feature_cols, key=lambda c: int(c[1:]))[:178]
        row = df[feature_cols].iloc[0].values.tolist()
        return row
    elif df.shape[1] >= 178:
        # Assume first 178 numeric columns are features
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:178]
        row = df[numeric_cols].iloc[0].values.tolist()
        return row
    else:
        raise ValueError(f"CSV must have at least 178 numeric columns. Found {df.shape[1]}.")


def generate_sample_eeg() -> list:
    """Generate sample EEG data for non-CSV uploads (image/PDF demo)."""
    np.random.seed(42)
    t = np.linspace(0, 1, 178)
    signal = (
        50 * np.sin(2 * np.pi * 8 * t)  # Alpha rhythm
        + 30 * np.sin(2 * np.pi * 20 * t)  # Beta rhythm
        + np.random.randn(178) * 20
    )
    return signal.tolist()


@router.post("/analyze")
async def upload_and_analyze(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
):
    """Upload EEG data and run seizure prediction."""
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    allowed_exts = {".csv", ".png", ".jpg", ".jpeg", ".pdf", ".edf", ".txt"}
    if ext not in allowed_exts:
        app_logger.warning(f"Rejected unsupported file upload: {filename} (extension: {ext})")
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_exts)}")

    app_logger.info(f"Received file upload from user {current_user['id']}: {filename}")
    content = await file.read()

    # Save file
    save_path = os.path.join(UPLOAD_DIR, f"{current_user['id']}_{filename}")
    with open(save_path, "wb") as f:
        f.write(content)

    # Parse EEG data
    is_csv = ext == ".csv"
    try:
        if ext in {".png", ".jpg", ".jpeg"}:
            app_logger.info(f"Processing EEG image: {filename}")
            eeg_values = extract_signal_from_image(content)
            file_type = "image"
        elif is_csv:
            eeg_values = parse_csv_eeg(content)
            file_type = "csv"
        else:
            eeg_values = generate_sample_eeg()
            file_type = ext.replace(".", "")
    except Exception as e:
        app_logger.error(f"File parsing error for {filename}: {e}")
        raise HTTPException(400, f"Error parsing file: {str(e)}")

    # Run prediction
    result = predict_seizure(eeg_values)

    # Save to database
    prediction_doc = {
        "user_id": current_user["id"],
        "filename": filename,
        "file_type": file_type,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "eeg_data": eeg_values[:178],
        "signal_stats": result.get("signal_stats"),
        "created_at": datetime.utcnow()
    }
    
    insert_result = db.predictions.insert_one(prediction_doc)
    app_logger.info(f"Analysis saved to database with ID: {insert_result.inserted_id}")

    return {
        "id": str(insert_result.inserted_id),
        "filename": filename,
        "file_type": file_type,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "class_probabilities": result.get("class_probabilities"),
        "signal_stats": result.get("signal_stats"),
        "eeg_data": eeg_values[:178],
        "demo_mode": result.get("demo_mode", False),
        "is_csv": is_csv,
        "created_at": prediction_doc["created_at"].isoformat(),
    }


@router.get("/sample-data")
def get_sample_data():
    """Return sample EEG data for testing."""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data.csv")
    if not os.path.exists(data_path):
        return {"eeg_values": generate_sample_eeg()}

    df = pd.read_csv(data_path, nrows=5)
    feature_cols = [f"X{i}" for i in range(1, 179)]
    samples = []
    for idx, row in df.iterrows():
        samples.append({
            "label": int(row["y"]),
            "values": row[feature_cols].values.tolist(),
        })
    return {"samples": samples}
