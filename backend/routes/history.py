from fastapi import APIRouter, Depends, HTTPException
from bson.objectid import ObjectId
from database import get_db
from auth import get_current_user
from logger import app_logger

router = APIRouter(prefix="/api/history", tags=["History"])


@router.get("/")
def get_history(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
):
    app_logger.info(f"User {current_user['id']} requested prediction history")
    predictions = list(
        db.predictions.find({"user_id": current_user["id"]})
        .sort("created_at", -1)
    )

    return [
        {
            "id": str(p["_id"]),
            "filename": p["filename"],
            "file_type": p["file_type"],
            "prediction": p["prediction"],
            "confidence": p["confidence"],
            "created_at": p["created_at"].isoformat(),
        }
        for p in predictions
    ]


@router.get("/{prediction_id}")
def get_prediction_detail(
    prediction_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
):
    app_logger.info(f"User {current_user['id']} requested details for prediction {prediction_id}")
    try:
        obj_id = ObjectId(prediction_id)
    except:
        app_logger.warning(f"Invalid Prediction ID format: {prediction_id}")
        raise HTTPException(400, "Invalid prediction ID format")

    prediction = db.predictions.find_one({
        "_id": obj_id,
        "user_id": current_user["id"]
    })
    
    if not prediction:
        app_logger.warning(f"Prediction {prediction_id} not found for user {current_user['id']}")
        raise HTTPException(404, "Prediction not found")

    return {
        "id": str(prediction["_id"]),
        "filename": prediction["filename"],
        "file_type": prediction["file_type"],
        "prediction": prediction["prediction"],
        "confidence": prediction["confidence"],
        "eeg_data": prediction["eeg_data"],
        "signal_stats": prediction.get("signal_stats"),
        "created_at": prediction["created_at"].isoformat(),
    }
