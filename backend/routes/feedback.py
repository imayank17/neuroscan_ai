from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from bson.objectid import ObjectId
from database import get_db
from auth import get_current_user
from logger import app_logger

router = APIRouter(prefix="/api/feedback", tags=["Feedback"])


class FeedbackCreate(BaseModel):
    prediction_id: str
    rating: int  # 1-5
    comment: str = ""


class FeedbackResponse(BaseModel):
    id: str
    prediction_id: str
    rating: int
    comment: str
    created_at: str


@router.post("/", response_model=FeedbackResponse)
def create_feedback(
    data: FeedbackCreate,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
):
    app_logger.info(f"User {current_user['id']} submitting feedback for prediction {data.prediction_id}")
    if data.rating < 1 or data.rating > 5:
        app_logger.warning(f"Invalid rating submitted: {data.rating}")
        raise HTTPException(400, "Rating must be between 1 and 5")

    try:
        obj_id = ObjectId(data.prediction_id)
    except:
        app_logger.warning(f"Invalid Prediction ID format in feedback: {data.prediction_id}")
        raise HTTPException(400, "Invalid prediction ID format")

    prediction = db.predictions.find_one({
        "_id": obj_id,
        "user_id": current_user["id"]
    })

    if not prediction:
        app_logger.warning(f"Feedback failed: Prediction {data.prediction_id} not found for user {current_user['id']}")
        raise HTTPException(404, "Prediction not found")

    feedback_doc = {
        "user_id": current_user["id"],
        "prediction_id": data.prediction_id,
        "rating": data.rating,
        "comment": data.comment or "",
        "created_at": datetime.utcnow()
    }
    
    result = db.feedbacks.insert_one(feedback_doc)
    app_logger.info(f"Feedback saved successfully with ID: {result.inserted_id}")

    return FeedbackResponse(
        id=str(result.inserted_id),
        prediction_id=feedback_doc["prediction_id"],
        rating=feedback_doc["rating"],
        comment=feedback_doc["comment"],
        created_at=feedback_doc["created_at"].isoformat(),
    )


@router.get("/{prediction_id}")
def get_feedback(
    prediction_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
):
    feedbacks = list(
        db.feedbacks.find({
            "prediction_id": prediction_id,
            "user_id": current_user["id"]
        }).sort("created_at", -1)
    )

    return [
        {
            "id": str(f["_id"]),
            "rating": f["rating"],
            "comment": f["comment"],
            "created_at": f["created_at"].isoformat() if isinstance(f["created_at"], datetime) else f["created_at"],
        }
        for f in feedbacks
    ]
