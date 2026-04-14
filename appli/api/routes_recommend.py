from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys

from appli.utils.logger import logger
from appli.utils.exception import CustomException

# Import your trained model loader (adjust if needed)
import pickle
import os

router = APIRouter()


# -------------------------
# REQUEST SCHEMA
# -------------------------
class RecommendRequest(BaseModel):
    query: str
    top_k: int = 5


# -------------------------
# LOAD MODEL
# -------------------------
MODEL_PATH = "Mac_Lear/models/best_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Recommendation model loaded successfully")

except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise CustomException(e, sys)


# -------------------------
# RECOMMEND API
# -------------------------
@router.post("/recommend")
def recommend(request: RecommendRequest):

    try:
        logger.info(f"Recommendation request received: {request.query}")

        # Get results from model
        results = model.search(request.query, top_k=request.top_k)

        logger.info(f"Returned {len(results)} recommendations")

        return {
            "query": request.query,
            "results": results
        }

    except Exception as e:
        logger.error(f"Error in recommend API: {str(e)}")
        raise CustomException(e, sys)