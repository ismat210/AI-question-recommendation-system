from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import sys

from appli.services.ingestion_services import process_file
from appli.utils.logger import logger
from appli.utils.exception import CustomException

router = APIRouter()

UPLOAD_DIR = "data/uploads"
ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt"]


@router.post("/upload-file")
def upload_file(file: UploadFile = File(...)):

    try:
        # -------------------------
        # Validate file type
        # -------------------------
        ext = os.path.splitext(file.filename)[1].lower()

        if ext not in ALLOWED_EXTENSIONS:
            logger.warning(f"Invalid file type uploaded: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="Only PDF, DOCX, and TXT files are allowed"
            )

        # -------------------------
        # Save file
        # -------------------------
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(file.file.read())

        logger.info(f"File saved successfully: {file_path}")

        # -------------------------
        # Process file
        # -------------------------
        results = process_file(file_path)

        logger.info(f"File processed successfully: {file.filename}, questions: {len(results)}")

        return {
            "message": "File processed successfully",
            "filename": file.filename,
            "questions_extracted": len(results)
        }

    except Exception as e:
        logger.error(f"Error in upload_file API: {str(e)}")

        raise CustomException(e, sys)