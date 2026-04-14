from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from appli.api.routes_upload import router as upload_router
from appli.api.routes_recommend import router as recommend_router

from appli.utils.logger import logger
from appli.utils.exception import CustomException


# -------------------------
# APP INIT
# -------------------------
app = FastAPI(
    title="AI Question Recommendation System",
    version="1.0.0"
)


# -------------------------
# STARTUP EVENT
# -------------------------
@app.on_event("startup")
def startup_event():
    logger.info(" FastAPI server started successfully")


# -------------------------
# INCLUDE ROUTERS
# -------------------------
app.include_router(upload_router, prefix="/api")
app.include_router(recommend_router, prefix="/api")


# -------------------------
# GLOBAL EXCEPTION HANDLER
# -------------------------
@app.exception_handler(CustomException)
def custom_exception_handler(request: Request, exc: CustomException):
    logger.error(f"CustomException: {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "message": "Something went wrong"
        }
    )


# -------------------------
# ROOT ENDPOINT
# -------------------------
@app.get("/")
def home():
    logger.info("Root endpoint called")

    return {
        "message": "Backend is running "
    }