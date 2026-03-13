from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from auth import router as auth_router
from routes.upload import router as upload_router
from routes.reports import router as reports_router
from routes.history import router as history_router
from routes.feedback import router as feedback_router
from logger import app_logger
import traceback

app = FastAPI(
    title="NeuroScan AI — Epileptic Seizure Detection",
    description="AI-powered EEG analysis for epileptic seizure detection",
    version="1.0.0",
)

@app.on_event("startup")
async def startup_event():
    app_logger.info("Starting NeuroScan AI Server...")

@app.on_event("shutdown")
async def shutdown_event():
    app_logger.info("Shutting down NeuroScan AI Server...")

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    response = JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    response = JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"Unhandled Exception on {request.method} {request.url.path}: {str(exc)}"
    app_logger.error(error_msg)
    app_logger.error(traceback.format_exc())
    response = JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later.", "error": str(exc)},
    )
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.middleware("http")
async def force_cors_middleware(request: Request, call_next):
    # This ensures success responses also have the headers
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Standard CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(reports_router)
app.include_router(history_router)
app.include_router(feedback_router)


@app.get("/")
def root():
    return {
        "name": "NeuroScan AI",
        "version": "1.0.0",
        "description": "Epileptic Seizure Detection from EEG Signals",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
