#C:\Users\varun\Downloads\RAGBot\app\main.py
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import traceback
import time
import os
from dotenv import load_dotenv

# Import routers
from .api.auth_router import router as auth_router
from .api.chat_router import router as chat_router
from .api.user_router import router as user_router
from .api.health_router import router as health_router

# Import utilities
from .utils.logger import get_logger
from .rag_engine.db.session import get_db
from .rag_engine.aws.s3_config import get_s3_client
from .rag_engine.chroma.chroma_client import ChromaClient

from fastapi.middleware import Middleware
from app.middleware.jwt_middleware import JWTAuthMiddleware
from app.api import admin_router   # make sure __init__.py exists in app/api
# ...


# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting RAGBot FastAPI application")
    try:
        db = next(get_db())
        logger.info("Database connection successful")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")

    try:
        s3_client = get_s3_client()
        s3_client.list_buckets()  # Simple health check
        logger.info("S3 connection successful")
    except Exception as e:
        logger.error(f"S3 connection failed: {e}")

    try:
        chroma_client = ChromaClient(persist_directory=os.getenv("CHROMA_PERSIST_DIR", "app/data/chroma"))
        collection = chroma_client.get_doc_collection()
        collection.count()

        logger.info("Vectorstore available")
    except Exception as e:
        logger.error(f"Vectorstore init failed: {e}")

    yield
    logger.info("Shutting down RAGBot FastAPI application")

middleware = [
    Middleware(JWTAuthMiddleware)
]

app = FastAPI(
    title="RAGBot API",
    description="API for RAGBot - AI-powered document chat system with local RAG and web search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    middleware=middleware
)
app.include_router(auth_router)



# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGIN", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    return response

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_id = f"error_{int(time.time())}"
    is_dev = os.getenv("ENV", "dev") == "dev"
    
    logger.error(f"Exception [{error_id}]: {str(exc)}")
    if is_dev:
        logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_id": error_id,
            "message": "Unexpected error. Try again later."
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

# Include routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(chat_router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(user_router, prefix="/api/v1/user", tags=["User Management"])
app.include_router(health_router, prefix="/api/v1/health", tags=["Health"])
app.include_router(admin_router.router, prefix="/api/v1/admin", tags=["Admin"]) 

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "RAGBot API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

@app.get("/api/info")
async def api_info():
    return {
        "name": "RAGBot API",
        "version": "1.0.0",
        "description": "AI-powered document chat system",
        "features": [
            "Document ingestion and vectorization",
            "Local RAG search",
            "Web search integration",
            "User authentication",
            "Chat history management",
            "Multi-format response support"
        ],
        "endpoints": {
            "auth": "/api/v1/auth",
            "chat": "/api/v1/chat",
            "user": "/api/v1/user",
            "health": "/api/v1/health"
        }
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
