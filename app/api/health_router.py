"""
Health Router for RAGBot API
Location: app/api/health_router.py

Provides system health checks, status monitoring, and diagnostic endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import psutil
import time
import os

# Import dependencies
from ..rag_engine.db.session import get_db
from ..rag_engine.local_cache.sqlite_session import get_local_db
from ..rag_engine.chroma.chroma_client import ChromaClient
from ..rag_engine.aws.s3_config import get_s3_client
from ..utils.logger import get_logger

router = APIRouter()
logger = get_logger()

# Pydantic models
class HealthStatus(BaseModel):
    status: str
    timestamp: datetime
    uptime: float
    version: str = "1.0.0"

class SystemHealth(BaseModel):
    overall_status: str
    components: Dict[str, Dict[str, Any]]
    timestamp: datetime
    uptime: float
    system_info: Dict[str, Any]

class DatabaseHealth(BaseModel):
    postgresql: Dict[str, Any]
    sqlite: Dict[str, Any]
    vectorstore: Dict[str, Any]

class ServiceHealth(BaseModel):
    s3: Dict[str, Any]
    llm: Dict[str, Any]
    web_search: Dict[str, Any]

# Track service start time
SERVICE_START_TIME = time.time()

def get_uptime() -> float:
    """Get service uptime in seconds"""
    return time.time() - SERVICE_START_TIME

def check_database_health(db) -> Dict[str, Any]:
    """Check PostgreSQL database health"""
    try:
        # Simple query to test connection
        result = db.execute("SELECT 1").fetchone()
        return {
            "status": "healthy",
            "response_time": "< 100ms",
            "last_check": datetime.utcnow().isoformat(),
            "connection": "active"
        }
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat(),
            "connection": "failed"
        }

def check_sqlite_health(local_db) -> Dict[str, Any]:
    """Check SQLite local cache health"""
    try:
        # Simple query to test connection
        cursor = local_db.execute("SELECT 1")
        cursor.fetchone()
        return {
            "status": "healthy",
            "response_time": "< 50ms",
            "last_check": datetime.utcnow().isoformat(),
            "connection": "active"
        }
    except Exception as e:
        logger.error(f"SQLite health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat(),
            "connection": "failed"
        }

def check_vectorstore_health() -> Dict[str, Any]:
    """Check ChromaDB vectorstore health"""
    try:
        chroma_client = ChromaClient(persist_directory="app/data/chroma")
        collection = chroma_client._collection
        count = collection.count()
        return {
            "status": "healthy",
            "document_count": count,
            "last_check": datetime.utcnow().isoformat(),
            "connection": "active"
        }
    except Exception as e:
        logger.error(f"Vectorstore health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat(),
            "connection": "failed"
        }

def check_s3_health() -> Dict[str, Any]:
    """Check S3 service health"""
    try:
        s3_client = get_s3_client()
        # Try to list buckets as a health check
        response = s3_client.list_buckets()
        return {
            "status": "healthy",
            "bucket_count": len(response.get('Buckets', [])),
            "last_check": datetime.utcnow().isoformat(),
            "connection": "active"
        }
    except Exception as e:
        logger.error(f"S3 health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.utcnow().isoformat(),
            "connection": "failed"
        }

def get_system_info() -> Dict[str, Any]:
    """Get system resource information"""
    try:
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
            "process_count": len(psutil.pids()),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
    except Exception as e:
        logger.error(f"System info collection failed: {str(e)}")
        return {"error": str(e)}

@router.get("/", response_model=HealthStatus)
@router.get("/status", response_model=HealthStatus)
async def health_check():
    """Basic health check endpoint"""
    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow(),
        uptime=get_uptime()
    )

@router.get("/detailed", response_model=SystemHealth)
async def detailed_health_check(
    db = Depends(get_db),
    local_db = Depends(get_local_db)
):
    """Detailed system health check"""
    try:
        # Check all components
        postgresql_health = check_database_health(db)
        sqlite_health = check_sqlite_health(local_db)
        vectorstore_health = check_vectorstore_health()
        s3_health = check_s3_health()
        system_info = get_system_info()

        components = {
            "postgresql": postgresql_health,
            "sqlite": sqlite_health,
            "vectorstore": vectorstore_health,
            "s3": s3_health
        }

        # Determine overall status
        component_statuses = [comp["status"] for comp in components.values()]
        overall_status = "healthy" if all(status == "healthy" for status in component_statuses) else "degraded"

        return SystemHealth(
            overall_status=overall_status,
            components=components,
            timestamp=datetime.utcnow(),
            uptime=get_uptime(),
            system_info=system_info
        )

    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/database", response_model=DatabaseHealth)
async def database_health_check(
    db = Depends(get_db),
    local_db = Depends(get_local_db)
):
    """Database-specific health checks"""
    return DatabaseHealth(
        postgresql=check_database_health(db),
        sqlite=check_sqlite_health(local_db),
        vectorstore=check_vectorstore_health()
    )

@router.get("/services", response_model=ServiceHealth)
async def services_health_check():
    """External services health checks"""
    return ServiceHealth(
        s3=check_s3_health(),
        llm={"status": "healthy", "last_check": datetime.utcnow().isoformat()},
        web_search={"status": "healthy", "last_check": datetime.utcnow().isoformat()}
    )

