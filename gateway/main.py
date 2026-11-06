import os
from datetime import datetime, timezone
from fastapi import FastAPI

app = FastAPI(
    title="CrossVerify API Gateway",
    description="Dual-Path LLM Verification & Traffic Shadowing Gateway",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {
        "service": "CrossVerify API Gateway",
        "version": "0.1.0",
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "gateway",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

