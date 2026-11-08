import os
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gateway.api.routes import router as v1_router
from gateway.core.config import get_settings
from gateway.core.broker import traffic_broker

# Setup logging
settings = get_settings()
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("crossverify.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management for connection pools and brokers."""
    logger.info("Initializing CrossVerify API Gateway...")
    await traffic_broker.connect()
    yield
    logger.info("Shutting down CrossVerify API Gateway...")
    await traffic_broker.disconnect()


app = FastAPI(
    title="CrossVerify API Gateway",
    description="Enterprise Continuous Verification Engine for Dual-Path LLM Ingress",
    version="0.3.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Measure full HTTP request processing time."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


# Include API routers
app.include_router(v1_router)


@app.get("/", tags=["system"])
async def root():
    """Service metadata and ingress status."""
    return {
        "service": "CrossVerify API Gateway",
        "version": "0.3.0",
        "status": "operational",
        "champion_model": settings.CHAMPION_MODEL_NAME,
        "shadow_stream": settings.REDIS_STREAM_KEY,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/health", tags=["system"])
async def health():
    """Healthcheck endpoint reporting service and message broker telemetry."""
    broker_metrics = traffic_broker.get_metrics()
    return {
        "status": "healthy",
        "service": "gateway",
        "environment": settings.ENVIRONMENT,
        "broker": broker_metrics,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
