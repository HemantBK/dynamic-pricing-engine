"""FastAPI application entry point for the Dynamic Pricing Engine.

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import lifespan
from src.api.routes import router
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Dynamic Pricing Engine",
    description=(
        "Real-time pricing recommendations using ML demand forecasting, "
        "price elasticity estimation, and revenue optimization."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with timestamp and latency."""
    start = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"-> {response.status_code} ({latency_ms:.1f}ms)"
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler with sanitized messages."""
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# Register routes
app.include_router(router)
