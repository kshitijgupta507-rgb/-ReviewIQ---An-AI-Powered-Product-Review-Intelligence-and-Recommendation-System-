"""FastAPI application factory for ReviewIQ.

Start with:
    uvicorn app.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import rag, recommend, search, sentiment, summary
from reviewiq.utils.logger import setup_logging

# Initialise logging once at startup
setup_logging()
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="ReviewIQ API",
        description="AI-Powered Product Review Intelligence and Recommendation System",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Allow Streamlit frontend and web clients to connect
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Register Routers ───────────────────────────────────────────────────────
    application.include_router(search.router, prefix="/search", tags=["Search & NER"])
    application.include_router(sentiment.router, prefix="/sentiment", tags=["Sentiment & ABSA"])
    application.include_router(recommend.router, prefix="/recommend", tags=["Recommendation"])
    application.include_router(summary.router, prefix="/summary", tags=["Summarization"])
    application.include_router(rag.router, prefix="/rag", tags=["RAG Assistant"])

    @application.get("/health", tags=["Health"])
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {
            "status": "ok",
            "version": "1.0.0",
            "service": "ReviewIQ Intelligence Engine",
        }

    logger.info("ReviewIQ API successfully configured with all routers.")
    return application


app = create_app()
