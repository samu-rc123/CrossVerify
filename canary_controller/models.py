import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from sqlalchemy import String, Text, Float, Boolean, DateTime, Index, Uuid, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(AsyncAttrs, DeclarativeBase):
    """Base SQLAlchemy 2.0 declarative class for canary controller."""
    pass


class EvaluationRecord(Base):
    """
    ORM model representing a single Champion vs. Candidate verification evaluation.
    Maps directly to the PostgreSQL 'evaluations' telemetry table.
    """
    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    request_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    champion_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    candidate_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    champion_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    candidate_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    champion_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    candidate_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    semantic_similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    structural_valid: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    discrepancy_detected: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extra_metadata: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True
    )

    __table_args__ = (
        Index("idx_canary_evaluations_models", "champion_model", "candidate_model"),
    )
