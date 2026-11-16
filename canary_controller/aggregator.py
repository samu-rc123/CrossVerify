import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from canary_controller.core.config import get_canary_settings
from canary_controller.models import EvaluationRecord

logger = logging.getLogger("crossverify.canary.aggregator")


class WindowMetrics(BaseModel):
    """Rolling statistical telemetry summary over a designated sliding time window."""
    window_start: datetime = Field(..., description="Start of evaluation window (UTC)")
    window_end: datetime = Field(..., description="End of evaluation window (UTC)")
    window_minutes: int = Field(..., description="Duration of evaluation window in minutes")
    total_samples: int = Field(..., description="Total inference evaluation samples in window")
    discrepancy_count: int = Field(..., description="Count of evaluations with discrepancy_detected=True")
    drift_rate: float = Field(..., description="Ratio of discrepancies to total samples (0.0 - 1.0)")
    error_count: int = Field(..., description="Count of evaluations with candidate errors or timeouts")
    error_rate: float = Field(..., description="Ratio of candidate errors to total samples (0.0 - 1.0)")
    avg_champion_latency_ms: float = Field(..., description="Mean Champion model latency in ms")
    avg_candidate_latency_ms: float = Field(..., description="Mean Candidate model latency in ms")
    p50_candidate_latency_ms: float = Field(..., description="50th percentile (median) Candidate latency in ms")
    p95_candidate_latency_ms: float = Field(..., description="95th percentile Candidate latency in ms")
    p99_candidate_latency_ms: float = Field(..., description="99th percentile Candidate latency in ms")
    avg_semantic_similarity: float = Field(..., description="Mean semantic cosine similarity (0.0 - 1.0)")
    structural_discrepancy_count: int = Field(..., description="Count of evaluations with structural defects")
    structural_pass_rate: float = Field(..., description="Ratio of structurally valid evaluations (0.0 - 1.0)")


class TelemetryAggregator:
    """
    Sliding window telemetry query and metrics computation engine.
    Aggregates live evaluation records from PostgreSQL over rolling intervals
    to calculate drift rates, latency percentiles, and reliability SLO metrics.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        engine: Optional[AsyncEngine] = None
    ):
        self.settings = get_canary_settings()
        self.database_url = database_url or self.settings.async_database_url
        if engine:
            self.engine = engine
            self._owns_engine = False
        else:
            is_sqlite = self.database_url.startswith("sqlite")
            if is_sqlite:
                self.engine = create_async_engine(self.database_url, echo=self.settings.DB_ECHO)
            else:
                self.engine = create_async_engine(
                    self.database_url,
                    pool_size=self.settings.DB_POOL_SIZE,
                    max_overflow=self.settings.DB_MAX_OVERFLOW,
                    pool_pre_ping=True,
                    echo=self.settings.DB_ECHO
                )
            self._owns_engine = True

        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def close(self):
        """Dispose database engine if owned."""
        if self._owns_engine and self.engine:
            await self.engine.dispose()

    async def compute_window_metrics(
        self,
        window_minutes: Optional[int] = None,
        candidate_model: Optional[str] = None,
        as_of: Optional[datetime] = None
    ) -> WindowMetrics:
        """
        Query database for records within [now - window_minutes, now] and calculate
        statistical aggregations (averages, p50, p95, p99, drift rates).
        """
        window_duration = window_minutes or self.settings.CANARY_EVAL_WINDOW_MINUTES
        window_end = as_of or datetime.now(timezone.utc)
        window_start = window_end - timedelta(minutes=window_duration)

        conditions = [
            EvaluationRecord.created_at >= window_start,
            EvaluationRecord.created_at <= window_end
        ]
        if candidate_model:
            conditions.append(EvaluationRecord.candidate_model == candidate_model)

        async with self.session_factory() as session:
            stmt = select(EvaluationRecord).where(and_(*conditions))
            result = await session.execute(stmt)
            records: List[EvaluationRecord] = list(result.scalars().all())

        total_samples = len(records)

        # Zero sample baseline handling
        if total_samples == 0:
            return WindowMetrics(
                window_start=window_start,
                window_end=window_end,
                window_minutes=window_duration,
                total_samples=0,
                discrepancy_count=0,
                drift_rate=0.0,
                error_count=0,
                error_rate=0.0,
                avg_champion_latency_ms=0.0,
                avg_candidate_latency_ms=0.0,
                p50_candidate_latency_ms=0.0,
                p95_candidate_latency_ms=0.0,
                p99_candidate_latency_ms=0.0,
                avg_semantic_similarity=1.0,
                structural_discrepancy_count=0,
                structural_pass_rate=1.0
            )

        # Discrepancy & Error counts
        discrepancy_count = sum(1 for r in records if r.discrepancy_detected)
        error_count = sum(1 for r in records if r.error_message or (r.candidate_latency_ms is None))
        structural_discrepancy_count = sum(
            1 for r in records if r.structural_valid is False
        )

        drift_rate = round(discrepancy_count / total_samples, 4)
        error_rate = round(error_count / total_samples, 4)
        structural_pass_rate = round(
            (total_samples - structural_discrepancy_count) / total_samples, 4
        )

        # Latencies
        champ_latencies = [
            r.champion_latency_ms for r in records if r.champion_latency_ms is not None
        ]
        cand_latencies = [
            r.candidate_latency_ms for r in records if r.candidate_latency_ms is not None
        ]

        avg_champ_latency = round(float(np.mean(champ_latencies)), 2) if champ_latencies else 0.0
        avg_cand_latency = round(float(np.mean(cand_latencies)), 2) if cand_latencies else 0.0

        if cand_latencies:
            cand_arr = np.array(cand_latencies, dtype=np.float64)
            p50 = round(float(np.percentile(cand_arr, 50)), 2)
            p95 = round(float(np.percentile(cand_arr, 95)), 2)
            p99 = round(float(np.percentile(cand_arr, 99)), 2)
        else:
            p50 = p95 = p99 = 0.0

        # Semantic similarities
        similarities = [
            r.semantic_similarity for r in records if r.semantic_similarity is not None
        ]
        avg_similarity = round(float(np.mean(similarities)), 4) if similarities else 0.0

        return WindowMetrics(
            window_start=window_start,
            window_end=window_end,
            window_minutes=window_duration,
            total_samples=total_samples,
            discrepancy_count=discrepancy_count,
            drift_rate=drift_rate,
            error_count=error_count,
            error_rate=error_rate,
            avg_champion_latency_ms=avg_champ_latency,
            avg_candidate_latency_ms=avg_cand_latency,
            p50_candidate_latency_ms=p50,
            p95_candidate_latency_ms=p95,
            p99_candidate_latency_ms=p99,
            avg_semantic_similarity=avg_similarity,
            structural_discrepancy_count=structural_discrepancy_count,
            structural_pass_rate=structural_pass_rate
        )
