import logging
from typing import Optional, Dict, Any, List

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from worker.core.config import get_worker_settings
from worker.db.models import Base, EvaluationRecord

logger = logging.getLogger("crossverify.worker.db")


class DatabaseManager:
    """
    Async PostgreSQL connection pool and telemetry repository for CrossVerify Worker.
    Manages SQLAlchemy 2.0 async engine lifecycle, schema provisioning, and
    fault-tolerant evaluation persistence.
    """

    def __init__(self, database_url: Optional[str] = None):
        self.settings = get_worker_settings()
        self.database_url = database_url or self.settings.async_database_url
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self.is_connected: bool = False
        self.stats = {
            "records_saved": 0,
            "save_failures": 0
        }

    def _build_engine(self) -> AsyncEngine:
        """Create SQLAlchemy AsyncEngine tailored to PostgreSQL or SQLite test dialects."""
        is_sqlite = self.database_url.startswith("sqlite")
        if is_sqlite:
            return create_async_engine(
                self.database_url,
                echo=self.settings.DB_ECHO
            )

        return create_async_engine(
            self.database_url,
            pool_size=self.settings.DB_POOL_SIZE,
            max_overflow=self.settings.DB_MAX_OVERFLOW,
            pool_pre_ping=True,
            echo=self.settings.DB_ECHO
        )

    async def connect(self) -> bool:
        """Initialize async engine pool and verify database connectivity."""
        if self.engine and self.is_connected:
            return True

        try:
            self.engine = self._build_engine()
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            self.is_connected = True
            logger.info("[DatabaseManager] Connected to evaluation telemetry database.")
            return True
        except Exception as e:
            self.is_connected = False
            logger.warning(f"[DatabaseManager] Unable to connect to database: {e}")
            return False

    async def ensure_schema(self) -> bool:
        """
        Ensure evaluations table and required columns exist.
        Safe and idempotent on repeated invocations.
        """
        if not self.engine or not self.is_connected:
            connected = await self.connect()
            if not connected or not self.engine:
                return False

        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                # Ensure incremental migration columns exist on pre-existing Postgres instances
                if not self.database_url.startswith("sqlite"):
                    await conn.execute(
                        text("ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS champion_model VARCHAR(128)")
                    )
                    await conn.execute(
                        text("ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS candidate_model VARCHAR(128)")
                    )
            logger.info("[DatabaseManager] Evaluation database schema verified.")
            return True
        except Exception as e:
            logger.error(f"[DatabaseManager] Schema provisioning failed: {e}")
            return False

    async def disconnect(self):
        """Dispose connection pool cleanly."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.session_factory = None
        self.is_connected = False
        logger.info("[DatabaseManager] Disconnected from evaluation telemetry database.")

    async def save_evaluation(
        self,
        request_id: str,
        prompt: str,
        champion_model: Optional[str] = None,
        candidate_model: Optional[str] = None,
        champion_response: Optional[str] = None,
        candidate_response: Optional[str] = None,
        champion_latency_ms: Optional[float] = None,
        candidate_latency_ms: Optional[float] = None,
        semantic_similarity: Optional[float] = None,
        structural_valid: Optional[bool] = None,
        discrepancy_detected: bool = False,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[EvaluationRecord]:
        """Persist an evaluation telemetry record to PostgreSQL."""
        if not self.session_factory or not self.is_connected:
            connected = await self.connect()
            if not connected or not self.session_factory:
                self.stats["save_failures"] += 1
                logger.warning(f"[DatabaseManager] DB offline; dropped telemetry for {request_id}")
                return None

        record = EvaluationRecord(
            request_id=request_id,
            champion_model=champion_model,
            candidate_model=candidate_model,
            prompt=prompt,
            champion_response=champion_response,
            candidate_response=candidate_response,
            champion_latency_ms=champion_latency_ms,
            candidate_latency_ms=candidate_latency_ms,
            semantic_similarity=semantic_similarity,
            structural_valid=structural_valid,
            discrepancy_detected=discrepancy_detected,
            error_message=error_message,
            extra_metadata=metadata or {}
        )

        try:
            async with self.session_factory() as session:
                session.add(record)
                await session.commit()
                await session.refresh(record)
            self.stats["records_saved"] += 1
            logger.info(
                f"[DatabaseManager] Saved evaluation {request_id} "
                f"(similarity={semantic_similarity}, discrepancy={discrepancy_detected})"
            )
            return record
        except Exception as e:
            self.stats["save_failures"] += 1
            logger.error(f"[DatabaseManager] Failed to persist evaluation {request_id}: {e}")
            return None

    async def get_evaluation_by_request_id(self, request_id: str) -> Optional[EvaluationRecord]:
        """Fetch an evaluation record by its request identifier."""
        if not self.session_factory or not self.is_connected:
            return None

        async with self.session_factory() as session:
            stmt = select(EvaluationRecord).where(EvaluationRecord.request_id == request_id)
            result = await session.execute(stmt)
            return result.scalars().first()

    async def get_recent_evaluations(self, limit: int = 50) -> List[EvaluationRecord]:
        """Retrieve the most recent evaluation records ordered chronologically descending."""
        if not self.session_factory or not self.is_connected:
            return []

        async with self.session_factory() as session:
            stmt = (
                select(EvaluationRecord)
                .order_by(EvaluationRecord.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())
