import json
import logging
from enum import Enum
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import redis.asyncio as aioredis
from pydantic import BaseModel, Field

from canary_controller.core.config import get_canary_settings
from canary_controller.aggregator import WindowMetrics

logger = logging.getLogger("crossverify.canary.circuit_breaker")


class CircuitBreakerState(str, Enum):
    """Operational state of the Canary Circuit Breaker."""
    CLOSED = "CLOSED"      # Normal healthy operation: traffic shadowing permitted
    OPEN = "OPEN"          # Aborted: SLO breached, traffic shadowing halted
    HALF_OPEN = "HALF_OPEN"# Cooldown recovery verification


class CircuitBreakerDecision(BaseModel):
    """Decision output produced on each rolling window evaluation cycle."""
    state: CircuitBreakerState
    tripped: bool
    violations: List[str] = Field(default_factory=list)
    reason: Optional[str] = None
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: WindowMetrics


class CanaryCircuitBreaker:
    """
    Automated Canary Circuit Breaker & SLO Governance Engine.
    Continuously compares rolling telemetry metrics against statistical thresholds:
    - Maximum Semantic Drift: 5% (CANARY_DRIFT_THRESHOLD)
    - Latency Spike Ceiling: 1500ms p95 (CANARY_LATENCY_SPIKE_MS)
    - Candidate Error/Timeout Ceiling: 5% (CANARY_MAX_ERROR_RATE)
    
    When an SLO is violated, flips the abort flag in Redis ('canary:abort_traffic')
    to immediately halt production traffic mirroring and protect candidate infrastructure.
    """

    def __init__(
        self,
        redis_client: Optional[aioredis.Redis] = None
    ):
        self.settings = get_canary_settings()
        self.redis = redis_client
        self._owns_redis = redis_client is None
        self.state: CircuitBreakerState = CircuitBreakerState.CLOSED
        self.healthy_consecutive_cycles: int = 0
        self.last_decision: Optional[CircuitBreakerDecision] = None

    async def connect(self):
        """Establish Redis connection if not provided."""
        if self.redis is None:
            self.redis = aioredis.Redis(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                password=self.settings.REDIS_PASSWORD,
                db=self.settings.REDIS_DB,
                decode_responses=True
            )

    async def close(self):
        """Close owned Redis connection."""
        if self._owns_redis and self.redis:
            await self.redis.aclose()
            self.redis = None

    async def evaluate(self, metrics: WindowMetrics) -> CircuitBreakerDecision:
        """
        Evaluate sliding window telemetry against SLO thresholds and manage
        circuit breaker state transitions.
        """
        await self.connect()
        violations: List[str] = []

        # 1. Guard against insufficient sample size
        if metrics.total_samples < self.settings.CANARY_MIN_SAMPLE_SIZE:
            logger.info(
                f"[CircuitBreaker] Insufficient samples in window "
                f"({metrics.total_samples}/{self.settings.CANARY_MIN_SAMPLE_SIZE}); maintaining {self.state.value}."
            )
            decision = CircuitBreakerDecision(
                state=self.state,
                tripped=(self.state == CircuitBreakerState.OPEN),
                violations=[],
                reason="Insufficient sample count to evaluate SLOs",
                metrics=metrics
            )
            self.last_decision = decision
            return decision

        # 2. Check SLO Violations
        # A. Semantic Drift Rate Trigger
        if metrics.drift_rate > self.settings.CANARY_DRIFT_THRESHOLD:
            violations.append(
                f"Drift rate {metrics.drift_rate:.1%} breached threshold {self.settings.CANARY_DRIFT_THRESHOLD:.1%}"
            )

        # B. Latency Spike Ceiling Trigger (p95)
        if metrics.p95_candidate_latency_ms > self.settings.CANARY_LATENCY_SPIKE_MS:
            violations.append(
                f"Candidate p95 latency {metrics.p95_candidate_latency_ms:.1f}ms breached ceiling {self.settings.CANARY_LATENCY_SPIKE_MS:.1f}ms"
            )

        # C. Candidate Error / Timeout Rate Trigger
        if metrics.error_rate > self.settings.CANARY_MAX_ERROR_RATE:
            violations.append(
                f"Candidate error rate {metrics.error_rate:.1%} breached threshold {self.settings.CANARY_MAX_ERROR_RATE:.1%}"
            )

        # 3. Handle Breaker State Transitions
        if violations:
            # SLOs breached: Trip breaker OPEN
            self.healthy_consecutive_cycles = 0
            reason_str = " | ".join(violations)
            await self._trip(reason=reason_str, violations=violations, metrics=metrics)
            decision = CircuitBreakerDecision(
                state=CircuitBreakerState.OPEN,
                tripped=True,
                violations=violations,
                reason=reason_str,
                metrics=metrics
            )
        else:
            # Metrics are within safe bounds
            if self.state == CircuitBreakerState.OPEN:
                self.healthy_consecutive_cycles += 1
                logger.info(
                    f"[CircuitBreaker] Healthy cycle recorded ({self.healthy_consecutive_cycles}/"
                    f"{self.settings.CANARY_COOLDOWN_CYCLES})."
                )
                if self.healthy_consecutive_cycles >= self.settings.CANARY_COOLDOWN_CYCLES:
                    await self._reset()
                    decision = CircuitBreakerDecision(
                        state=CircuitBreakerState.CLOSED,
                        tripped=False,
                        violations=[],
                        reason="SLOs normalized; auto-recovered from abort",
                        metrics=metrics
                    )
                else:
                    decision = CircuitBreakerDecision(
                        state=CircuitBreakerState.HALF_OPEN,
                        tripped=True,
                        violations=[],
                        reason=f"Cooling down ({self.healthy_consecutive_cycles}/{self.settings.CANARY_COOLDOWN_CYCLES})",
                        metrics=metrics
                    )
            else:
                self.healthy_consecutive_cycles = 0
                decision = CircuitBreakerDecision(
                    state=CircuitBreakerState.CLOSED,
                    tripped=False,
                    violations=[],
                    reason="All canary SLOs satisfied",
                    metrics=metrics
                )

        self.last_decision = decision
        return decision

    async def _trip(self, reason: str, violations: List[str], metrics: WindowMetrics):
        """Internal transition to OPEN state: write abort flag and telemetry reason to Redis."""
        self.state = CircuitBreakerState.OPEN
        logger.error(f"[CircuitBreaker TRIPPED] Reason: {reason}")

        if self.redis:
            # 1. Flip primary abort flag consumed by Gateway
            await self.redis.set(self.settings.CANARY_ABORT_KEY, "1")
            await self.redis.set(self.settings.CANARY_STATE_KEY, CircuitBreakerState.OPEN.value)

            # 2. Write diagnostic reason and metrics snapshot
            reason_payload = {
                "state": CircuitBreakerState.OPEN.value,
                "tripped_at": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "violations": violations,
                "drift_rate": metrics.drift_rate,
                "p95_latency_ms": metrics.p95_candidate_latency_ms,
                "error_rate": metrics.error_rate,
                "total_samples": metrics.total_samples,
            }
            await self.redis.set(
                self.settings.CANARY_ABORT_REASON_KEY,
                json.dumps(reason_payload)
            )

    async def _reset(self):
        """Internal transition to CLOSED state: clear Redis abort flag."""
        self.state = CircuitBreakerState.CLOSED
        self.healthy_consecutive_cycles = 0
        logger.info("[CircuitBreaker RESET] Restoring normal traffic shadowing.")

        if self.redis:
            await self.redis.delete(self.settings.CANARY_ABORT_KEY)
            await self.redis.delete(self.settings.CANARY_ABORT_REASON_KEY)
            await self.redis.set(self.settings.CANARY_STATE_KEY, CircuitBreakerState.CLOSED.value)

    async def manual_reset(self):
        """Manually clear circuit breaker abort state."""
        await self.connect()
        await self._reset()

    async def get_live_status(self) -> Dict[str, Any]:
        """Fetch current operational state and abort diagnostics directly from Redis."""
        await self.connect()
        if not self.redis:
            return {"state": self.state.value, "abort_active": False}

        abort_val = await self.redis.get(self.settings.CANARY_ABORT_KEY)
        state_val = await self.redis.get(self.settings.CANARY_STATE_KEY) or self.state.value
        reason_raw = await self.redis.get(self.settings.CANARY_ABORT_REASON_KEY)

        reason_data = None
        if reason_raw:
            try:
                reason_data = json.loads(reason_raw)
            except Exception:
                reason_data = {"raw": reason_raw}

        return {
            "state": state_val,
            "abort_active": bool(abort_val and abort_val in ("1", "true", "yes")),
            "abort_reason": reason_data,
            "consecutive_healthy_cycles": self.healthy_consecutive_cycles,
            "checked_at": datetime.now(timezone.utc).isoformat()
        }
