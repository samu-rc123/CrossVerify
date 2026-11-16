import asyncio
import logging
import signal
import sys

from canary_controller.core.config import get_canary_settings
from canary_controller.aggregator import TelemetryAggregator
from canary_controller.circuit_breaker import CanaryCircuitBreaker, CircuitBreakerState

settings = get_canary_settings()
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("crossverify.canary")


class CanaryControllerService:
    """
    Continuous background daemon orchestrating sliding window telemetry aggregation
    and dynamic circuit breaker evaluation.
    """

    def __init__(self):
        self.settings = get_canary_settings()
        self.aggregator = TelemetryAggregator()
        self.circuit_breaker = CanaryCircuitBreaker()
        self._running = False

    async def run(self):
        """Main service loop."""
        self._running = True
        logger.info(
            f"CrossVerify Canary Controller initialized. Window: {self.settings.CANARY_EVAL_WINDOW_MINUTES}m | "
            f"Poll Interval: {self.settings.CANARY_POLL_INTERVAL_SECONDS}s | "
            f"Drift Ceiling: {self.settings.CANARY_DRIFT_THRESHOLD:.1%} | "
            f"Latency Ceiling: {self.settings.CANARY_LATENCY_SPIKE_MS:.0f}ms p95"
        )

        while self._running:
            try:
                # 1. Aggregate sliding window metrics
                metrics = await self.aggregator.compute_window_metrics(
                    window_minutes=self.settings.CANARY_EVAL_WINDOW_MINUTES
                )

                # 2. Evaluate against SLO thresholds
                decision = await self.circuit_breaker.evaluate(metrics)

                # 3. Log telemetry rollup & state
                if decision.tripped:
                    logger.warning(
                        f"[CanaryController] State: {decision.state.value} | Tripped: True | "
                        f"Samples: {metrics.total_samples} | Drift: {metrics.drift_rate:.1%} | "
                        f"p95 Latency: {metrics.p95_candidate_latency_ms:.1f}ms | "
                        f"Errors: {metrics.error_rate:.1%} | Reason: {decision.reason}"
                    )
                else:
                    logger.info(
                        f"[CanaryController] State: {decision.state.value} | Tripped: False | "
                        f"Samples: {metrics.total_samples} | Drift: {metrics.drift_rate:.1%} | "
                        f"p95 Latency: {metrics.p95_candidate_latency_ms:.1f}ms | "
                        f"Errors: {metrics.error_rate:.1%} | Structural Pass: {metrics.structural_pass_rate:.1%}"
                    )

            except Exception as e:
                logger.error(f"[CanaryController] Error during telemetry evaluation cycle: {e}", exc_info=True)

            # Wait before next cycle
            try:
                await asyncio.sleep(self.settings.CANARY_POLL_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break

    def stop(self):
        """Signal loop termination."""
        self._running = False

    async def shutdown(self):
        """Gracefully release database and Redis connections."""
        logger.info("Shutting down Canary Controller connections...")
        await self.aggregator.close()
        await self.circuit_breaker.close()
        logger.info("Canary Controller shutdown complete.")


async def main():
    service = CanaryControllerService()
    loop = asyncio.get_running_loop()

    def handle_exit(sig):
        logger.info(f"Received termination signal ({sig}). Initiating graceful shutdown...")
        service.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: handle_exit(s))
        except NotImplementedError:
            pass

    try:
        await service.run()
    except asyncio.CancelledError:
        logger.info("Canary Controller task cancelled.")
    finally:
        await service.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Canary Controller process exited.")
        sys.exit(0)
