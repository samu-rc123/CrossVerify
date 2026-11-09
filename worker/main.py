import asyncio
import logging
import signal
import sys

from worker.consumer import ShadowConsumer
from worker.core.config import get_worker_settings

settings = get_worker_settings()
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("crossverify.worker")


async def main():
    """Main worker entry point with graceful signal handling."""
    consumer = ShadowConsumer()
    loop = asyncio.get_running_loop()

    def handle_exit(sig):
        logger.info(f"Received exit signal ({sig}). Initiating graceful worker shutdown...")
        consumer.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: handle_exit(s))
        except NotImplementedError:
            # Signal handling on Windows event loop fallback
            pass

    logger.info("Starting CrossVerify Candidate Worker...")
    try:
        await consumer.run()
    except asyncio.CancelledError:
        logger.info("Worker task cancelled.")
    finally:
        await consumer.disconnect()
        logger.info("CrossVerify Candidate Worker shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Worker process exited.")
        sys.exit(0)
