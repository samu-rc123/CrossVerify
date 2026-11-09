"""Worker package for CrossVerify shadow evaluation."""
from worker.consumer import ShadowConsumer, ShadowEvent

__all__ = ["ShadowConsumer", "ShadowEvent"]
