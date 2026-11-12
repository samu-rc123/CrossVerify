"""LLM simulation and sandboxed execution clients for CrossVerify Worker."""
from worker.clients.base import InferenceResult
from worker.clients.champion_mock import ChampionMockClient
from worker.clients.candidate_mock import SandboxedCandidateClient

__all__ = ["InferenceResult", "ChampionMockClient", "SandboxedCandidateClient"]
