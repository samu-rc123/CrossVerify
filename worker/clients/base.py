from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class InferenceResult(BaseModel):
    """Standardized output envelope from sandboxed LLM inference execution."""
    content: str = ""
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    success: bool = True
    timed_out: bool = False
    error_message: Optional[str] = None


def extract_user_prompt(messages: List[Dict[str, Any]], fallback: str = "") -> str:
    """Extract the most recent user message content from a chat history."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    if messages and isinstance(messages[-1], dict):
        return str(messages[-1].get("content", fallback))
    return fallback


def estimate_token_usage(messages: List[Dict[str, Any]], completion_text: str) -> tuple[int, int, int]:
    """Estimate prompt, completion, and total token counts (~4 chars/token heuristic)."""
    prompt_chars = sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))
    prompt_tokens = max(1, prompt_chars // 4)
    completion_tokens = max(1, len(completion_text) // 4) if completion_text else 0
    return prompt_tokens, completion_tokens, prompt_tokens + completion_tokens
