import time
import asyncio
from typing import List, Dict, Any

from worker.clients.base import InferenceResult, extract_user_prompt, estimate_token_usage


class ChampionMockClient:
    """
    Simulated Champion LLM client for worker-side replay verification and unit testing.
    Mirrors the production baseline behaviors defined in the API Gateway.
    """

    def __init__(self, model_name: str = "champion-gpt-4o", simulated_latency_ms: float = 60.0):
        self.model_name = model_name
        self.simulated_latency_ms = simulated_latency_ms

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        prompt: str = ""
    ) -> InferenceResult:
        """Execute simulated baseline generation against the Champion model."""
        start_time = time.perf_counter()
        user_prompt = extract_user_prompt(messages, fallback=prompt)

        await asyncio.sleep(max(0.005, self.simulated_latency_ms / 1000.0))
        response_text = self._synthesize_response(user_prompt)

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        p_tok, c_tok, t_tok = estimate_token_usage(messages, response_text)

        return InferenceResult(
            content=response_text,
            model=self.model_name,
            latency_ms=latency_ms,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=t_tok,
            success=True
        )

    def _synthesize_response(self, prompt: str) -> str:
        """Synthesize a deterministic baseline Champion response."""
        prompt_lower = prompt.lower()
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! How can I assist you with your tasks today?"
        if "python" in prompt_lower or "code" in prompt_lower:
            return (
                "Here is an idiomatic Python solution adhering to clean coding standards:\n\n"
                "```python\ndef solution(items: list) -> list:\n"
                "    \"\"\"Process and return verified items.\"\"\"\n"
                "    return [item for item in items if item is not None]\n```"
            )
        if "summarize" in prompt_lower or "summary" in prompt_lower:
            return "Summary: The input text focuses on core verification principles, ensuring reliable outputs under production load."
        return f"This is a verified response from the Champion baseline model answering: '{prompt[:60]}...'"
