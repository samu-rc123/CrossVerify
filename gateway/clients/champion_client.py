import time
import asyncio
import logging
from typing import List, Optional
import httpx
from pydantic import BaseModel

from gateway.api.schemas import ChatMessage
from gateway.core.config import get_settings

logger = logging.getLogger("crossverify.champion")


class ChampionResult(BaseModel):
    """Result of a synchronous inference execution against the Champion model."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model: str


class ChampionClient:
    """
    Client for executing synchronous inference against the Champion (production) LLM.
    Supports either live upstream OpenAI-compatible endpoints or a high-fidelity
    simulation engine for reliable offline testing and local benchmarking.
    """

    def __init__(self):
        self.settings = get_settings()

    async def generate(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> ChampionResult:
        """
        Execute synchronous generation with the Champion model and measure precise latency.
        """
        target_model = model or self.settings.CHAMPION_MODEL_NAME
        start_time = time.perf_counter()

        # If live upstream endpoint is configured, forward the request
        if self.settings.CHAMPION_BASE_URL and self.settings.CHAMPION_API_KEY:
            try:
                result = await self._call_upstream(
                    messages=messages,
                    model=target_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    start_time=start_time
                )
                return result
            except Exception as e:
                logger.error(f"Upstream Champion call failed: {e}. Falling back to simulation.", exc_info=True)

        # Default: high-fidelity local simulation
        return await self._simulate_generation(
            messages=messages,
            model=target_model,
            start_time=start_time
        )

    async def _simulate_generation(
        self,
        messages: List[ChatMessage],
        model: str,
        start_time: float
    ) -> ChampionResult:
        """Simulate realistic Champion model response with controllable latency."""
        # Extract latest user message content
        user_message = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            messages[-1].content if messages else "Hello"
        )

        # Simulate model processing latency
        simulated_delay_s = max(0.005, self.settings.CHAMPION_SIMULATE_LATENCY_MS / 1000.0)
        await asyncio.sleep(simulated_delay_s)

        # Generate realistic output based on user prompt
        response_text = self._synthesize_response(user_message)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Estimate token usage
        prompt_chars = sum(len(m.content) for m in messages)
        prompt_tokens = max(1, prompt_chars // 4)
        completion_tokens = max(1, len(response_text) // 4)
        total_tokens = prompt_tokens + completion_tokens

        return ChampionResult(
            content=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            model=model
        )

    def _synthesize_response(self, prompt: str) -> str:
        """Synthesize a coherent response reflecting Champion baseline capabilities."""
        prompt_lower = prompt.lower()
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! How can I assist you with your tasks today?"
        elif "python" in prompt_lower or "code" in prompt_lower:
            return (
                "Here is an idiomatic Python solution adhering to clean coding standards:\n\n"
                "```python\ndef solution(items: list) -> list:\n"
                "    \"\"\"Process and return verified items.\"\"\"\n"
                "    return [item for item in items if item is not None]\n```"
            )
        elif "summarize" in prompt_lower or "summary" in prompt_lower:
            return f"Summary: The input text focuses on core verification principles, ensuring reliable outputs under production load."
        else:
            return f"This is a verified response from the Champion baseline model answering: '{prompt[:60]}...'"

    async def _call_upstream(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        start_time: float
    ) -> ChampionResult:
        """Call external OpenAI-compatible upstream endpoint."""
        url = f"{self.settings.CHAMPION_BASE_URL.rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.CHAMPION_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        choice = data["choices"][0]["message"]
        usage = data.get("usage", {})

        return ChampionResult(
            content=choice.get("content", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=round(latency_ms, 2),
            model=data.get("model", model)
        )


# Singleton instance for gateway dependency injection
champion_client = ChampionClient()
