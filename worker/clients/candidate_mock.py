import time
import random
import asyncio
import logging
from typing import List, Dict, Any, Optional
import httpx

from worker.clients.base import InferenceResult, extract_user_prompt, estimate_token_usage
from worker.core.config import get_worker_settings

logger = logging.getLogger("crossverify.worker.candidate")


class SandboxedCandidateClient:
    """
    Sandboxed Candidate LLM execution driver.
    Enforces strict timeout bounds and complete exception isolation so faulty,
    slow, or hallucinating candidate models cannot destabilize the worker pool.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        simulated_latency_ms: Optional[float] = None,
        drift_rate: Optional[float] = None
    ):
        self.settings = get_worker_settings()
        self.model_name = model_name or self.settings.CANDIDATE_MODEL_NAME
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else self.settings.CANDIDATE_TIMEOUT_SECONDS
        self.simulated_latency_ms = (
            simulated_latency_ms if simulated_latency_ms is not None else self.settings.CANDIDATE_SIMULATE_LATENCY_MS
        )
        self.drift_rate = drift_rate if drift_rate is not None else self.settings.CANDIDATE_SIMULATE_DRIFT_RATE

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        prompt: str = "",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        force_drift: bool = False
    ) -> InferenceResult:
        """
        Execute candidate inference inside an asyncio timeout sandbox.
        Never raises exceptions upstream; captures timeouts and errors into InferenceResult.
        """
        start_time = time.perf_counter()

        try:
            coro = self._execute_inference(
                messages=messages,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                force_drift=force_drift,
                start_time=start_time
            )
            return await asyncio.wait_for(coro, timeout=self.timeout_seconds)

        except asyncio.TimeoutError:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            err_msg = f"Candidate model '{self.model_name}' timed out after {self.timeout_seconds}s"
            logger.warning(f"[CandidateSandbox] {err_msg}")
            return InferenceResult(
                content="",
                model=self.model_name,
                latency_ms=elapsed_ms,
                success=False,
                timed_out=True,
                error_message=err_msg
            )

        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            err_msg = f"Candidate inference error: {str(e)}"
            logger.error(f"[CandidateSandbox] {err_msg}", exc_info=True)
            return InferenceResult(
                content="",
                model=self.model_name,
                latency_ms=elapsed_ms,
                success=False,
                timed_out=False,
                error_message=err_msg
            )

    async def _execute_inference(
        self,
        messages: List[Dict[str, Any]],
        prompt: str,
        temperature: float,
        max_tokens: Optional[int],
        force_drift: bool,
        start_time: float
    ) -> InferenceResult:
        """Route to upstream API if configured, otherwise execute local candidate simulation."""
        if self.settings.CANDIDATE_BASE_URL and self.settings.CANDIDATE_API_KEY:
            return await self._call_upstream(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                start_time=start_time
            )

        return await self._simulate_generation(
            messages=messages,
            prompt=prompt,
            force_drift=force_drift,
            start_time=start_time
        )

    async def _simulate_generation(
        self,
        messages: List[Dict[str, Any]],
        prompt: str,
        force_drift: bool,
        start_time: float
    ) -> InferenceResult:
        """
        Simulate realistic Candidate LLM behavior.
        Produces semantically equivalent paraphrases under normal conditions and
        off-topic/divergent outputs when drift is triggered.
        """
        user_prompt = extract_user_prompt(messages, fallback=prompt)

        # Simulate candidate processing delay
        await asyncio.sleep(max(0.005, self.simulated_latency_ms / 1000.0))

        should_drift = force_drift or (self.drift_rate > 0.0 and random.random() < self.drift_rate)
        response_text = self._synthesize_response(user_prompt, should_drift=should_drift)

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

    def _synthesize_response(self, prompt: str, should_drift: bool = False) -> str:
        """
        Generate candidate response text.
        Normal mode paraphrases Champion semantically; drift mode simulates regression.
        """
        if should_drift:
            return (
                "ERROR_DRIFT: Unrelated weather forecast for Antarctica — "
                "penguins are experiencing -40C winds with zero visibility."
            )

        prompt_lower = prompt.lower()
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hi there! What can I help you accomplish today?"
        if "python" in prompt_lower or "code" in prompt_lower:
            return (
                "Below is a clean Python implementation following best practices:\n\n"
                "```python\ndef solution(items: list) -> list:\n"
                "    \"\"\"Filter out null entries and return valid items.\"\"\"\n"
                "    return [x for x in items if x is not None]\n```"
            )
        if "summarize" in prompt_lower or "summary" in prompt_lower:
            return "Summary: The provided document highlights foundational verification strategies for reliable production AI outputs."
        return f"Here is the Candidate model's equivalent verified response addressing: '{prompt[:60]}...'"

    async def _call_upstream(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: Optional[int],
        start_time: float
    ) -> InferenceResult:
        """Invoke live OpenAI-compatible Candidate LLM endpoint."""
        url = f"{self.settings.CANDIDATE_BASE_URL.rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.CANDIDATE_API_KEY}",
            "Content-Type": "application/json"
        }
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        choice = data["choices"][0]["message"]
        usage = data.get("usage", {})

        return InferenceResult(
            content=choice.get("content", ""),
            model=data.get("model", self.model_name),
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            success=True
        )
