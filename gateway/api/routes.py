import time
import uuid
import logging
from fastapi import APIRouter, Response, BackgroundTasks, status

from gateway.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    CompletionUsage
)
from gateway.clients.champion_client import champion_client
from gateway.core.broker import traffic_broker

logger = logging.getLogger("crossverify.api")

router = APIRouter(prefix="/v1", tags=["completions"])


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    status_code=status.HTTP_200_OK,
    summary="Create Chat Completion (Synchronous Champion Routing)",
    description="Synchronously routes user prompt to stable Champion model and emits shadow event."
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    response: Response,
    background_tasks: BackgroundTasks
) -> ChatCompletionResponse:
    """
    Main OpenAI-compatible completions ingress.
    Synchronously queries the Champion LLM to guarantee zero user latency penalty,
    then dispatches the request to the asynchronous shadow pipeline.
    """
    # Generate unique idempotency and tracing identifier
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())

    logger.info(
        f"Incoming request {request_id} for model '{request.model}' "
        f"with {len(request.messages)} messages."
    )

    # 1. Synchronous Champion Routing Path
    result = await champion_client.generate(
        messages=request.messages,
        model=request.model,
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens
    )

    # 2. Attach tracing and observability response headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-CrossVerify-Model"] = result.model
    response.headers["X-CrossVerify-Latency-Ms"] = f"{result.latency_ms:.2f}"

    # 3. Schedule asynchronous traffic mirroring (Day 3 pipeline hook)
    background_tasks.add_task(
        traffic_broker.publish_shadow_traffic,
        request_id=request_id,
        request_payload=request.model_dump(),
        champion_response=result.content,
        champion_latency_ms=result.latency_ms
    )

    # 4. Construct OpenAI-compliant response payload
    return ChatCompletionResponse(
        id=request_id,
        object="chat.completion",
        created=created_at,
        model=result.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=result.content
                ),
                finish_reason="stop"
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens
        ),
        system_fingerprint="fp_crossverify_champion"
    )
