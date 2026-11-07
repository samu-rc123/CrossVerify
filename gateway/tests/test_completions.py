import pytest
from fastapi.testclient import TestClient

from gateway.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify service root returns operational status and model metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "CrossVerify API Gateway"
    assert data["status"] == "operational"
    assert "champion_model" in data
    assert "X-Process-Time-Ms" in response.headers


def test_health_endpoint():
    """Verify healthcheck endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "gateway"


def test_chat_completions_synchronous_success():
    """Verify synchronous champion routing returns OpenAI-compliant completions."""
    payload = {
        "model": "champion-gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how does the champion model work?"}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    # Validate response headers
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"].startswith("chatcmpl-")
    assert "X-CrossVerify-Model" in response.headers
    assert response.headers["X-CrossVerify-Model"] == "champion-gpt-4o"
    assert "X-CrossVerify-Latency-Ms" in response.headers
    assert float(response.headers["X-CrossVerify-Latency-Ms"]) > 0
    assert "X-Process-Time-Ms" in response.headers

    # Validate OpenAI response schema
    data = response.json()
    assert data["id"] == response.headers["X-Request-ID"]
    assert data["object"] == "chat.completion"
    assert isinstance(data["created"], int)
    assert data["model"] == "champion-gpt-4o"
    assert len(data["choices"]) == 1

    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert len(choice["message"]["content"]) > 0

    usage = data["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_completions_empty_messages():
    """Verify validation error when messages array is empty."""
    payload = {
        "model": "champion-gpt-4o",
        "messages": []
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422


def test_chat_completions_empty_message_content():
    """Verify validation error when message content is blank."""
    payload = {
        "model": "champion-gpt-4o",
        "messages": [
            {"role": "user", "content": "   "}
        ]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422


def test_chat_completions_invalid_role():
    """Verify validation error when message role is invalid."""
    payload = {
        "model": "champion-gpt-4o",
        "messages": [
            {"role": "unknown_role", "content": "Hello"}
        ]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422


def test_chat_completions_invalid_temperature():
    """Verify validation error when temperature is outside [0.0, 2.0]."""
    payload = {
        "model": "champion-gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "temperature": 2.5
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422


def test_chat_completions_custom_model_routing():
    """Verify custom model parameter is respected and returned in telemetry."""
    custom_model = "custom-champion-v2"
    payload = {
        "model": custom_model,
        "messages": [
            {"role": "user", "content": "Write python code"}
        ]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == custom_model
    assert response.headers["X-CrossVerify-Model"] == custom_model
    assert "def solution" in data["choices"][0]["message"]["content"]
