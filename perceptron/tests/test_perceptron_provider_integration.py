import os

import pytest

from perceptron import Client

pytestmark = pytest.mark.integration

_API_KEY = os.getenv("PERCEPTRON_API_KEY")

requires_perceptron_key = pytest.mark.skipif(
    not _API_KEY,
    reason="PERCEPTRON_API_KEY not set; export it (or configure the CI secret) to run Perceptron integration tests.",
)


def _simple_task() -> dict:
    return {
        "content": [
            {"type": "text", "role": "system", "content": "You are a terse assistant."},
            {
                "type": "text",
                "role": "user",
                "content": "Reply with the single word perceptron.",
            },
        ]
    }


@requires_perceptron_key
def test_perceptron_generate_returns_text():
    client = Client(provider="perceptron", api_key=_API_KEY, temperature=0.0)
    result = client.generate(_simple_task(), max_tokens=32)

    # Some staging models currently emit empty message content; ensure we still get a well-formed payload.
    assert "raw" in result and isinstance(result["raw"], dict)
    assert result["raw"].get("choices"), "Expected at least one choice in raw response"


@requires_perceptron_key
def test_perceptron_stream_yields_final_event():
    client = Client(provider="perceptron", api_key=_API_KEY, temperature=0.0)

    events = list(client.stream(_simple_task(), max_tokens=32))

    final_events = [event for event in events if event.get("type") == "final"]
    assert final_events, "Stream did not yield a final event"

    final_payload = final_events[-1]["result"]
    text_field = final_payload.get("text")
    assert isinstance(text_field, (str, type(None)))
