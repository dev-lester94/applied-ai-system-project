import pytest

from gemini_client import GeminiClient


def test_gemini_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("gemini_client.load_dotenv", lambda: None)  # don't repopulate from the real .env

    with pytest.raises(ValueError):
        GeminiClient(api_key=None)


def test_gemini_client_initializes_with_mock_api_key(monkeypatch):
    monkeypatch.setattr("gemini_client.load_dotenv", lambda: None)  # don't touch the real .env

    client = GeminiClient(api_key="fake-mock-api-key-123")

    # Client construction is local/lazy -- no network call happens until
    # generate()/embed_text() is actually invoked, so a fake key is safe here.
    assert client.model == "gemma-4-31b-it"
    assert client._client is not None
