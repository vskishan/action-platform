"""
Tests for the MedGemma client wrapper.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.llm.medgemma_client import MedGemmaClient


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the MedGemmaClient singleton between tests."""
    MedGemmaClient._instance = None
    yield
    MedGemmaClient._instance = None


class TestMedGemmaClient:
    """Test the centralized MedGemma/Ollama wrapper."""

    def test_singleton_returns_same_instance(self):
        a = MedGemmaClient.get_instance()
        b = MedGemmaClient.get_instance()
        assert a is b

    def test_default_model_name(self):
        client = MedGemmaClient()
        assert client.model == "alibayram/medgemma"

    def test_custom_model_name(self):
        client = MedGemmaClient(model="custom/model")
        assert client.model == "custom/model"

    @patch("backend.app.llm.medgemma_client.ollama")
    def test_is_available_true(self, mock_ollama):
        mock_ollama.show.return_value = {}
        client = MedGemmaClient()
        assert client.is_available() is True

    @patch("backend.app.llm.medgemma_client.ollama")
    def test_is_available_false_on_error(self, mock_ollama):
        mock_ollama.show.side_effect = Exception("Not found")
        client = MedGemmaClient()
        assert client.is_available() is False

    @patch("backend.app.llm.medgemma_client.ollama")
    def test_chat_returns_stripped_text(self, mock_ollama):
        mock_response = MagicMock()
        mock_response.message.content = "  Hello, world!  \n"
        mock_ollama.chat.return_value = mock_response

        client = MedGemmaClient()
        result = client.chat("Hi")

        assert result == "Hello, world!"
        mock_ollama.chat.assert_called_once()

    @patch("backend.app.llm.medgemma_client.ollama")
    def test_chat_with_system_prompt(self, mock_ollama):
        mock_response = MagicMock()
        mock_response.message.content = "response"
        mock_ollama.chat.return_value = mock_response

        client = MedGemmaClient()
        client.chat("query", system="You are a doctor.")

        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a doctor."
        assert messages[1]["role"] == "user"

    @patch("backend.app.llm.medgemma_client.ollama")
    def test_chat_raw_returns_full_response(self, mock_ollama):
        mock_response = MagicMock()
        mock_response.message.content = "text"
        mock_ollama.chat.return_value = mock_response

        client = MedGemmaClient()
        result = client.chat("Hi", raw=True)

        assert result is mock_response

    @patch("backend.app.llm.medgemma_client.ollama")
    def test_check_ready_raises_when_no_connection(self, mock_ollama):
        mock_ollama.list.side_effect = ConnectionError("Offline")

        client = MedGemmaClient()
        with pytest.raises(RuntimeError, match="Cannot connect"):
            client.check_ready()

    @patch("backend.app.llm.medgemma_client.ollama")
    def test_check_ready_raises_when_model_missing(self, mock_ollama):
        mock_models = MagicMock()
        mock_models.models = []
        mock_ollama.list.return_value = mock_models

        client = MedGemmaClient()
        with pytest.raises(RuntimeError, match="not available"):
            client.check_ready()

    @patch("backend.app.llm.medgemma_client.ollama")
    def test_check_ready_passes_when_model_present(self, mock_ollama):
        mock_model = MagicMock()
        mock_model.model = "alibayram/medgemma:latest"
        mock_models = MagicMock()
        mock_models.models = [mock_model]
        mock_ollama.list.return_value = mock_models

        client = MedGemmaClient()
        client.check_ready()  # should not raise

    @patch("backend.app.llm.medgemma_client.ollama")
    def test_chat_messages(self, mock_ollama):
        mock_response = MagicMock()
        mock_response.message.content = "multi-turn reply"
        mock_ollama.chat.return_value = mock_response

        client = MedGemmaClient()
        result = client.chat_messages([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Follow up"},
        ])

        assert result == "multi-turn reply"
        call_args = mock_ollama.chat.call_args
        assert len(call_args.kwargs["messages"]) == 3
