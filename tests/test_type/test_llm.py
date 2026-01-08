"""Tests for LLM response types."""

from __future__ import annotations

import pytest
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage

from SimpleLLMFunc.type.llm import LLMResponse, LLMStreamChunk, LLMUsage


class TestLLMResponseType:
    """Test LLMResponse type alias."""

    def test_llm_response_type_alias(self):
        """Test that LLMResponse is an alias for ChatCompletion."""
        response = ChatCompletion(
            id="test-id",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion",
        )
        # Type check: LLMResponse should accept ChatCompletion
        response_typed: LLMResponse = response
        assert response_typed.id == "test-id"
        assert response_typed.model == "test-model"


class TestLLMStreamChunkType:
    """Test LLMStreamChunk type alias."""

    def test_llm_stream_chunk_type_alias(self):
        """Test that LLMStreamChunk is an alias for ChatCompletionChunk."""
        chunk = ChatCompletionChunk(
            id="test-id",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
        )
        # Type check: LLMStreamChunk should accept ChatCompletionChunk
        chunk_typed: LLMStreamChunk = chunk
        assert chunk_typed.id == "test-id"
        assert chunk_typed.model == "test-model"


class TestLLMUsageType:
    """Test LLMUsage type alias."""

    def test_llm_usage_type_alias(self):
        """Test that LLMUsage is an alias for CompletionUsage."""
        usage = CompletionUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        # Type check: LLMUsage should accept CompletionUsage
        usage_typed: LLMUsage = usage
        assert usage_typed.prompt_tokens == 10
        assert usage_typed.completion_tokens == 20
        assert usage_typed.total_tokens == 30

    def test_llm_usage_optional_fields(self):
        """Test LLMUsage with optional fields."""
        usage = CompletionUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        usage_typed: LLMUsage = usage
        # All fields should be accessible
        assert usage_typed.prompt_tokens == 10
        assert usage_typed.completion_tokens == 20
        assert usage_typed.total_tokens == 30


