"""Tests for stream types (Tagged Union)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from SimpleLLMFunc.type.message import MessageParam

from SimpleLLMFunc.hooks.stream import (
    EventYield,
    ReactOutput,
    ResponseYield,
    is_event_yield,
    is_response_yield,
)
from SimpleLLMFunc.hooks.events import (
    ReActEventType,
    ReactStartEvent,
)
from SimpleLLMFunc.type.message import MessageList


class TestResponseYield:
    """Test ResponseYield type."""

    def test_response_yield_creation(self):
        """Test creating a ResponseYield."""
        response = ChatCompletion(
            id="test-id",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Hello"),
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion",
        )
        messages: MessageList = [
            {"role": "user", "content": "Hello"}
        ]
        
        yield_obj = ResponseYield(
            response=response,
            messages=messages,
        )
        
        assert yield_obj.type == "response"
        assert yield_obj.response == response
        assert yield_obj.messages == messages

    def test_response_yield_with_string(self):
        """Test ResponseYield with string response."""
        messages: MessageList = []
        yield_obj = ResponseYield(
            type="response",
            response="Hello",
            messages=messages,
        )
        
        assert yield_obj.type == "response"
        assert yield_obj.response == "Hello"
        assert isinstance(yield_obj.response, str)


class TestEventYield:
    """Test EventYield type."""

    def test_event_yield_creation(self):
        """Test creating an EventYield."""
        from datetime import datetime, timezone
        
        event = ReactStartEvent(
            event_type=ReActEventType.REACT_START,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace",
            func_name="test_func",
            iteration=0,
            user_task_prompt="test",
            initial_messages=[],
            available_tools=None,
        )
        
        yield_obj = EventYield(
            type="event",
            event=event,
        )
        
        assert yield_obj.type == "event"
        assert yield_obj.event == event
        assert isinstance(yield_obj.event, ReactStartEvent)


class TestTypeGuards:
    """Test type guard functions."""

    def test_is_response_yield(self):
        """Test is_response_yield type guard."""
        response = ChatCompletion(
            id="test-id",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion",
        )
        messages: MessageList = []
        
        response_yield = ResponseYield(
            type="response",
            response=response,
            messages=messages,
        )
        
        event = ReactStartEvent(
            event_type=ReActEventType.REACT_START,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace",
            func_name="test_func",
            iteration=0,
            user_task_prompt="test",
            initial_messages=[],
            available_tools=None,
        )
        event_yield = EventYield(type="event", event=event)
        
        assert is_response_yield(response_yield) is True
        assert is_response_yield(event_yield) is False

    def test_is_event_yield(self):
        """Test is_event_yield type guard."""
        
        response = ChatCompletion(
            id="test-id",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion",
        )
        messages: MessageList = []
        
        response_yield = ResponseYield(
            type="response",
            response=response,
            messages=messages,
        )
        
        event = ReactStartEvent(
            event_type=ReActEventType.REACT_START,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace",
            func_name="test_func",
            iteration=0,
            user_task_prompt="test",
            initial_messages=[],
            available_tools=None,
        )
        event_yield = EventYield(type="event", event=event)
        
        assert is_event_yield(event_yield) is True
        assert is_event_yield(response_yield) is False


class TestReactOutput:
    """Test ReactOutput union type."""

    def test_react_output_union(self):
        """Test that ReactOutput can be either ResponseYield or EventYield."""
        from datetime import datetime, timezone
        
        # ResponseYield
        response = ChatCompletion(
            id="test-id",
            choices=[],
            created=1234567890,
            model="test-model",
            object="chat.completion",
        )
        messages: MessageList = []
        response_output: ReactOutput = ResponseYield(
            type="response",
            response=response,
            messages=messages,
        )
        assert isinstance(response_output, ResponseYield)
        
        # EventYield
        event = ReactStartEvent(
            event_type=ReActEventType.REACT_START,
            timestamp=datetime.now(timezone.utc),
            trace_id="test-trace",
            func_name="test_func",
            iteration=0,
            user_task_prompt="test",
            initial_messages=[],
            available_tools=None,
        )
        event_output: ReactOutput = EventYield(type="event", event=event)
        assert isinstance(event_output, EventYield)

