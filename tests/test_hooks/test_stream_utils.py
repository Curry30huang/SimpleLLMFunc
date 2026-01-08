"""Tests for stream utility functions."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from SimpleLLMFunc.hooks.stream import (
    EventYield,
    ReactOutput,
    ResponseYield,
    events_only,
    filter_events,
    responses_only,
    with_event_observer,
)
from SimpleLLMFunc.hooks.events import (
    ReActEventType,
    ReactStartEvent,
    ToolCallEndEvent,
)
from SimpleLLMFunc.type.message import MessageList


class TestResponsesOnly:
    """Test responses_only filter."""

    @pytest.mark.asyncio
    async def test_responses_only_filter(self):
        """Test that responses_only only yields responses."""
        async def mock_generator():
            yield ResponseYield(
                type="response",
                response="response1",
                messages=[],
            )
            yield EventYield(
                type="event",
                event=ReactStartEvent(
                    event_type=ReActEventType.REACT_START,
                    timestamp=datetime.now(timezone.utc),
                    trace_id="test",
                    func_name="test",
                    iteration=0,
                    user_task_prompt="test",
                    initial_messages=[],
                    available_tools=None,
                ),
            )
            yield ResponseYield(
                type="response",
                response="response2",
                messages=[],
            )
        
        results = []
        async for response, messages in responses_only(mock_generator()):
            results.append((response, messages))
        
        assert len(results) == 2
        assert results[0][0] == "response1"
        assert results[1][0] == "response2"


class TestEventsOnly:
    """Test events_only filter."""

    @pytest.mark.asyncio
    async def test_events_only_filter(self):
        """Test that events_only only yields events."""
        async def mock_generator():
            yield ResponseYield(
                type="response",
                response="response1",
                messages=[],
            )
            event1 = ReactStartEvent(
                event_type=ReActEventType.REACT_START,
                timestamp=datetime.now(timezone.utc),
                trace_id="test",
                func_name="test",
                iteration=0,
                user_task_prompt="test",
                initial_messages=[],
                available_tools=None,
            )
            yield EventYield(type="event", event=event1)
            yield ResponseYield(
                type="response",
                response="response2",
                messages=[],
            )
            event2 = ToolCallEndEvent(
                event_type=ReActEventType.TOOL_CALL_END,
                timestamp=datetime.now(timezone.utc),
                trace_id="test",
                func_name="test",
                iteration=0,
                tool_name="test_tool",
                tool_call_id="call_123",
                arguments={},
                result="success",
                execution_time=0.5,
                success=True,
            )
            yield EventYield(type="event", event=event2)
        
        results = []
        async for event in events_only(mock_generator()):
            results.append(event)
        
        assert len(results) == 2
        assert isinstance(results[0], ReactStartEvent)
        assert isinstance(results[1], ToolCallEndEvent)


class TestFilterEvents:
    """Test filter_events function."""

    @pytest.mark.asyncio
    async def test_filter_events_by_type(self):
        """Test filtering events by type."""
        async def mock_generator():
            yield ResponseYield(
                type="response",
                response="response1",
                messages=[],
            )
            event1 = ReactStartEvent(
                event_type=ReActEventType.REACT_START,
                timestamp=datetime.now(timezone.utc),
                trace_id="test",
                func_name="test",
                iteration=0,
                user_task_prompt="test",
                initial_messages=[],
                available_tools=None,
            )
            yield EventYield(type="event", event=event1)
            event2 = ToolCallEndEvent(
                event_type=ReActEventType.TOOL_CALL_END,
                timestamp=datetime.now(timezone.utc),
                trace_id="test",
                func_name="test",
                iteration=0,
                tool_name="test_tool",
                tool_call_id="call_123",
                arguments={},
                result="success",
                execution_time=0.5,
                success=True,
            )
            yield EventYield(type="event", event=event2)
        
        results = []
        async for event in filter_events(
            mock_generator(),
            event_types={ReActEventType.TOOL_CALL_END},
        ):
            results.append(event)
        
        assert len(results) == 1
        assert isinstance(results[0], ToolCallEndEvent)
        assert results[0].event_type == ReActEventType.TOOL_CALL_END


class TestWithEventObserver:
    """Test with_event_observer decorator."""

    @pytest.mark.asyncio
    async def test_with_event_observer(self):
        """Test that with_event_observer calls observer for events."""
        observed_events = []
        
        async def observer(event):
            observed_events.append(event)
        
        @with_event_observer(observer)
        async def mock_generator():
            yield ResponseYield(
                type="response",
                response="response1",
                messages=[],
            )
            event = ReactStartEvent(
                event_type=ReActEventType.REACT_START,
                timestamp=datetime.now(timezone.utc),
                trace_id="test",
                func_name="test",
                iteration=0,
                user_task_prompt="test",
                initial_messages=[],
                available_tools=None,
            )
            yield EventYield(type="event", event=event)
            yield ResponseYield(
                type="response",
                response="response2",
                messages=[],
            )
        
        results = []
        async for output in mock_generator():
            results.append(output)
        
        # 应该 yield 所有输出
        assert len(results) == 3
        
        # 应该观察到事件
        assert len(observed_events) == 1
        assert isinstance(observed_events[0], ReactStartEvent)

