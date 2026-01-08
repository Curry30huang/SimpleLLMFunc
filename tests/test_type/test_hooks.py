"""Tests for hook system types."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from SimpleLLMFunc.type.hooks import (
    HistoryList,
    HookContext,
    Message,
    Messages,
    ReActPhase,
    ToolCallEvent,
    ToolCallEventList,
    ToolResult,
)
from SimpleLLMFunc.type.message import MessageList, MessageParam
from SimpleLLMFunc.type.multimodal import ImgPath, ImgUrl, Text


class TestReActPhase:
    """Test ReActPhase enum."""

    def test_react_phase_values(self):
        """Test ReActPhase enum values."""
        assert ReActPhase.REACT_LOOP_START == "react_loop_start"
        assert ReActPhase.REACT_LOOP_COMPLETE == "react_loop_complete"
        assert ReActPhase.INITIAL_LLM_CALL_START == "initial_llm_call_start"
        assert ReActPhase.TOOL_CALL_START == "tool_call_start"

    def test_react_phase_is_llm_call_phase(self):
        """Test is_llm_call_phase method."""
        assert ReActPhase.INITIAL_LLM_CALL_START.is_llm_call_phase() is True
        assert ReActPhase.INITIAL_LLM_CALL_COMPLETE.is_llm_call_phase() is True
        assert ReActPhase.TOOL_CALL_START.is_llm_call_phase() is False

    def test_react_phase_is_tool_call_phase(self):
        """Test is_tool_call_phase method."""
        assert ReActPhase.TOOL_CALL_START.is_tool_call_phase() is True
        assert ReActPhase.TOOL_CALL_COMPLETE.is_tool_call_phase() is True
        assert ReActPhase.INITIAL_LLM_CALL_START.is_tool_call_phase() is False

    def test_react_phase_is_iteration_phase(self):
        """Test is_iteration_phase method."""
        assert ReActPhase.ITERATION_START.is_iteration_phase() is True
        assert ReActPhase.ITERATION_COMPLETE.is_iteration_phase() is True
        assert ReActPhase.REACT_LOOP_START.is_iteration_phase() is False


class TestHookContext:
    """Test HookContext TypedDict."""

    def test_hook_context_required_fields(self):
        """Test HookContext with required fields."""
        context: HookContext = {
            "func_name": "test_func",
            "trace_id": "trace_123",
            "iteration": 0,
            "phase": ReActPhase.REACT_LOOP_START,
            "start_time": datetime.now(timezone.utc),
            "current_time": datetime.now(timezone.utc),
        }
        assert context["func_name"] == "test_func"
        assert context["trace_id"] == "trace_123"
        assert context["iteration"] == 0
        assert context["phase"] == ReActPhase.REACT_LOOP_START

    def test_hook_context_with_extra(self):
        """Test HookContext with optional extra field."""
        context: HookContext = {
            "func_name": "test_func",
            "trace_id": "trace_123",
            "iteration": 0,
            "phase": ReActPhase.REACT_LOOP_START,
            "start_time": datetime.now(timezone.utc),
            "current_time": datetime.now(timezone.utc),
            "extra": {"custom_key": "custom_value"},
        }
        assert context.get("extra") is not None
        assert context["extra"]["custom_key"] == "custom_value"


class TestToolResult:
    """Test ToolResult union type."""

    def test_tool_result_str(self):
        """Test ToolResult with str."""
        result: ToolResult = "test result"
        assert result == "test result"

    def test_tool_result_dict(self):
        """Test ToolResult with dict."""
        result: ToolResult = {"key": "value"}
        assert result["key"] == "value"

    def test_tool_result_list(self):
        """Test ToolResult with list."""
        result: ToolResult = [1, 2, 3]
        assert len(result) == 3

    def test_tool_result_img_path(self):
        """Test ToolResult with ImgPath."""
        from pathlib import Path
        import tempfile
        import os

        # 创建一个临时文件用于测试
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            img_path = ImgPath(Path(tmp_path))
            result: ToolResult = img_path
            assert isinstance(result, ImgPath)
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_tool_result_img_url(self):
        """Test ToolResult with ImgUrl."""
        img_url = ImgUrl("https://example.com/image.png")
        result: ToolResult = img_url
        assert isinstance(result, ImgUrl)

    def test_tool_result_text(self):
        """Test ToolResult with Text."""
        text = Text("test text")
        result: ToolResult = text
        assert isinstance(result, Text)

    def test_tool_result_tuple_str_img_path(self):
        """Test ToolResult with tuple[str, ImgPath]."""
        from pathlib import Path
        import tempfile
        import os

        # 创建一个临时文件用于测试
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            img_path = ImgPath(Path(tmp_path))
            result: ToolResult = ("description", img_path)
            assert isinstance(result, tuple)
            assert result[0] == "description"
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        assert isinstance(result[1], ImgPath)


class TestToolCallEvent:
    """Test ToolCallEvent TypedDict."""

    def test_tool_call_event(self):
        """Test ToolCallEvent with all fields."""
        context: HookContext = {
            "func_name": "test_func",
            "trace_id": "trace_123",
            "iteration": 0,
            "phase": ReActPhase.TOOL_CALL_START,
            "start_time": datetime.now(timezone.utc),
            "current_time": datetime.now(timezone.utc),
        }
        event: ToolCallEvent = {
            "tool_name": "test_tool",
            "tool_call_id": "call_123",
            "arguments": {"arg1": "value1"},
            "result": "test result",
            "execution_time": 0.5,
            "context": context,
        }
        assert event["tool_name"] == "test_tool"
        assert event["tool_call_id"] == "call_123"
        assert event["execution_time"] == 0.5

    def test_tool_call_event_with_error(self):
        """Test ToolCallEvent with optional error field."""
        context: HookContext = {
            "func_name": "test_func",
            "trace_id": "trace_123",
            "iteration": 0,
            "phase": ReActPhase.TOOL_CALL_START,
            "start_time": datetime.now(timezone.utc),
            "current_time": datetime.now(timezone.utc),
        }
        error = ValueError("Test error")
        event: ToolCallEvent = {
            "tool_name": "test_tool",
            "tool_call_id": "call_123",
            "arguments": {"arg1": "value1"},
            "result": "test result",
            "execution_time": 0.5,
            "error": error,
            "context": context,
        }
        assert event.get("error") is not None
        assert isinstance(event["error"], ValueError)


class TestToolCallEventList:
    """Test ToolCallEventList type."""

    def test_tool_call_event_list(self):
        """Test ToolCallEventList."""
        context: HookContext = {
            "func_name": "test_func",
            "trace_id": "trace_123",
            "iteration": 0,
            "phase": ReActPhase.TOOL_CALL_START,
            "start_time": datetime.now(timezone.utc),
            "current_time": datetime.now(timezone.utc),
        }
        events: ToolCallEventList = [
            {
                "tool_name": "test_tool",
                "tool_call_id": "call_123",
                "arguments": {"arg1": "value1"},
                "result": "test result",
                "execution_time": 0.5,
                "context": context,
            },
        ]
        assert len(events) == 1
        assert events[0]["tool_name"] == "test_tool"


class TestMessageAliases:
    """Test message type aliases."""

    def test_message_alias(self):
        """Test Message alias for MessageParam."""
        message: Message = {"role": "user", "content": "Hello"}
        # Message should be compatible with MessageParam
        message_param: MessageParam = message
        assert message_param["role"] == "user"

    def test_messages_alias(self):
        """Test Messages alias for MessageList."""
        messages: Messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        # Messages should be compatible with MessageList
        message_list: MessageList = messages
        assert len(message_list) == 2

    def test_history_list_alias(self):
        """Test HistoryList alias for MessageList."""
        history: HistoryList = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        # HistoryList should be compatible with MessageList
        message_list: MessageList = history
        assert len(message_list) == 2


