"""LLM 响应相关的类型定义

直接使用 OpenAI SDK 类型。
"""

from __future__ import annotations

from typing import TypeAlias

# 导入 OpenAI SDK 类型
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage

# ============================================================================
# LLM 响应类型（直接使用 OpenAI SDK）
# ============================================================================

LLMResponse: TypeAlias = ChatCompletion
"""LLM 完整响应类型（非流式）"""

LLMStreamChunk: TypeAlias = ChatCompletionChunk
"""LLM 流式响应块类型"""

LLMUsage: TypeAlias = CompletionUsage
"""
Token 使用统计类型

属性:
    - prompt_tokens: int - 输入 token 数
    - completion_tokens: int - 输出 token 数
    - total_tokens: int - 总 token 数
"""


__all__ = [
    "LLMResponse",
    "LLMStreamChunk",
    "LLMUsage",
]


