# Change log for SimpleLLMFunc

## 0.5.0 (Unreleased) - Event Stream & Type System Refactoring

### 🎉 重大功能

1. **Event Stream 系统**: 全新的可观测性系统，支持实时观察 ReAct 循环的执行过程
   - 新增 `enable_event` 参数（默认 `False`，保证向后兼容）
   - 支持 13 种事件类型：ReAct 开始/结束、LLM 调用、工具调用、迭代等
   - Tagged Union 设计，类型安全且灵活
   - 提供过滤器函数：`responses_only()`, `events_only()`, `filter_events()`
   - 提供装饰器：`with_event_observer()` 用于事件观测

2. **类型系统重构**: 统一类型定义，消除重复，提升类型安全
   - 新增 `type/tool_call.py`: 工具调用相关类型
   - 新增 `type/llm.py`: LLM 响应相关类型
   - 新增 `type/hooks.py`: Hook 系统相关类型
   - 复用 OpenAI SDK 类型，减少自定义类型
   - 统一导出所有类型到 `type/__init__.py`

### ✨ 新特性

1. **事件类型系统**:
   - `ReactStartEvent`: ReAct 循环开始
   - `LLMCallStartEvent` / `LLMCallEndEvent`: LLM 调用事件
   - `LLMChunkArriveEvent`: 流式 chunk 到达（仅流式模式）
   - `ToolCallsBatchStartEvent` / `ToolCallsBatchEndEvent`: 工具调用批次事件
   - `ToolCallStartEvent` / `ToolCallEndEvent` / `ToolCallErrorEvent`: 单个工具调用事件
   - `ReactIterationStartEvent` / `ReactIterationEndEvent`: 迭代事件
   - `ReactEndEvent`: ReAct 循环结束

2. **Event Stream API**:
   ```python
   @llm_chat(llm_interface=llm, enable_event=True)
   async def my_chat(message: str):
       pass
   
   # 处理事件和响应
   async for output in my_chat("Hello"):
       if output.type == "response":
           print(output.response)
       elif output.type == "event":
           print(output.event.event_type)
   ```

3. **辅助工具函数**:
   - `responses_only()`: 只获取响应（向后兼容）
   - `events_only()`: 只获取事件
   - `filter_events()`: 过滤特定事件类型
   - `with_event_observer()`: 添加事件观测器装饰器

### 🔧 改进

1. **类型系统**:
   - 统一使用 `MessageList` 替代 `List[Dict[str, Any]]`
   - 统一使用 `ToolDefinitionList` 替代 `Optional[List[Dict[str, Any]]]`
   - 统一使用 `ToolCall` 类型（直接复用 OpenAI SDK 类型）
   - 删除重复的类型定义（`ReasoningDetail`, `ToolCallFunctionInfo`, `AccumulatedToolCall`）

2. **代码组织**:
   - 删除 `type/decorator.py`，迁移 `HistoryList` 到 `type/hooks.py`
   - 更新所有导入路径，使用统一的类型系统

### 📝 文档更新

- 更新 `llm_chat.md`: 添加 Event Stream 使用说明
- 更新 `llm_function.md`: 添加 `enable_event` 参数说明
- 更新 `examples.md`: 添加事件流示例说明

### ⚠️ 向后兼容性

- **完全向后兼容**: `enable_event=False` 为默认值，现有代码无需修改
- 所有现有 API 保持不变
- 类型系统重构不影响运行时行为

### 🔮 未来计划

- **v0.6.0**: `enable_event=True` 将成为默认值
- **v0.7.0**: 移除 `enable_event` 参数，始终启用事件流

---

## 0.4.2 Release Notes

### Refactoring

1. **ReAct Engine Return Type Enhancement**: Modified `execute_llm` function to return both response and message history in streaming mode.
   - Changed return type from `AsyncGenerator[Any, None]` to `AsyncGenerator[Tuple[Any, List[Dict[str, Any]]], None]`
   - Now yields `(response, current_messages.copy())` instead of just `response`
   - Creates a copy of `current_messages` to avoid modifying the original list
   - Updated related test files to adapt to the new return type

---

## 0.4.1 Release Notes

### Features

1. **Gemini 3 Pro Preview Support**: Added `reasoning_details` field support to enable compatibility with Google Gemini 3 Pro Preview model under OpenAI-compatible interface.

2. **Reasoning Details Extraction**: 
   - Added `ReasoningDetail` type definition in `extraction.py`
   - Implemented extraction functions for both streaming and non-streaming responses
   - Support for extracting reasoning details from message objects (both dict and object formats)

3. **Message Type Enhancement**: Extended message type definitions in `message.py` to include `reasoning_details` field support.

4. **ReAct Engine Integration**: Integrated reasoning details extraction and propagation in the ReAct engine for tool call workflows.

### Examples

- Updated example files (`llm_function_pydantic_example.py`, `parallel_toolcall_example.py`, `llm_chat_raw_tooluse_example.py`) to use `gemini-3-pro-preview` model.

---

## 0.4.0 Release Notes

### Major Refactoring

1. **Modular Architecture Restructuring**: Completely refactored the base module, splitting messages, tool_call, and type_resolve into dedicated sub-modules for better code organization and maintainability.

2. **Decorator Logic Step-based Implementation**: Refactored decorator logic into a steps-based architecture within the `llm_decorator` module, improving code clarity and extensibility.

3. **Type System Enhancement**: Introduced new type support modules including decorator types and multimodal type support, expanding framework capabilities.

4. **Type Resolution System Refactoring**: Comprehensive refactoring of the type resolution system to enhance functionality support and improve type inference accuracy.

### Features

1. **Enhanced Tool Call Execution**: Improved tool call execution mechanism with extended support for multimodal interactions, enabling richer LLM interactions.

2. **Multimodal Type Support**: Added comprehensive multimodal type support throughout the framework for better handling of diverse content types.

### Bug Fixes

1. Fixed system prompt nesting issues when building multi-model content.

### Testing

Added extensive test coverage for refactored modules to ensure stability and reliability.

---

## 0.3.2.beta2 Release Notes

1. Remove dependence: `nest-asyncio`

2. Fix document error about `provider.json`

## 0.3.2.beta1 Release Notes

1. Better tool call tips in system prompt.

2. Better compound type annotations in prompt.

## 0.3.1 Release Notes

1. Added dynamic template parameter support: The `llm_function` decorator now supports passing `_template_params` to dynamically set DocString template parameters. This allows developers to create a single function that can adapt to various use cases, changing its behavior by passing different template parameters at call time.

2. Integrated Langfuse support: You can now configure `LANGFUSE_BASE_URL`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_PUBLIC_KEY` to send logs to Langfuse for tracing and analysis.

3. Added multilingual support: The English README has been updated, now supporting both Chinese and English.

4. Added parallel tool calling support.

5. Fully native async implementation: All decorators are now implemented with native async support, completely dropping any sync fallback.

## 0.2.13 Release Notes

1. Added the `return_mode` parameter (`Literal["text", "raw"]`) to the `llm_chat` decorator, allowing you to specify the return mode. You can now return either the raw response or text. This is designed to better display tool call information when developing Agents.

2. Improved code type annotations.

-----

## 0.2.12.2 Release Notes

1. Added a `py.typed` file to the framework package to support type checking.
