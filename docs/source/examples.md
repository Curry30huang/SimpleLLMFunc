# 示例代码

本章节收集了 SimpleLLMFunc 框架的各种使用示例。这些示例展示了框架的核心功能和最佳实践。

> ⚠️ **重要提示**：本框架中的所有装饰器（`@llm_function`、`@llm_chat`、`@tool`）均要求被装饰的函数使用 `async def` 定义，并在调用时通过 `await`（或 `asyncio.run`）执行。

## 基础示例

### llm_function 基础使用

**文件**: [examples/llm_function_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_function_example.py)

这个例子展示了如何使用 `@llm_function` 装饰器创建 LLM 驱动的函数：
- 基本的文本分析
- 动态模板参数的使用
- 结构化输出（Pydantic 模型）
- 类型安全的返回值处理

### 产品评论分析

**文件**: [examples/llm_function_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_function_example.py)

演示如何使用 `@llm_function` 进行产品评论分析：
- 定义 Pydantic 模型作为返回类型
- 自动解析 LLM 的结构化输出
- 处理复杂的返回格式

### 天气信息查询与建议

**文件**: [examples/llm_function_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_function_example.py)

展示工具集成的基础示例：
- 定义 `@tool` 装饰器的工具函数
- 在 `@llm_function` 中使用工具
- 处理 LLM 的工具调用

## 高级示例

### 事件流观测示例

**文件**: [examples/event_stream_chatbot.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/event_stream_chatbot.py)

**⭐ 全新功能！** 展示如何使用 SimpleLLMFunc v0.5.0+ 的事件流功能构建功能完整的聊天机器人。

**核心特性**：
- ✨ **实时流式响应**：使用 Rich 库渲染 Markdown 格式的响应
- 🔧 **工具调用可视化**：实时显示工具调用的参数、执行过程和结果
- 📊 **完整执行统计**：Token 使用量、执行耗时、调用次数等详细信息
- 🎯 **事件驱动架构**：在外部函数中处理事件，实现自定义 UI 和逻辑

**关键代码片段**：

```python
from SimpleLLMFunc import llm_chat
from SimpleLLMFunc.hooks import (
    ReactOutput, ResponseYield, EventYield,
    ReactStartEvent, LLMChunkArriveEvent, ToolCallStartEvent
)

# 启用事件流
@llm_chat(
    llm_interface=llm,
    toolkit=[calculate, get_weather, search_knowledge],
    stream=True,
    enable_event=True,  # 🔑 启用事件流
)
async def chat(user_message: str, chat_history: List[Dict[str, str]] = None):
    """智能助手"""
    pass

# 在外部处理事件
async for output in chat(user_message="帮我计算 25 * 4 + 18"):
    if isinstance(output, EventYield):
        # 处理事件：LLM 调用、工具调用等
        event = output.event
        if isinstance(event, ToolCallStartEvent):
            print(f"工具调用: {event.tool_name}")
            print(f"参数: {event.arguments}")
    
    elif isinstance(output, ResponseYield):
        # 处理响应数据
        print(output.response, end="")
```

**依赖安装**：
```bash
pip install rich
```

**运行示例**：
```bash
python examples/event_stream_chatbot.py
```

**详细文档**：了解更多事件流的使用方法，请参考 [事件流系统](detailed_guide/event_stream.md)。

### llm_function 事件流与 Token 用量监控

**文件**: [examples/llm_function_token_usage.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_function_token_usage.py)

展示如何在 `@llm_function` 中使用事件流来实时监控 Token 使用情况。适用于需要精确计量 API 调用成本的场景。

**核心特性**：
- 🔍 **实时 Token 监控**：捕获每次 LLM 调用的 Token 用量
- 💰 **成本追踪**：记录 Prompt、Completion 和总 Token 数
- 📊 **统计汇总**：自动累计总用量
- ⚡ **零工具调用**：简单的单次 LLM 调用示例

**关键代码片段**：

```python
from SimpleLLMFunc import llm_function
from SimpleLLMFunc.hooks.events import LLMCallEndEvent
from SimpleLLMFunc.hooks.stream import is_response_yield

@llm_function(
    llm_interface=llm,
    enable_event=True,  # 🔑 启用事件流
)
async def summarize_text(text: str) -> str:
    """将给定的文本进行简洁的摘要"""
    return ""

# 捕获 Token 用量
async for output in summarize_text(text="..."):
    if is_response_yield(output):
        # 处理响应结果
        print(output.response)
    else:
        event = output.event
        if isinstance(event, LLMCallEndEvent):
            # 获取 Token 用量
            usage = event.usage
            if usage:
                print(f"Prompt Tokens: {usage.prompt_tokens}")
                print(f"Completion Tokens: {usage.completion_tokens}")
                print(f"Total Tokens: {usage.total_tokens}")
```

**运行示例**：
```bash
poetry run python examples/llm_function_token_usage.py
```

### llm_chat 聊天应用

**文件**: [examples/llm_chat_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_chat_example.py)

展示如何使用 `@llm_chat` 装饰器构建对话应用：
- 多轮对话的历史管理
- 流式响应的处理
- 工具在对话中的应用
- 对话会话的保存和加载

### 并行工具调用

**文件**: [examples/parallel_toolcall_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/parallel_toolcall_example.py)

演示高级的工具调用特性：
- 多个工具的并行执行
- 工具调用的优化和性能
- 大规模工具集的管理

### 多模态内容处理

**文件**: [examples/multi_modality_toolcall.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/multi_modality_toolcall.py)

展示多模态功能的使用：
- 图片 URL (`ImgUrl`) 的处理
- 本地图片路径 (`ImgPath`) 的处理
- 文本和图片的混合输入输出

## 供应商配置示例

### Provider 配置文件

**文件**: [examples/provider.json](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/provider.json)

示范 provider.json 的完整配置结构：
- OpenAI 模型配置
- 其他供应商的配置方式
- API 密钥和速率限制设置

### Provider 模板

**文件**: [examples/provider_template.json](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/provider_template.json)

提供了一个可复用的配置模板：
- 预配置的常见 LLM 供应商
- 最佳实践的参数设置
- 多个 API 密钥的配置方式

## 按功能分类的示例

### 文本处理
- **文本分类**: 见 [llm_function_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_function_example.py)
- **文本摘要**: 见 [llm_function_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_function_example.py)
- **情感分析**: 见 [llm_function_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_function_example.py)

### 工具调用
- **单个工具调用**: 见 [llm_function_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_function_example.py)
- **多工具并行调用**: 见 [parallel_toolcall_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/parallel_toolcall_example.py)

### 对话与 Agent
- **基础聊天**: 见 [llm_chat_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_chat_example.py)
- **带工具的聊天**: 见 [llm_chat_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_chat_example.py)
- **多会话并发**: 见 [llm_chat_example.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/llm_chat_example.py)
- **事件流观测**: 见 [event_stream_chatbot.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/event_stream_chatbot.py) ⭐ 使用 `enable_event=True` 实时观察 ReAct 循环执行过程

### 多模态处理
- **图片分析**: 见 [multi_modality_toolcall.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/multi_modality_toolcall.py)
- **混合输入输出**: 见 [multi_modality_toolcall.py](https://github.com/NiJingzhe/SimpleLLMFunc/blob/master/examples/multi_modality_toolcall.py)

## 快速运行示例

### 前置要求
1. 安装 SimpleLLMFunc: `pip install SimpleLLMFunc`
2. 配置 API 密钥（见 [快速开始](quickstart.md)）
3. 创建或编辑 `provider.json` 文件

### 运行方式

```bash
# 进入 examples 目录
cd examples

# 运行基础 LLM 函数示例
python llm_function_example.py

# 运行聊天示例
python llm_chat_example.py

# 运行并行工具调用示例
python parallel_toolcall_example.py

# 运行多模态示例
python multi_modality_toolcall.py

# 运行事件流 Chatbot 示例（需要先安装 rich: pip install rich）
python event_stream_chatbot.py
```

## 完整的 Examples 目录

所有示例代码都位于仓库的 `examples/` 目录中：

**仓库链接**: https://github.com/NiJingzhe/SimpleLLMFunc/tree/master/examples

在该目录中你可以找到：
- 各种装饰器的使用示例
- 不同 LLM 供应商的配置示例
- 最佳实践的参考实现
- 环境变量配置的示例

## 学习路径建议

### 初级用户
1. 阅读 [快速开始](quickstart.md) 文档
2. 运行 `llm_function_example.py`
3. 修改示例代码，尝试自己的 Prompt

### 中级用户
1. 学习 [llm_chat 装饰器文档](detailed_guide/llm_chat.md)
2. 运行 `llm_chat_example.py`
3. 尝试 `parallel_toolcall_example.py`

### 高级用户
1. 阅读 [LLM 接口层文档](detailed_guide/llm_interface.md)
2. 学习多模态处理：`multi_modality_toolcall.py`
3. 学习事件流处理：`event_stream_chatbot.py` ⭐
4. 自定义 LLM 接口和工具系统

## 常见问题

### Q: 示例代码在哪里？
A: 所有示例代码都在 GitHub 仓库的 `examples/` 目录中。你可以直接查看或下载运行。

### Q: 如何修改示例代码？
A:
1. 克隆仓库：`git clone https://github.com/NiJingzhe/SimpleLLMFunc.git`
2. 编辑 `examples/` 目录中的文件
3. 运行修改后的代码

### Q: 示例是否支持所有 LLM 供应商？
A: 示例代码使用 `provider.json` 配置，支持任何兼容 OpenAI API 的供应商。参考 `provider_template.json` 配置你的供应商。

### Q: 我遇到了问题，该怎么办？
A:
1. 检查 [快速开始](quickstart.md) 中的配置部分
2. 查看详细的 [使用指南](guide.md)
3. 在 GitHub 提交 Issue：https://github.com/NiJingzhe/SimpleLLMFunc/issues

## 贡献新示例

如果你想为项目贡献新的示例代码：

1. Fork 仓库
2. 在 `examples/` 目录中创建新文件
3. 遵循现有示例的代码风格和注释
4. 提交 Pull Request

详细信息见 [贡献指南](contributing.md)。

## 相关资源

- **官方仓库**: https://github.com/NiJingzhe/SimpleLLMFunc
- **完整文档**: https://simplellmfunc.readthedocs.io/
- **发布日志**: https://github.com/NiJingzhe/SimpleLLMFunc/releases
- **问题反馈**: https://github.com/NiJingzhe/SimpleLLMFunc/issues
