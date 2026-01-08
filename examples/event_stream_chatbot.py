"""
事件流 Chatbot 示例

展示如何使用 SimpleLLMFunc 的事件流功能构建一个功能完整的聊天机器人。
使用 rich 库实现美观的终端 UI，实时显示：
- LLM 流式响应（Markdown 渲染）
- 工具调用过程（参数和结果）
- 执行统计信息（token 使用、耗时等）

运行要求：
    pip install rich

使用方法：
    python event_stream_chatbot.py

功能特性：
- 实时流式响应渲染
- 工具调用可视化
- 事件驱动的 UI 更新
- 完整的执行统计
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from rich.console import Console
from rich.table import Table

from SimpleLLMFunc import llm_chat, tool
from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.hooks import (
    ReactOutput,
    ResponseYield,
    EventYield,
    ReActEventType,
    ReactStartEvent,
    LLMCallStartEvent,
    LLMChunkArriveEvent,
    LLMCallEndEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolCallsBatchStartEvent,
    ToolCallsBatchEndEvent,
    ReactEndEvent,
)

# 初始化 Rich Console
console = Console()


# ============================================================================
# 工具定义
# ============================================================================

@tool(name="calculate", description="执行数学计算，支持基本的算术运算")
async def calculate(expression: str) -> str:
    """
    执行数学计算。
    
    Args:
        expression: 数学表达式，例如 "2 + 3 * 4"
    
    Returns:
        计算结果
    """
    try:
        # 安全的数学计算（仅支持基本运算）
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool(name="get_weather", description="查询指定城市的天气信息")
async def get_weather(city: str) -> str:
    """
    查询天气信息（模拟）。
    
    Args:
        city: 城市名称
    
    Returns:
        天气信息
    """
    # 模拟异步 API 调用
    await asyncio.sleep(0.5)
    
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，温度 15-25°C，空气质量良好",
        "上海": "多云，温度 18-28°C，有轻微雾霾",
        "深圳": "阴天，温度 22-30°C，可能有小雨",
    }
    
    return weather_data.get(city, f"{city}的天气信息：晴天，温度适中")


@tool(name="search_knowledge", description="搜索知识库获取专业信息")
async def search_knowledge(query: str) -> str:
    """
    搜索知识库（模拟）。
    
    Args:
        query: 搜索查询
    
    Returns:
        搜索结果
    """
    await asyncio.sleep(0.3)
    
    # 模拟知识库搜索
    knowledge = {
        "python": "Python 是一种高级编程语言，以其简洁的语法和强大的库生态系统而闻名。",
        "ai": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "llm": "大语言模型（LLM）是一种深度学习模型，通过在大量文本数据上训练，能够理解和生成人类语言。",
    }
    
    for key, value in knowledge.items():
        if key in query.lower():
            return f"知识库结果：{value}"
    
    return f"关于 '{query}' 的信息：这是一个有趣的话题，建议进一步探索。"


# ============================================================================
# 事件处理和 UI 渲染
# ============================================================================

class ChatbotUI:
    """聊天机器人 UI 管理器"""
    
    def __init__(self):
        self.console = console
        self.current_response = ""
        self.tool_calls_info: List[Dict[str, Any]] = []
        self.stats = {
            "total_llm_calls": 0,
            "total_tool_calls": 0,
            "total_chunks": 0,
            "start_time": None,
            "end_time": None,
        }
    
    def print_user_message(self, message: str):
        """打印用户消息"""
        self.console.print()
        self.console.print("[bold cyan]👤 用户[/bold cyan]")
        self.console.print(f"[cyan]{message}[/cyan]")
        self.console.print()
    
    def print_system_info(self, message: str, style: str = "dim"):
        """打印系统信息"""
        self.console.print(f"[{style}]ℹ️  {message}[/{style}]")
    
    def handle_react_start(self, event: ReactStartEvent):
        """处理 ReAct 开始事件"""
        self.stats["start_time"] = event.timestamp
        self.console.print()
        self.print_system_info(
            f"🚀 开始处理请求 (trace_id: {event.trace_id[:8]}...)",
            "bold green"
        )
    
    def handle_llm_call_start(self, event: LLMCallStartEvent):
        """处理 LLM 调用开始事件"""
        self.stats["total_llm_calls"] += 1
        self.current_response = ""
        
        # 显示 LLM 调用信息
        tools_info = f", {len(event.tools)} 工具可用" if event.tools else ", 无工具"
        mode_info = "流式" if event.stream else "非流式"
        
        self.print_system_info(
            f"🤖 LLM 调用 #{self.stats['total_llm_calls']} ({mode_info}{tools_info})"
        )
    
    def handle_llm_chunk(self, event: LLMChunkArriveEvent):
        """处理 LLM 流式块事件"""
        self.stats["total_chunks"] += 1
        # 累积响应内容
        self.current_response = event.accumulated_content
        
        # 首次 chunk 时打印标题和上分隔线
        if event.chunk_index == 0:
            self.console.print()
            self.console.print("[bold green]🤖 助手回复[/bold green]")
            # 打印横线分隔符
            self.console.print("─" * 80)
        
        # 从 chunk 中提取增量内容并打印
        from SimpleLLMFunc.base.post_process import extract_content_from_stream_response
        chunk_content = extract_content_from_stream_response(event.chunk, "chatbot")
        if chunk_content:
            self.render_response_chunk(chunk_content)
    
    def handle_llm_call_end(self, event: LLMCallEndEvent):
        """处理 LLM 调用结束事件"""
        # 打印换行符（最后一个 token 后面）
        self.console.print()
        # 打印下分隔线
        self.console.print("─" * 80)
        
        # 显示 token 使用情况（现在 usage 是 CompletionUsage 对象，有良好的代码补全）
        if event.usage:
            self.print_system_info(
                f"📊 Token 使用: "
                f"输入={event.usage.prompt_tokens}, "
                f"输出={event.usage.completion_tokens}, "
                f"总计={event.usage.total_tokens}"
            )
    
    def handle_tool_calls_batch_start(self, event: ToolCallsBatchStartEvent):
        """处理工具调用批次开始事件"""
        self.console.print()
        self.print_system_info(
            f"🔧 开始执行 {event.batch_size} 个工具调用",
            "bold yellow"
        )
        self.tool_calls_info = []
    
    def handle_tool_call_start(self, event: ToolCallStartEvent):
        """处理单个工具调用开始事件"""
        self.stats["total_tool_calls"] += 1
        
        # 创建工具调用信息表格
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("参数", style="cyan")
        table.add_column("值", style="white")
        
        for key, value in event.arguments.items():
            table.add_row(key, str(value))
        
        self.console.print()
        self.console.print(f"[bold yellow]🛠️  工具调用: {event.tool_name}[/bold yellow]")
        self.console.print(table)
    
    def handle_tool_call_end(self, event: ToolCallEndEvent):
        """处理工具调用结束事件"""
        # 显示工具执行结果
        result_text = str(event.result)[:200]  # 限制长度
        if len(str(event.result)) > 200:
            result_text += "..."
        
        status = "✅ 成功" if event.success else "❌ 失败"
        self.console.print(
            f"  {status} ({event.execution_time:.2f}s): [dim]{result_text}[/dim]"
        )
    
    def handle_tool_calls_batch_end(self, event: ToolCallsBatchEndEvent):
        """处理工具调用批次结束事件"""
        success_count = sum(1 for r in event.tool_results if r["success"])
        self.print_system_info(
            f"✨ 工具调用完成: {success_count}/{len(event.tool_results)} 成功, "
            f"总耗时 {event.total_execution_time:.2f}s"
        )
    
    def handle_react_end(self, event: ReactEndEvent):
        """处理 ReAct 结束事件"""
        self.stats["end_time"] = event.timestamp
        
        # 计算总耗时
        if self.stats["start_time"] and self.stats["end_time"]:
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        else:
            duration = 0
        
        # 显示完整统计信息
        self.console.print()
        stats_table = Table(show_header=False, box=None, padding=(0, 2))
        stats_table.add_column("项目", style="bold")
        stats_table.add_column("值", style="cyan")
        
        stats_table.add_row("总耗时", f"{duration:.2f}s")
        stats_table.add_row("LLM 调用次数", str(event.total_llm_calls))
        stats_table.add_row("工具调用次数", str(event.total_tool_calls))
        stats_table.add_row("流式块数量", str(self.stats["total_chunks"]))
        
        if event.total_token_usage:
            # total_token_usage 是 CompletionUsage 对象，有良好的代码补全
            stats_table.add_row(
                "总 Token 使用",
                f"{event.total_token_usage.total_tokens} "
                f"(输入: {event.total_token_usage.prompt_tokens}, "
                f"输出: {event.total_token_usage.completion_tokens})"
            )
        
        self.console.print()
        self.console.print("[bold green]📈 执行统计[/bold green]")
        self.console.print(stats_table)
    
    def render_response_chunk(self, text: str):
        """渲染响应文本块（流式）"""
        # 直接打印文本块，不换行
        self.console.print(text, end="", style="bold white")


# ============================================================================
# 事件流处理包装器
# ============================================================================

async def chatbot_with_event_ui(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    llm_interface: Optional[LLM_Interface] = None,
) -> tuple[str, List[Dict[str, str]]]:
    """
    带有事件流 UI 的聊天机器人包装器。
    
    这个函数展示如何在外部处理事件流，实现自定义的 UI 和逻辑。
    
    Args:
        message: 用户消息
        history: 对话历史
        llm_interface: LLM 接口实例
        
    Returns:
        (最终响应, 更新后的历史) 元组
        
    Raises:
        ValueError: 当 llm_interface 为 None 时
    """
    ui = ChatbotUI()
    
    # 打印用户消息
    ui.print_user_message(message)
    
    # 确保 llm_interface 不为 None
    if llm_interface is None:
        raise ValueError("llm_interface 不能为 None")
    
    # 创建聊天函数（启用事件流）
    @llm_chat(
        llm_interface=llm_interface,
        toolkit=[calculate, get_weather, search_knowledge],
        stream=True,
        enable_event=True,  # 🔑 启用事件流
    )
    async def chat(
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ):
        """智能助手，可以进行计算、查询天气和搜索知识库。"""
        pass
    
    # 开始处理事件流
    final_response = ""
    final_history: List[Dict[str, str]] = history if history is not None else []
    
    async for output in chat(user_message=message, chat_history=history):
        # 类型检查：判断是事件还是响应
        if isinstance(output, EventYield):
            # 处理事件
            event = output.event
            
            if isinstance(event, ReactStartEvent):
                ui.handle_react_start(event)
            
            elif isinstance(event, LLMCallStartEvent):
                ui.handle_llm_call_start(event)
            
            elif isinstance(event, LLMChunkArriveEvent):
                ui.handle_llm_chunk(event)
                # 更新最终响应（使用累积内容）
                final_response = ui.current_response
            
            elif isinstance(event, LLMCallEndEvent):
                ui.handle_llm_call_end(event)
            
            elif isinstance(event, ToolCallsBatchStartEvent):
                ui.handle_tool_calls_batch_start(event)
            
            elif isinstance(event, ToolCallStartEvent):
                ui.handle_tool_call_start(event)
            
            elif isinstance(event, ToolCallEndEvent):
                ui.handle_tool_call_end(event)
            
            elif isinstance(event, ToolCallsBatchEndEvent):
                ui.handle_tool_calls_batch_end(event)
            
            elif isinstance(event, ReactEndEvent):
                ui.handle_react_end(event)
        
        elif isinstance(output, ResponseYield):
            # 处理响应数据
            response = output.response
            # 类型转换：MessageList 转为 List[Dict[str, str]]
            final_history = output.messages  # type: ignore[assignment]
            
            # 流式模式下，response 是 LLMStreamChunk，内容已经在 LLMChunkArriveEvent 中处理
            # 非流式模式下，response 可能是字符串或 LLMResponse 对象
            if isinstance(response, str):
                # 非流式模式的字符串响应
                console.print()
                console.print("[bold green]🤖 助手回复[/bold green]")
                console.print("─" * 80)  # 上分隔线
                ui.render_response_chunk(response)
                console.print()  # 最后一个 token 后面的换行符
                console.print("─" * 80)  # 下分隔线
                final_response = response
    
    # 响应结束后换行（如果有内容输出）
    if final_response:
        console.print()
    
    return final_response, final_history


# ============================================================================
# 主程序
# ============================================================================

async def main():
    """主程序入口"""
    # 加载 LLM 配置
    try:
        models = OpenAICompatible.load_from_json_file("provider.json")
        llm = models["openrouter"]["google/gemini-3-flash-preview"]
    except Exception as e:
        console.print(f"[bold red]错误: 无法加载 LLM 配置: {e}[/bold red]")
        console.print("[yellow]提示: 请确保 provider.json 文件存在且配置正确[/yellow]")
        return
    
    # 打印欢迎信息
    console.print("[bold cyan]SimpleLLMFunc 事件流 Chatbot 示例[/bold cyan]")
    console.print()
    console.print("这个示例展示了如何使用事件流功能构建一个功能完整的聊天机器人。")
    console.print("特性：")
    console.print("  • 实时流式响应渲染")
    console.print("  • 工具调用可视化")
    console.print("  • 完整的执行统计")
    console.print()
    console.print("可用命令：")
    console.print("  - 输入消息开始对话")
    console.print("  - 输入 'quit' 或 'exit' 退出")
    console.print("  - 输入 'clear' 清空对话历史")
    console.print()
    
    # 对话历史
    history: List[Dict[str, str]] = []
    
    # 主循环
    while True:
        try:
            # 获取用户输入
            console.print()
            user_input = console.input("[bold cyan]你:[/bold cyan] ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.lower() in ["quit", "exit"]:
                console.print("[yellow]再见！[/yellow]")
                break
            
            if user_input.lower() == "clear":
                history = []
                console.print("[yellow]对话历史已清空[/yellow]")
                continue
            
            # 处理用户消息
            response, history = await chatbot_with_event_ui(
                message=user_input,
                history=history,
                llm_interface=llm,
            )
        
        except KeyboardInterrupt:
            console.print("\n[yellow]程序已中断[/yellow]")
            break
        
        except Exception as e:
            console.print(f"[bold red]错误: {e}[/bold red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())

