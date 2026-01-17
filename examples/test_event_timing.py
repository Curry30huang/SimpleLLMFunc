"""
测试事件时序：验证长时间运行工具的事件发射时机

这个测试专门用于验证：
- ToolCallStartEvent 和 ToolCallEndEvent 之间的时间间隔是否正确
- 事件是否在工具执行前后立即发射，而非批量延迟
"""

import asyncio
import time
from typing import List, Dict, Optional

from SimpleLLMFunc import llm_chat, tool
from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible
from SimpleLLMFunc.interface.llm_interface import LLM_Interface
from SimpleLLMFunc.hooks import (
    EventYield,
    ResponseYield,
    ToolCallStartEvent,
    ToolCallEndEvent,
    LLMCallStartEvent,
    LLMCallEndEvent,
)


# ============================================================================
# 测试工具：长时间运行
# ============================================================================

@tool(name="slow_task", description="执行一个需要 10 秒的任务")
async def slow_task(task_name: str) -> str:
    """
    模拟长时间运行的任务。
    
    Args:
        task_name: 任务名称
    
    Returns:
        任务结果
    """
    print(f"  [工具执行中] {task_name} - 开始执行...")
    await asyncio.sleep(10)  # 睡眠 10 秒
    print(f"  [工具执行中] {task_name} - 执行完成!")
    return f"任务 '{task_name}' 已完成（耗时 10 秒）"


# ============================================================================
# 事件时间戳记录器
# ============================================================================

class EventTimingRecorder:
    """记录事件的精确时间戳"""
    
    def __init__(self):
        self.start_time = time.time()
        self.events: List[Dict] = []
    
    def record(self, event_type: str, event_name: str = "", details: str = ""):
        """记录事件"""
        elapsed = time.time() - self.start_time
        record = {
            "type": event_type,
            "name": event_name,
            "details": details,
            "elapsed": elapsed,
            "timestamp": time.time(),
        }
        self.events.append(record)
        
        # 立即打印（带颜色）
        if "Start" in event_type:
            color = "\033[92m"  # 绿色
        elif "End" in event_type:
            color = "\033[93m"  # 黄色
        else:
            color = "\033[94m"  # 蓝色
        reset = "\033[0m"
        
        print(f"{color}[{elapsed:7.3f}s]{reset} {event_type:25s} {event_name:20s} {details}")
    
    def print_summary(self):
        """打印时间间隔摘要"""
        print("\n" + "="*80)
        print("时间间隔分析：")
        print("="*80)
        
        # 找出所有 ToolCallStart 和 ToolCallEnd 的配对
        tool_calls = {}
        for event in self.events:
            if event["type"] == "ToolCallStartEvent":
                tool_name = event["name"]
                tool_calls[tool_name] = {"start": event}
            elif event["type"] == "ToolCallEndEvent":
                tool_name = event["name"]
                if tool_name in tool_calls:
                    tool_calls[tool_name]["end"] = event
        
        # 打印每个工具调用的时间间隔
        for tool_name, events in tool_calls.items():
            if "start" in events and "end" in events:
                interval = events["end"]["elapsed"] - events["start"]["elapsed"]
                print(f"  工具 '{tool_name}':")
                print(f"    开始时间: {events['start']['elapsed']:.3f}s")
                print(f"    结束时间: {events['end']['elapsed']:.3f}s")
                
                # 高亮显示时间间隔
                if interval < 0.1:
                    color = "\033[91m"  # 红色 - 异常短
                    status = "❌ 异常（应该是 10 秒）"
                elif 9.5 <= interval <= 10.5:
                    color = "\033[92m"  # 绿色 - 正常
                    status = "✅ 正常"
                else:
                    color = "\033[93m"  # 黄色 - 可疑
                    status = "⚠️  可疑"
                reset = "\033[0m"
                
                print(f"    {color}时间间隔: {interval:.3f}s  {status}{reset}")
                print()


# ============================================================================
# 测试函数
# ============================================================================

async def test_event_timing(llm_interface: LLM_Interface):
    """测试事件时序"""
    
    print("\n" + "="*80)
    print("开始测试：长时间运行工具的事件时序")
    print("="*80)
    print("测试场景：请求 LLM 使用 slow_task 工具执行一个 10 秒任务")
    print("预期结果：ToolCallStartEvent 和 ToolCallEndEvent 之间应该间隔约 10 秒")
    print("="*80 + "\n")
    
    recorder = EventTimingRecorder()
    
    # 创建聊天函数
    @llm_chat(
        llm_interface=llm_interface,
        toolkit=[slow_task],
        stream=True,
        enable_event=True,
        temperature=0.0
    )
    async def chat(
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ):
        """你是一个助手，帮助用户执行任务。"""
        pass
    
    # 发送测试请求
    test_message = "请使用 slow_task 工具执行一个名为 'test_task' 的任务"
    
    print(f"用户消息: {test_message}\n")
    
    # 消费事件流（模拟 TUI 的方式）
    async for output in chat(user_message=test_message, chat_history=None):
        if isinstance(output, EventYield):
            event = output.event
            
            if isinstance(event, LLMCallStartEvent):
                recorder.record(
                    "LLMCallStartEvent",
                    details=f"stream={event.stream}, tools={len(event.tools) if event.tools else 0}"
                )
            
            elif isinstance(event, LLMCallEndEvent):
                recorder.record(
                    "LLMCallEndEvent",
                    details=f"usage={event.usage}"
                )
            
            elif isinstance(event, ToolCallStartEvent):
                recorder.record(
                    "ToolCallStartEvent",
                    event_name=event.tool_name,
                    details=f"args={event.arguments}"
                )
            
            elif isinstance(event, ToolCallEndEvent):
                recorder.record(
                    "ToolCallEndEvent",
                    event_name=event.tool_name,
                    details=f"success={event.success}, time={event.execution_time:.2f}s"
                )
        
        elif isinstance(output, ResponseYield):
            # 响应数据
            pass
    
    # 打印摘要
    recorder.print_summary()


# ============================================================================
# 主程序
# ============================================================================

async def main():
    """主程序入口"""
    
    # 加载 LLM 配置
    try:
        models = OpenAICompatible.load_from_json_file("provider.json")
        # 使用一个便宜的模型进行测试
        llm = models["openrouter"]["google/gemini-3-flash-preview"]
    except Exception as e:
        print(f"错误: 无法加载 LLM 配置: {e}")
        print("提示: 请确保 provider.json 文件存在且配置正确")
        return
    
    # 运行测试
    await test_event_timing(llm)
    
    print("\n测试完成！")


if __name__ == "__main__":
    asyncio.run(main())
