#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
流式响应生成器
处理SSE流式输出，支持思维链和答案分离
"""

import json
import time
import uuid
import asyncio
import logging
from typing import AsyncGenerator, Optional, Dict, Any

# 关键修复：添加logger定义
logger = logging.getLogger(__name__)

from src.server.schemas import ChatCompletionChunk


async def create_streaming_response(
    chat_system,
    messages: list,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    think_chain: bool = True,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    创建流式响应生成器
    
    输出格式兼容OpenAI SSE标准：
    data: {"id": "...", "choices": [{"delta": {"content": "..."}}]}
    """
    
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())
    
    # 提取用户输入
    user_input = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            user_input = msg["content"]
            break
    
    if not user_input:
        error_chunk = {"error": "No user message found"}
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return
    
    # 发送开始标记
    start_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }],
    }
    yield f"data: {json.dumps(start_chunk, ensure_ascii=False)}\n\n"
    await asyncio.sleep(0)
    
    # 缓冲区
    think_buffer = []
    answer_buffer = []
    
    try:
        # 调用chat_system的流式生成方法
        logger.info(f"开始流式生成，用户输入: {user_input[:50]}...")
        
        token_count = 0
        async for token in chat_system.stream_generate_response(user_input, session_id):
            token_count += 1
            
            # 处理思维链token
            if token.startswith("THINK:"):
                think_text = token[6:]
                think_buffer.append(think_text)
                
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"think_token": think_text},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
            
            # 处理答案token
            else:
                answer_buffer.append(token)
                
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
        
        logger.info(f"流式生成完成，共 {token_count} 个token")
        
        # 发送完成标记
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
            "usage": {
                "think_tokens": len(think_buffer),
                "answer_tokens": len(answer_buffer),
                "total_tokens": len(think_buffer) + len(answer_buffer),
            },
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"流式生成错误: {e}")
        error_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "error",
            }],
            "error": str(e),
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


def format_non_streaming_response(
    think_content: str,
    answer_content: str,
    model_id: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Dict[str, Any]:
    """
    格式化非流式响应
    """
    total_tokens = prompt_tokens + completion_tokens
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": answer_content,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "think_content": think_content if think_content else None,
    }