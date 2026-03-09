#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI推理服务器
提供兼容OpenAI API的接口，支持流式输出和队列管理
"""

import os
import sys
import time
import uuid
import json
import logging
import argparse
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# 导入自定义模块
from src.server.config import ServerConfig
from src.server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    UsageInfo,
    ModelListResponse,
    ModelInfo,
    HealthResponse,
)
from src.server.model_manager import model_manager, ModelManager
from src.server.streaming import create_streaming_response, format_non_streaming_response

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# 全局配置
config = ServerConfig()

# lifespan管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    启动时加载模型，关闭时清理
    """
    logger.info("=" * 80)
    logger.info("服务启动中...")
    logger.info("=" * 80)
    
    # 启动时加载模型
    success = model_manager.load_model()
    if not success:
        logger.error("模型加载失败，服务将以降级模式运行")
    
    yield
    
    # 关闭时清理
    logger.info("服务关闭中...")
    model_manager.unload_model()
    logger.info("服务已关闭")

# 创建FastAPI应用
app = FastAPI(
    title="Qwen3-14B Medical LLM Inference API",
    description="""
    基于Unsloth微调的Qwen3-14B医学大模型推理服务
    
    针对Tesla T4 16GB GPU优化，支持：
    - 4-bit量化加载
    - 思维链(CoT)输出
    - 流式响应(SSE)
    - 请求队列管理
    
    与OpenAI API兼容，可直接使用OpenAI客户端调用
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= 健康检查端点 =============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查端点
    返回模型加载状态、GPU内存、队列状态等
    """
    status = model_manager.get_status()
    
    return HealthResponse(
        status="healthy" if status["model_loaded"] else "degraded",
        model_loaded=status["model_loaded"],
        gpu_available=status["gpu_memory"]["available"],
        gpu_memory=status["gpu_memory"],
        queue_status={
            "enabled": False,
            "max_concurrent": config.max_concurrent_requests,
        },
    )

@app.get("/ready")
async def readiness_check():
    """
    Kubernetes就绪探针
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.get("/live")
async def liveness_check():
    """
    Kubernetes存活探针
    """
    return {"status": "alive"}

# ============= 队列状态端点（Web UI需要） =============

@app.get("/queue/status")
async def queue_status():
    """
    队列状态端点 - 供Web UI查询
    """
    return {
        "enabled": False,
        "queue_size": 0,
        "processing_count": 1 if model_manager.is_loaded else 0,
        "max_concurrent": config.max_concurrent_requests,
        "wait_time_estimate": 0,
    }

# ============= 模型管理端点 =============

@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """
    列出可用模型（兼容OpenAI API）
    """
    return ModelListResponse(
        object="list",
        data=[
            ModelInfo(
                id="qwen3-14b-medical",
                object="model",
                created=int(time.time()),
                owned_by="medical-llm",
            )
        ],
    )

@app.post("/v1/models/{model_id}/load")
async def load_model_endpoint(model_id: str, background_tasks: BackgroundTasks):
    """
    手动触发模型加载（管理接口）
    """
    if model_manager.is_loading:
        return {"status": "loading", "message": "Model is already loading"}
    
    background_tasks.add_task(model_manager.load_model)
    return {"status": "started", "message": "Model loading started in background"}

@app.post("/v1/models/unload")
async def unload_model_endpoint():
    """
    卸载模型，释放显存（管理接口）
    """
    model_manager.unload_model()
    return {"status": "success", "message": "Model unloaded"}

# ============= 核心推理端点 =============

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    聊天完成接口（兼容OpenAI API）- 修复版
    
    支持流式(stream=true)和非流式输出
    支持思维链输出（通过think_chain参数控制）
    """
    
    # 检查模型状态
    if not model_manager.is_loaded:
        if model_manager.is_loading:
            raise HTTPException(
                status_code=503,
                detail="Model is loading, please retry later",
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="Model not available",
            )
    
    # 准备参数
    model_id = request.model or "qwen3-14b-medical"
    max_new_tokens = request.max_tokens or config.max_new_tokens
    temperature = request.temperature or config.temperature
    top_p = request.top_p or config.top_p
    think_chain = request.think_chain if request.think_chain is not None else config.enable_think_chain
    
    # 转换消息格式
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # 提取用户输入
    user_input = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            user_input = msg["content"]
            break
    
    if not user_input:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # 流式输出 - 关键修复：确保使用异步生成器
    if request.stream:
        return StreamingResponse(
            create_streaming_response(
                chat_system=model_manager.chat_system,
                messages=messages,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                think_chain=think_chain,
                session_id=request.session_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
                "Content-Type": "text/event-stream; charset=utf-8",
            },
        )
    
    # 非流式输出
    else:
        return await _handle_direct_request(
            messages, model_id, max_new_tokens,
            temperature, top_p, think_chain
        )

async def _handle_direct_request(
    messages: List[Dict[str, str]],
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    think_chain: bool,
) -> Dict[str, Any]:
    """
    直接处理请求（无队列）
    """
    start_time = time.time()
    
    # 提取用户输入
    user_input = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            user_input = msg["content"]
            break
    
    # 生成响应
    think_content = []
    answer_content = []
    
    with model_manager.generation_context() as chat_system:
        # 临时设置参数
        original_max = chat_system.max_new_tokens
        original_temp = chat_system.temperature
        original_top_p = chat_system.top_p
        chat_system.max_new_tokens = max_new_tokens
        chat_system.temperature = temperature
        chat_system.top_p = top_p
        
        try:
            # 关键修复：使用 async for 迭代异步生成器
            async for token in chat_system.stream_generate_response(user_input):
                if token.startswith("THINK:"):
                    think_content.append(token[6:])
                else:
                    answer_content.append(token)
        finally:
            # 恢复原始参数
            chat_system.max_new_tokens = original_max
            chat_system.temperature = original_temp
            chat_system.top_p = original_top_p
    
    think_str = "".join(think_content)
    answer_str = "".join(answer_content)
    
    # 估算token数
    prompt_tokens = len(user_input) // 4
    completion_tokens = len(answer_str) // 4
    
    # 构建响应
    response = format_non_streaming_response(
        think_content=think_str,
        answer_content=answer_str,
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    
    response["queue_time"] = 0.0
    
    return response

# ============= 自定义端点（非OpenAI兼容） =============

@app.post("/generate")
async def simple_generate(prompt: str, max_tokens: int = 512):
    """
    简单生成接口（非OpenAI兼容，便于测试）
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    think_content = []
    answer_content = []
    
    with model_manager.generation_context() as chat_system:
        for token in chat_system.stream_generate_response(prompt):
            if token.startswith("THINK:"):
                think_content.append(token[6:])
            else:
                answer_content.append(token)
    
    return {
        "prompt": prompt,
        "think_content": "".join(think_content),
        "response": "".join(answer_content),
        "model": "qwen3-14b-medical",
    }

@app.get("/stats")
async def get_stats():
    """
    获取服务统计信息
    """
    model_status = model_manager.get_status()
    return {
        "model": model_status,
        "queue": {
            "enabled": False,
            "queue_size": 0,
            "processing_count": 1 if model_manager.is_loaded else 0,
            "max_concurrent": config.max_concurrent_requests,
        },
    }

# ============= 错误处理 =============

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "internal_error",
                "code": 500,
            }
        },
    )

# ============= 启动函数 =============

def start_server():
    """
    启动服务器的入口函数
    支持从命令行参数读取配置
    """
    import uvicorn
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LLM Inference Server")
    parser.add_argument("--model", type=str, required=True, help="Base model path")
    parser.add_argument("--adapter", type=str, default=None, help="Adapter path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=12001, help="Port to listen")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="Max sequence length")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Max concurrent requests")
    parser.add_argument("--domain", type=str, default="medical", help="Domain")
    args = parser.parse_args()
    
    # 更新全局配置
    global config
    config.model_path = args.model
    config.adapter_path = args.adapter
    config.host = args.host
    config.port = args.port
    config.max_seq_length = args.max_seq_length
    config.max_concurrent_requests = args.max_concurrent
    config.domain = args.domain
    
    logger.info(f"启动服务器: {config.host}:{config.port}")
    logger.info(f"模型: {config.model_path}")
    logger.info(f"适配器: {config.adapter_path}")
    
    uvicorn.run(
        "src.server.inference_server:app",
        host=config.host,
        port=config.port,
        workers=1,
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    start_server()