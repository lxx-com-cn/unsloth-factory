#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pydantic模型定义
定义所有API请求和响应的数据结构
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """单条消息"""
    role: Literal["system", "user", "assistant"] = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容", min_length=1)


class ChatCompletionRequest(BaseModel):
    """
    聊天完成请求
    
    兼容OpenAI API格式，同时支持自定义参数
    """
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    model: Optional[str] = Field(default="qwen3-14b-medical", description="模型标识")
    
    # 生成参数
    max_tokens: Optional[int] = Field(default=None, description="最大生成token数")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    repetition_penalty: Optional[float] = Field(default=None, ge=1.0)
    
    # 流式控制
    stream: bool = Field(default=False, description="是否流式输出")
    
    # 自定义参数
    think_chain: Optional[bool] = Field(default=None, description="是否启用思维链")
    domain: Optional[str] = Field(default=None, description="领域：medical/finance/legal等")
    session_id: Optional[str] = Field(default=None, description="会话ID，用于上下文保持")
    user_id: Optional[str] = Field(default=None, description="用户ID，用于限流统计")
    
    # 高级参数
    stop: Optional[List[str]] = Field(default=None, description="停止词列表")
    presence_penalty: Optional[float] = Field(default=0.0)
    frequency_penalty: Optional[float] = Field(default=0.0)


class ChatCompletionChoice(BaseModel):
    """完成结果选项"""
    index: int = Field(default=0)
    message: ChatMessage
    finish_reason: Optional[str] = Field(default=None)


class ChatCompletionChunk(BaseModel):
    """流式输出块"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


class UsageInfo(BaseModel):
    """用量统计"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """标准聊天完成响应"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    
    # 自定义字段
    think_content: Optional[str] = Field(default=None, description="思维链内容")
    queue_time: Optional[float] = Field(default=None, description="队列等待时间（秒）")


class ModelInfo(BaseModel):
    """模型信息"""
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    """模型列表响应"""
    object: str = "list"
    data: List[ModelInfo]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory: Dict[str, Any]
    queue_status: Dict[str, Any]
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """错误响应"""
    error: Dict[str, Any]