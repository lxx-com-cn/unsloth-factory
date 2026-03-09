#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
服务器配置模块
针对Tesla T4 16GB GPU的优化配置
"""

import os
import torch
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    """
    T4 GPU优化配置类
    
    关键优化点：
    1. 4-bit量化加载（NF4）
    2. 梯度检查点禁用（推理模式）
    3. 序列并行限制（控制并发）
    4. KV缓存管理
    """
    
    # 模型路径配置
    model_path: str = field(default="/home/yaoxp/models/Qwen3-14B/")
    adapter_path: Optional[str] = field(
        default="/home/yaoxp/work/sft/unsloth-factory05/output/sft-qwen3-14b/final_adapter/"
    )
    
    # 服务配置
    host: str = field(default="0.0.0.0")
    port: int = field(default=8000)
    workers: int = field(default=1)  # T4上必须单进程
    
    # 推理配置 - T4优化
    max_seq_length: int = field(default=8192)  # T4建议不超过8K
    max_new_tokens: int = field(default=4096)
    temperature: float = field(default=0.3)  # CoT模式用较低温度
    top_p: float = field(default=0.85)
    top_k: int = field(default=50)
    repetition_penalty: float = field(default=1.05)
    
    # 量化配置 - T4必须4-bit
    load_in_4bit: bool = field(default=True)
    bnb_4bit_quant_type: str = field(default="nf4")
    bnb_4bit_compute_dtype: str = field(default="bfloat16")
    
    # 并发控制 - T4 16GB关键配置
    max_concurrent_requests: int = field(default=2)  # T4上最多2并发
    queue_timeout: int = field(default=120)  # 队列等待超时（秒）
    request_timeout: int = field(default=300)  # 单个请求超时（秒）
    
    # 内存管理
    gpu_memory_fraction: float = field(default=0.95)  # T4上可以用到95%
    enable_cpu_offload: bool = field(default=False)  # T4上一般不需要，4bit已足够
    
    # 功能开关
    enable_think_chain: bool = field(default=True)  # 启用CoT
    enable_queue: bool = field(default=True)  # 启用队列管理
    enable_streaming: bool = field(default=True)  # 启用流式输出
    
    # 领域配置
    domain: str = field(default="medical")
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        """从环境变量加载配置"""
        return cls(
            model_path=os.getenv("MODEL_PATH", cls.model_path),
            adapter_path=os.getenv("ADAPTER_PATH", cls.adapter_path),
            host=os.getenv("SERVER_HOST", cls.host),
            port=int(os.getenv("SERVER_PORT", cls.port)),
            max_seq_length=int(os.getenv("MAX_SEQ_LENGTH", cls.max_seq_length)),
            max_concurrent_requests=int(
                os.getenv("MAX_CONCURRENT", cls.max_concurrent_requests)
            ),
            enable_queue=os.getenv("ENABLE_QUEUE", "true").lower() == "true",
            domain=os.getenv("DOMAIN", cls.domain),
        )
    
    def get_torch_dtype(self):
        """获取PyTorch数据类型"""
        if self.bnb_4bit_compute_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    
    def validate(self) -> bool:
        """验证配置合理性"""
        if self.max_concurrent_requests > 3:
            print("警告：T4 GPU上并发数超过3可能导致OOM，建议设置为1-2")
            return False
        if self.max_seq_length > 16384:
            print("警告：T4上序列长度超过16K可能导致性能急剧下降")
        return True


# 全局配置实例
config = ServerConfig.from_env()