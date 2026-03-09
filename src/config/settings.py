#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一配置管理模块
支持环境变量和配置文件两种方式
"""

import os
from typing import Optional


class Config:
    """全局配置类"""

    # 模型路径配置 - 通过环境变量或硬编码默认值
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        "/home/yaoxp/models/Qwen3-14B/"
    )

    # 适配器路径配置 - 使用最新的实验目录
    ADAPTER_PATH: str = os.getenv(
        "ADAPTER_PATH",
        "/home/yaoxp/work/sft/uf8/output/experiments/sft_medical_002_0305_194002/final_adapter/"
    )

    # API服务配置
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "12001"))
    API_URL: str = os.getenv("API_URL", f"http://{API_HOST}:{API_PORT}")

    # Web UI配置
    WEB_HOST: str = os.getenv("WEB_HOST", "0.0.0.0")
    WEB_PORT: int = int(os.getenv("WEB_PORT", "8080"))

    # 推理配置
    MAX_SEQ_LENGTH: int = int(os.getenv("MAX_SEQ_LENGTH", "8192"))
    MAX_CONCURRENT: int = int(os.getenv("MAX_CONCURRENT", "2"))
    DOMAIN: str = os.getenv("DOMAIN", "medical")

    @classmethod
    def get_api_url(cls) -> str:
        """获取完整的API URL"""
        return f"http://{cls.API_HOST}:{cls.API_PORT}"

    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("=" * 60)
        print("当前配置:")
        print("=" * 60)
        print(f"模型路径:    {cls.MODEL_PATH}")
        print(f"适配器路径:  {cls.ADAPTER_PATH}")
        print(f"API地址:     {cls.get_api_url()}")
        print(f"Web地址:     http://{cls.WEB_HOST}:{cls.WEB_PORT}")
        print(f"序列长度:    {cls.MAX_SEQ_LENGTH}")
        print(f"最大并发:    {cls.MAX_CONCURRENT}")
        print(f"领域:        {cls.DOMAIN}")
        print("=" * 60)


# 向后兼容的导出
config = Config()