# src/server/model_manager.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型管理器
实现单例模式管理模型生命周期，确保T4 GPU上只有一个模型实例
"""

import os
import sys
import json
import logging
import threading
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager

import torch
import gc

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.server.config import ServerConfig, config
from src.server.chat_system import ChatSystem

logger = logging.getLogger(__name__)


class ModelManager:
    """
    模型管理器（单例模式）
    
    关键设计：
    1. 全局唯一实例，避免重复加载
    2. 延迟加载，首次请求时才加载模型
    3. 显存监控，防止OOM
    4. 自动清理，请求完成后释放KV缓存
    """
    
    _instance: Optional["ModelManager"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, cfg: Optional[ServerConfig] = None):
        if ModelManager._initialized:
            return
            
        with ModelManager._lock:
            if ModelManager._initialized:
                return
                
            self.config = cfg or config
            self.chat_system: Optional[ChatSystem] = None
            self._model_loaded: bool = False
            self._loading: bool = False
            self._load_error: Optional[str] = None
            self._load_time: float = 0.0
            
            # 统计信息
            self.stats = {
                "total_requests": 0,
                "total_tokens_generated": 0,
                "avg_generation_time": 0.0,
            }
            
            ModelManager._initialized = True
            logger.info("ModelManager初始化完成")
    
    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._model_loaded and self.chat_system is not None
    
    @property
    def is_loading(self) -> bool:
        """检查是否正在加载"""
        return self._loading
    
    def load_model(self, force_reload: bool = False) -> bool:
        """
        加载模型
        
        Args:
            force_reload: 强制重新加载
            
        Returns:
            bool: 是否成功
        """
        if self.is_loaded and not force_reload:
            logger.info("模型已加载，跳过")
            return True
        
        if self._loading:
            logger.info("模型正在加载中，等待...")
            # 等待加载完成
            for _ in range(60):  # 最多等60秒
                if not self._loading:
                    break
                time.sleep(1)
            return self.is_loaded
        
        with ModelManager._lock:
            if self.is_loaded and not force_reload:
                return True
            
            self._loading = True
            self._load_error = None
            
            try:
                start_time = time.time()
                logger.info("=" * 80)
                logger.info("开始加载模型...")
                logger.info(f"模型路径: {self.config.model_path}")
                logger.info(f"适配器路径: {self.config.adapter_path}")
                logger.info(f"量化配置: 4-bit {self.config.bnb_4bit_quant_type}")
                logger.info("=" * 80)
                
                # 清理显存
                self._clear_memory()
                
                # 创建ChatSystem的参数字典
                class Args:
                    def __init__(self, cfg: ServerConfig):
                        self.model = cfg.model_path
                        self.adapter = cfg.adapter_path
                        self.max_seq_length = cfg.max_seq_length
                        self.max_new_tokens = cfg.max_new_tokens
                        self.temperature = cfg.temperature
                        self.top_p = cfg.top_p
                        self.top_k = cfg.top_k
                        self.think_chain = cfg.enable_think_chain
                        self.domain = cfg.domain
                        self.system = ""  # 使用默认系统提示
                
                args = Args(self.config)
                
                # 初始化ChatSystem
                self.chat_system = ChatSystem(args)
                success = self.chat_system.load_model()
                
                if not success:
                    raise RuntimeError("ChatSystem模型加载失败")
                
                self._load_time = time.time() - start_time
                self._model_loaded = True
                
                logger.info("=" * 80)
                logger.info(f"模型加载成功！耗时: {self._load_time:.2f}秒")
                logger.info(f"模型类型: {self.chat_system.model_type}")
                
                # 报告显存使用
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                    logger.info(f"显存占用: {mem_allocated:.2f}GB (预留: {mem_reserved:.2f}GB)")
                
                logger.info("=" * 80)
                return True
                
            except Exception as e:
                self._load_error = str(e)
                logger.error(f"模型加载失败: {e}", exc_info=True)
                self._model_loaded = False
                self.chat_system = None
                return False
                
            finally:
                self._loading = False
    
    def unload_model(self):
        """卸载模型，释放显存"""
        with ModelManager._lock:
            if not self.is_loaded:
                return
            
            logger.info("卸载模型...")
            
            if self.chat_system:
                self.chat_system.unload()
            
            self.chat_system = None
            self._model_loaded = False
            
            self._clear_memory()
            logger.info("模型已卸载")
    
    def _clear_memory(self):
        """清理GPU内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @contextmanager
    def generation_context(self):
        """
        生成上下文管理器
        确保生成完成后清理KV缓存
        """
        self.stats["total_requests"] += 1
        start_time = time.time()
        
        try:
            yield self.chat_system
        finally:
            generation_time = time.time() - start_time
            
            # 更新统计
            n = self.stats["total_requests"]
            self.stats["avg_generation_time"] = (
                (self.stats["avg_generation_time"] * (n - 1) + generation_time) / n
            )
            
            # 清理KV缓存（关键：防止T4 OOM）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """获取GPU内存信息"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        try:
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            free = total - allocated
            
            return {
                "available": True,
                "total_gb": round(total, 2),
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "free_gb": round(free, 2),
                "utilization_percent": round(allocated / total * 100, 1),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """获取完整状态信息"""
        return {
            "model_loaded": self.is_loaded,
            "model_loading": self.is_loading,
            "load_error": self._load_error,
            "load_time_seconds": self._load_time,
            "gpu_memory": self.get_gpu_memory_info(),
            "stats": self.stats.copy(),
            "config": {
                "model_path": self.config.model_path,
                "max_seq_length": self.config.max_seq_length,
                "max_concurrent": self.config.max_concurrent_requests,
            },
        }


# 全局模型管理器实例
model_manager = ModelManager()