# config/queue_config.py
"""
队列配置模块
支持通过环境变量或配置文件调整并发数
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class QueueConfig:
    """队列配置类"""
    
    def __init__(self):
        self.max_concurrent = self._get_concurrent_limit()
        self.max_queue_size = self._get_queue_size()
        self.enable_priority = self._get_bool_setting('ENABLE_QUEUE_PRIORITY', True)
        self.timeout_seconds = self._get_timeout()
        
        logger.info(f"队列配置加载完成: 最大并发={self.max_concurrent}, 队列大小={self.max_queue_size}")
    
    def _get_concurrent_limit(self) -> int:
        """获取并发限制 - 修复并发限制"""
        # 环境变量优先
        env_value = os.getenv('CHAT_CONCURRENT_LIMIT')
        if env_value:
            try:
                value = int(env_value)
                if 1 <= value <= 10:  # 限制在1-10之间
                    return value
                else:
                    logger.warning(f"环境变量 CHAT_CONCURRENT_LIMIT={env_value} 超出范围(1-10)，使用默认值2")
            except ValueError:
                logger.warning(f"环境变量 CHAT_CONCURRENT_LIMIT={env_value} 不是有效数字，使用默认值2")
        
        # 配置文件
        try:
            from . import settings
            if hasattr(settings, 'CHAT_CONCURRENT_LIMIT'):
                value = getattr(settings, 'CHAT_CONCURRENT_LIMIT')
                if 1 <= value <= 10:
                    return value
        except ImportError:
            pass
        
        # 默认值 - 根据GPU显存调整（更宽松的限制）
        gpu_memory = self._get_gpu_memory()
        if gpu_memory >= 24:  # 24GB以上
            return 4
        elif gpu_memory >= 16:  # 16GB以上
            return 3
        elif gpu_memory >= 12:  # 12GB (T4有15GB，设置为2个并发)
            return 2
        else:  # 小于12GB
            return 1
    
    def _get_gpu_memory(self) -> float:
        """获取GPU显存(GB)"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            pass
        return 0
    
    def _get_queue_size(self) -> int:
        """获取队列大小"""
        env_value = os.getenv('CHAT_QUEUE_SIZE')
        if env_value:
            try:
                return int(env_value)
            except ValueError:
                logger.warning(f"环境变量 CHAT_QUEUE_SIZE={env_value} 不是有效数字，使用默认值50")
        
        try:
            from . import settings
            if hasattr(settings, 'CHAT_QUEUE_SIZE'):
                return getattr(settings, 'CHAT_QUEUE_SIZE')
        except ImportError:
            pass
        
        return 50
    
    def _get_bool_setting(self, env_var: str, default: bool) -> bool:
        """获取布尔设置"""
        env_value = os.getenv(env_var)
        if env_value:
            return env_value.lower() in ('true', '1', 'yes', 'on')
        
        try:
            from . import settings
            if hasattr(settings, env_var):
                return getattr(settings, env_var)
        except ImportError:
            pass
        
        return default
    
    def _get_timeout(self) -> int:
        """获取超时时间"""
        env_value = os.getenv('CHAT_REQUEST_TIMEOUT')
        if env_value:
            try:
                return int(env_value)
            except ValueError:
                logger.warning(f"环境变量 CHAT_REQUEST_TIMEOUT={env_value} 不是有效数字，使用默认值30")
        
        try:
            from . import settings
            if hasattr(settings, 'CHAT_REQUEST_TIMEOUT'):
                return getattr(settings, 'CHAT_REQUEST_TIMEOUT')
        except ImportError:
            pass
        
        return 30
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'max_concurrent': self.max_concurrent,
            'max_queue_size': self.max_queue_size,
            'enable_priority': self.enable_priority,
            'timeout_seconds': self.timeout_seconds
        }

# 全局配置实例
config = QueueConfig()