# src/trainers/trainer_factory.py
import logging
from .sft_trainer import SFTTrainer
from .dpo_trainer import DPOTrainer

logger = logging.getLogger(__name__)

class TrainerFactory:
    """创建训练器的工厂类"""
    
    TRAINER_MAP = {
        "sft": SFTTrainer,
        "dpo": DPOTrainer,
        # 其他训练器占位
        "kto": None,
        "ppo": None,
        "rm": None,
    }
    
    @classmethod
    def create_trainer(cls, method, args):
        """创建指定类型的训练器 - 移除了template参数"""
        trainer_class = cls.TRAINER_MAP.get(method.lower())
        
        if not trainer_class:
            raise ValueError(f"不支持的训练方法: {method}")
        
        if trainer_class is None:
            raise NotImplementedError(f"{method.upper()} 训练器尚未实现")
        
        logger.info(f"创建 {method.upper()} 训练器")
        return trainer_class(args)