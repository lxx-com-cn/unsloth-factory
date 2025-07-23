# src/trainers/__init__.py
from .base_trainer import BaseTrainer
from .sft_trainer import SFTTrainer
from .dpo_trainer import DPOTrainer  # 添加DPO训练器
from .trainer_factory import TrainerFactory