# src/trainers/dpo_trainer.py
import logging
import torch
from transformers import TrainingArguments
from trl import DPOTrainer
from .base_trainer import BaseTrainer
from src.utils.helpers import calculate_dataset_stats

logger = logging.getLogger(__name__)

class DPOTrainer(BaseTrainer):
    """直接偏好优化训练器"""
    
    def __init__(self, args):
        super().__init__(args)  # 移除了template参数
        self.training_args = self.prepare_training_arguments()
        self.trainer = self.create_trainer()
    
    def prepare_training_arguments(self):
        """准备训练参数"""
        return TrainingArguments(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.accumulation_steps,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.epochs,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            save_total_limit=3,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type=self.args.lr_scheduler_type,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            remove_unused_columns=True,
            report_to="none",
            gradient_checkpointing=True,
        )
    
    def create_trainer(self):
        """创建DPO训练器"""
        # 确保数据集格式正确
        if not hasattr(self.dataset, "prompt") or not hasattr(self.dataset, "chosen") or not hasattr(self.dataset, "rejected"):
            logger.error("DPO训练需要特定格式的数据集，包含prompt、chosen和rejected字段")
            raise ValueError("Invalid dataset format for DPO training")
        
        return DPOTrainer(
            model=self.model,
            ref_model=None,  # 使用当前模型作为参考模型
            args=self.training_args,
            beta=self.args.beta if hasattr(self.args, "beta") else 0.1,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            max_length=self.args.max_seq_length,
            max_prompt_length=self.args.max_seq_length // 2,
        )
    
    def train(self):
        """执行训练"""
        logger.info("Starting DPO training...")
        try:
            self.trainer.train()
            self.save_model()
            return True
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA OOM: Try reducing batch_size or max_seq_length")
                logger.error(f"Current config: batch_size={self.args.batch_size}, max_seq_length={self.args.max_seq_length}")
                return False
            else:
                logger.error(f"Runtime error: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False