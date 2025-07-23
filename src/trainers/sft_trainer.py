# src/trainers/sft_trainer.py
import os
import logging
import warnings
import torch
from transformers import TrainingArguments
from trl import SFTTrainer as TRLSFTTrainer
from .base_trainer import BaseTrainer
from src.utils.helpers import calculate_dataset_stats, log_memory_usage

# 忽略特定警告
warnings.filterwarnings("ignore", 
    message="You passed a `max_seq_length` argument to the SFTTrainer")
warnings.filterwarnings("ignore", 
    message="You passed a `dataset_text_field` argument to the SFTTrainer")

logger = logging.getLogger(__name__)

class SFTTrainer(BaseTrainer):
    """监督微调训练器 - 完全修复Unsloth支持"""
    
    def __init__(self, args):
        super().__init__(args)
        self.starting_step = 0
        self.training_args = self.prepare_training_arguments()
        self.trainer = self.create_trainer()
    
    def prepare_training_arguments(self):
        """准备训练参数"""
        logging_steps = self.args.logging_steps
        
        # 检查是否从检查点恢复
        resume_from_checkpoint = None
        self.starting_step = 0
        
        if self.args.resume:
            if self.args.resume.lower() == "auto":
                if os.path.exists(self.args.output_dir):
                    checkpoint_dirs = [d for d in os.listdir(self.args.output_dir) 
                                      if d.startswith("checkpoint") and os.path.isdir(os.path.join(self.args.output_dir, d))]
                    if checkpoint_dirs:
                        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
                        latest_checkpoint = checkpoint_dirs[-1]
                        resume_from_checkpoint = os.path.join(self.args.output_dir, latest_checkpoint)
                        
                        try:
                            self.starting_step = int(latest_checkpoint.split("-")[1])
                            logger.info(f"自动恢复最新检查点: {resume_from_checkpoint} (起始步数: {self.starting_step})")
                        except:
                            self.starting_step = 0
                    else:
                        logger.info("未找到检查点，从头开始训练")
                else:
                    logger.info("输出目录不存在，从头开始训练")
            else:
                if os.path.exists(self.args.resume):
                    resume_from_checkpoint = self.args.resume
                    if "checkpoint-" in self.args.resume:
                        try:
                            self.starting_step = int(self.args.resume.split("checkpoint-")[-1])
                        except ValueError:
                            self.starting_step = 0
                    logger.info(f"从指定检查点恢复: {resume_from_checkpoint} (起始步数: {self.starting_step})")
                else:
                    logger.warning(f"检查点不存在: {self.args.resume}, 从头开始训练")
        
        logger.info(f"训练配置: domain={self.args.domain}, batch_size={self.args.batch_size}, "
                    f"accumulation_steps={self.args.accumulation_steps}, "
                    f"logging_steps={logging_steps}, "
                    f"data_limit={self.args.data_limit}, "
                    f"resume={resume_from_checkpoint}, "
                    f"starting_step={self.starting_step}")
        
        return TrainingArguments(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.accumulation_steps,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.epochs,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps,
            save_steps=self.args.save_steps,
            save_total_limit=3,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type=self.args.lr_scheduler_type,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            remove_unused_columns=True,
            report_to="none",
            gradient_checkpointing=True,
            dataloader_num_workers=self.args.dataloader_workers,
            dataloader_pin_memory=True,
            resume_from_checkpoint=resume_from_checkpoint
        )
    
    def create_trainer(self):
        """创建SFT训练器"""
        logger.info(f"创建训练器前内存使用: {log_memory_usage()}")
        
        # 确定是否使用打包
        stats = calculate_dataset_stats(self.dataset)
        packing = not self.args.no_packing
        
        if packing:
            if stats['total_samples'] < 10 or stats['avg_length'] * 10 < self.args.max_seq_length:
                logger.warning("样本数量不足，已自动禁用打包")
                packing = False
            if stats['max_length'] > self.args.max_seq_length:
                logger.warning("发现超长样本，已自动禁用打包")
                packing = False
        
        torch.cuda.empty_cache()
        
        # 修复：根据是否使用Unsloth选择正确的trainer创建方式
        if self.is_unsloth_model:
            # 使用Unsloth优化
            trainer = TRLSFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=self.training_args,
                train_dataset=self.dataset,
                max_seq_length=self.args.max_seq_length,
                packing=packing,
                dataset_text_field="text",
            )
        else:
            # 标准方式
            trainer = TRLSFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=self.training_args,
                train_dataset=self.dataset,
                max_seq_length=self.args.max_seq_length,
                packing=packing,
                dataset_text_field="text",
            )
        
        logger.info(f"训练器创建后内存使用: {log_memory_usage()}")
        logger.info(f"数据加载器配置: {self.args.dataloader_workers} 工作进程")
        
        return trainer
    
    def train(self):
        """执行训练"""
        logger.info(f"开始{self.args.domain}领域的SFT训练...")
        try:
            self.model.train()
            
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("梯度检查点已启用")
            
            torch.cuda.empty_cache()
            logger.info(f"训练开始前内存使用: {log_memory_usage()}")
            
            # 计算总步数
            effective_batch_size = self.args.batch_size * self.args.accumulation_steps
            steps_per_epoch = max(1, len(self.dataset) // effective_batch_size)
            total_steps = steps_per_epoch * self.args.epochs
            
            logger.info(f"总训练步数: {total_steps}")
            
            self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
            
            torch.cuda.empty_cache()
            logger.info(f"训练完成后内存使用: {log_memory_usage()}")
            
            self.save_model()
            return True
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("显存不足错误: 请尝试减小batch_size或max_seq_length")
                logger.error(f"当前配置: batch_size={self.args.batch_size}, max_seq_length={self.args.max_seq_length}")
                return False
            else:
                logger.error(f"运行时错误: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"训练失败: {str(e)}")
            return False
    
    def save_model(self):
        """保存模型"""
        adapter_path = os.path.join(self.args.output_dir, "final_adapter")
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)
        logger.info(f"模型已保存至: {adapter_path}")