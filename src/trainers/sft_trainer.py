# src/trainers/sft_trainer.py - 完整修改版
import os
import json
import logging
import warnings
import torch
from transformers import TrainingArguments
from trl import SFTTrainer as TRLSFTTrainer
from .base_trainer import BaseTrainer
from src.utils.helpers import calculate_dataset_stats, log_memory_usage

warnings.filterwarnings("ignore", message="You passed a `max_seq_length` argument to the SFTTrainer")
warnings.filterwarnings("ignore", message="You passed a `dataset_text_field` argument to the SFTTrainer")

logger = logging.getLogger(__name__)

class SFTTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.starting_step = 0
        self.training_args = self.prepare_training_arguments()
        self.trainer = self.create_trainer()

    def _extract_checkpoint_step(self, dirname: str) -> int:
        try:
            if not dirname.startswith("checkpoint-"):
                return -1
            parts = dirname.split("-")
            if len(parts) < 2:
                return -1
            step_str = parts[1]
            step_str = step_str.split("_")[0].split(".")[0]
            return int(step_str)
        except (ValueError, IndexError):
            return -1

    def _find_checkpoints(self, output_dir: str) -> list:
        if not os.path.exists(output_dir):
            return []
        checkpoint_dirs = []
        for d in os.listdir(output_dir):
            full_path = os.path.join(output_dir, d)
            if not os.path.isdir(full_path):
                continue
            if not d.startswith("checkpoint-"):
                continue
            step = self._extract_checkpoint_step(d)
            if step < 0:
                logger.warning(f"跳过无效的检查点目录名: {d}")
                continue
            trainer_state_path = os.path.join(full_path, "trainer_state.json")
            if not os.path.exists(trainer_state_path):
                logger.warning(f"检查点缺少trainer_state.json，跳过: {d}")
                continue
            checkpoint_dirs.append((step, d, full_path))
        checkpoint_dirs.sort(key=lambda x: x[0])
        return checkpoint_dirs

    def prepare_training_arguments(self):
        logging_steps = self.args.logging_steps
        resume_from_checkpoint = None
        self.starting_step = 0

        if hasattr(self.args, 'resume_checkpoint') and self.args.resume_checkpoint:
            resume_from_checkpoint = self.args.resume_checkpoint
            try:
                state_path = os.path.join(resume_from_checkpoint, "trainer_state.json")
                if os.path.exists(state_path):
                    with open(state_path, 'r') as f:
                        state = json.load(f)
                    self.starting_step = state.get("global_step", 0)
                    logger.info(f"从检查点恢复: {resume_from_checkpoint}, 起始步数: {self.starting_step}")
            except Exception as e:
                logger.warning(f"读取检查点状态失败: {e}")
        elif self.args.resume and self.args.resume.lower() == "auto":
            checkpoints = self._find_checkpoints(self.args.output_dir)
            if checkpoints:
                latest_step, latest_dir, latest_path = checkpoints[-1]
                resume_from_checkpoint = latest_path
                self.starting_step = latest_step
                logger.info(f"自动恢复最新检查点: {latest_dir} (步数: {self.starting_step})")
            else:
                logger.info("未找到有效的检查点，将从头开始训练")

        # ========== 优化训练参数 ==========
        return TrainingArguments(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.accumulation_steps,
            learning_rate=self.args.learning_rate if hasattr(self.args, 'learning_rate') else 3e-5,   # 提高学习率
            num_train_epochs=self.args.epochs if hasattr(self.args, 'epochs') else 5,                # 增加epochs
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps,
            save_steps=self.args.save_steps,
            save_total_limit=3,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type=self.args.lr_scheduler_type if hasattr(self.args, 'lr_scheduler_type') else "cosine",
            warmup_ratio=0.1,
            max_grad_norm=2.0,          # 增大梯度裁剪，稳定训练
            remove_unused_columns=True,
            report_to="none",
            gradient_checkpointing=True,
            dataloader_num_workers=self.args.dataloader_workers,
            dataloader_pin_memory=True,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    def create_trainer(self):
        stats = calculate_dataset_stats(self.dataset)
        packing = not self.args.no_packing
        if packing and (stats['total_samples'] < 10 or stats['avg_length'] * 10 < self.args.max_seq_length):
            packing = False

        torch.cuda.empty_cache()
        logger.info(f"创建训练器前内存使用: {log_memory_usage()}")

        return TRLSFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            train_dataset=self.dataset,
            max_seq_length=self.args.max_seq_length,
            packing=packing,
            dataset_text_field="text",
        )

    def train(self):
        logger.info(f"开始{self.args.domain}领域的SFT训练...")
        logger.info(f"实验ID: {getattr(self.args, 'experiment_id', 'N/A')}")

        try:
            torch.cuda.empty_cache()
            logger.info(f"训练开始前内存使用: {log_memory_usage()}")

            self.model.train()
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

            train_result = self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)

            final_step = train_result.global_step if hasattr(train_result, 'global_step') else 0
            final_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else 0

            if hasattr(self.args, 'experiment_id'):
                self.experiment_manager.update_experiment_status(
                    "training_completed",
                    {
                        "final_step": final_step,
                        "final_loss": final_loss,
                        "trained_epochs": self.args.epochs
                    }
                )

            torch.cuda.empty_cache()
            logger.info(f"训练完成后内存使用: {log_memory_usage()}")

            self.save_model()
            return True

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA OOM: Try reducing batch_size or max_seq_length")
                logger.error(f"当前配置: batch_size={self.args.batch_size}, max_seq_length={self.args.max_seq_length}")
                return False
            else:
                logger.error(f"Runtime error: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False