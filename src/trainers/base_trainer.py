# src/trainers/base_trainer.py - 完整修改版
import os
import logging
import torch
import gc
from peft import LoraConfig, get_peft_model
from src.core.model_factory import ModelFactory
from src.core.dataset_factory import DatasetFactory
from src.utils.helpers import copy_original_tokenizer_files
from src.core.experiment_manager import get_experiment_manager

logger = logging.getLogger(__name__)

class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self.is_unsloth_model = False
        self.experiment_manager = get_experiment_manager(
            getattr(args, 'experiments_root', 'output/experiments')
        )

        torch.cuda.empty_cache()
        gc.collect()

        weights = getattr(args, 'weights', None)
        mixing_strategy = getattr(args, 'mixing_strategy', 'concat')

        self.model, self.tokenizer, self.target_modules, self.is_unsloth_model = ModelFactory.create_model(
            model_path=args.model,
            max_seq_length=args.max_seq_length,
            use_unsloth=True
        )

        self.model.train()
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = False

        dataset_factory = DatasetFactory()
        # ========== 关键：传递 domain 参数 ==========
        self.dataset = dataset_factory.create_dataset(
            file_path=args.dataset,
            format_name=args.dataset_format,
            data_limit=args.data_limit,
            mixing_strategy=mixing_strategy,
            weights=weights,
            domain=args.domain
        )

        self.apply_lora_config()
        self.enable_lora_gradients()

    def enable_lora_gradients(self):
        trainable_params = 0
        all_params = 0
        for name, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(f"总参数: {all_params}, 可训练参数: {trainable_params} ({trainable_params/all_params*100:.2f}%)")
        if trainable_params == 0:
            logger.warning("未找到可训练参数，强制启用LoRA层梯度")
            for name, param in self.model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
                    logger.info(f"强制启用梯度: {name}")

    def apply_lora_config(self):
        lora_config = LoraConfig(
            r=self.args.lora_r if hasattr(self.args, 'lora_r') else 32,
            lora_alpha=self.args.lora_alpha if hasattr(self.args, 'lora_alpha') else 64,
            target_modules=self.target_modules,
            lora_dropout=self.args.lora_dropout if hasattr(self.args, 'lora_dropout') else 0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

        if self.is_unsloth_model:
            try:
                from unsloth import FastLanguageModel
                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=lora_config.r,
                    target_modules=lora_config.target_modules,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                    bias=lora_config.bias,
                    use_gradient_checkpointing=True,
                )
                logger.info("使用Unsloth的LoRA配置")
            except Exception as e:
                logger.warning(f"Unsloth LoRA失败: {e}，回退到标准PEFT")
                self.model = get_peft_model(self.model, lora_config)
                self.model.enable_input_require_grads()
        else:
            self.model = get_peft_model(self.model, lora_config)
            self.model.enable_input_require_grads()

    def save_model(self):
        if hasattr(self.args, 'experiment_id') and self.args.experiment_id:
            adapter_path = os.path.join(self.args.output_dir, "final_adapter")
        else:
            adapter_path = os.path.join(self.args.output_dir, "final_adapter")

        os.makedirs(adapter_path, exist_ok=True)

        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        copy_original_tokenizer_files(self.args.model, adapter_path)

        if hasattr(self.args, 'experiment_id'):
            self.experiment_manager.update_experiment_status(
                "adapter_saved",
                {"adapter_path": adapter_path}
            )

        logger.info(f"模型已保存至: {adapter_path}，并复制原始 tokenizer 配置")