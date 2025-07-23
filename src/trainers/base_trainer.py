# src/trainers/base_trainer.py
import os
import logging
import torch
import gc
from peft import LoraConfig, get_peft_model
from src.core.model_factory import ModelFactory
from src.core.dataset_factory import DatasetFactory
import psutil

logger = logging.getLogger(__name__)

class BaseTrainer:
    """训练器基类 - 修复Unsloth支持"""

    def __init__(self, args):
        self.args = args
        self.is_unsloth_model = False

        # 清除缓存
        torch.cuda.empty_cache()
        gc.collect()

        # 加载模型 - 修复：允许使用Unsloth
        self.model, self.tokenizer, self.target_modules, self.is_unsloth_model = ModelFactory.create_model(
            model_path=args.model,
            max_seq_length=args.max_seq_length,
            use_unsloth=True  # 修复：启用Unsloth
        )

        # 确保模型在训练模式
        self.model.train()
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = False

        # 加载数据集
        dataset_factory = DatasetFactory()
        self.dataset = dataset_factory.create_dataset(
            file_path=args.dataset,
            format_name=args.dataset_format,
            data_limit=args.data_limit
        )

        # 应用LoRA配置 - 修复Unsloth支持
        self.apply_lora_config()

        # 确保所有LoRA参数可训练
        self.enable_lora_gradients()

    def enable_lora_gradients(self):
        """确保LoRA参数可训练"""
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
        """应用LoRA配置 - 修复Unsloth支持"""
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        if self.is_unsloth_model:
            try:
                # 使用Unsloth的LoRA配置
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
            # 标准PEFT
            self.model = get_peft_model(self.model, lora_config)
            self.model.enable_input_require_grads()

    def detect_model_type(self, model_path):
        """检测模型类型"""
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            return "unknown"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            model_name = config.get("_name_or_path", "").lower()
            if "deepseek" in model_name and "qwen3" in model_name:
                return "deepseek_r1_qwen3"
            elif "deepseek" in model_name and "qwen" in model_name:
                return "deepseek_r1_qwen"
            elif "qwen3" in model_name:
                return "qwen3"
            elif "qwen" in model_name:
                return "qwen"
            else:
                return "unknown"
        except:
            return "unknown"

    def prepare_training_arguments(self):
        raise NotImplementedError("子类必须实现该方法")

    def create_trainer(self):
        raise NotImplementedError("子类必须实现该方法")

    def train(self):
        raise NotImplementedError("子类必须实现该方法")

    def save_model(self):
        """保存模型"""
        adapter_path = os.path.join(self.args.output_dir, "final_adapter")
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)
        logger.info(f"模型已保存至: {adapter_path}")