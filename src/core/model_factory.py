# src/core/model_factory.py
import os
import json
import logging
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import unsloth
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)

class ModelFactory:
    """创建和管理大语言模型实例的工厂类 - 修复Qwen3检测和性能问题"""

    @classmethod
    def create_model(cls, model_path, max_seq_length, adapter_path=None, use_unsloth=True):
        """创建模型实例 - 修复Qwen3检测并支持14B模型"""
        # 验证原始模型路径
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"基础模型路径不存在: {model_path}")

        # 检测模型类型（修复检测逻辑）
        model_type = cls.detect_model_type(model_path)
        logger.info(f"检测到模型类型: {model_type}")

        # 检查Unsloth兼容性 - 添加对deepseek-r1-qwen3的特殊处理
        supports_unsloth = cls.check_unsloth_support(model_type) and use_unsloth

        # 始终使用原始tokenizer
        tokenizer = cls.load_tokenizer(model_path)

        # 根据支持情况选择加载方式
        if supports_unsloth:
            try:
                # 针对14B模型添加特殊处理 - 仅用于微调(SFT)，不添加显存优化参数
                if "14b" in model_path.lower():
                    logger.info("检测到14B大模型，应用特殊优化配置 (SFT模式)")
                    # 微调时使用Unsloth默认加载，不传max_memory等参数
                    model, _ = FastLanguageModel.from_pretrained(
                        model_name=model_path,
                        max_seq_length=max_seq_length,
                        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                        load_in_4bit=True, # 保持4bit量化
                        token=os.environ.get("HF_TOKEN", None),
                        # 注意：移除了可能导致冲突的显存优化参数
                    )
                else:
                    # 其他支持Unsloth的模型使用标准加载
                    model, _ = FastLanguageModel.from_pretrained(
                        model_name=model_path,
                        max_seq_length=max_seq_length,
                        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                        load_in_4bit=True,
                        token=os.environ.get("HF_TOKEN", None),
                    )
                logger.info("使用Unsloth优化加载")
                target_modules = cls.get_target_modules(model_type)
                # 确保应用所有必要的优化层
                if hasattr(FastLanguageModel, 'configure_optimized_parameters'):
                    model = FastLanguageModel.configure_optimized_parameters(model)
                logger.info("已应用优化层配置")
                # 合并适配器（如果提供了路径且用于非训练场景，如chat/eval）
                if adapter_path:
                    model = cls.merge_adapter(model, adapter_path)
                return model, tokenizer, target_modules, True
            except Exception as e:
                logger.warning(f"Unsloth加载失败: {str(e)}，回退到标准方式")
                supports_unsloth = False

        # 标准加载方式（回退 或 use_unsloth=False）
        model = cls.load_base_model(model_path, max_seq_length, supports_unsloth=False) # 显式传False
        # 合并适配器（如果提供了路径）
        if adapter_path:
            model = cls.merge_adapter(model, adapter_path)
        target_modules = cls.get_target_modules(model_type)
        return model, tokenizer, target_modules, supports_unsloth

    @classmethod
    def detect_model_type(cls, model_path: str) -> str:
        """修复模型类型检测"""
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            return "unknown"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            model_name = config.get("_name_or_path", "").lower()
            # 修复检测逻辑，优先检查路径
            actual_path = model_path.lower()

            # 添加对 deepseek-r1-0528-qwen3 的特殊检测
            if "deepseek" in actual_path and "0528" in actual_path and "qwen3" in actual_path:
                return "deepseek_r1_0528_qwen3"
            elif "qwen3" in actual_path or "qwen3" in model_name:
                if "14b" in actual_path:
                    return "qwen3_14b" # 新增14B标识
                return "qwen3"
            elif "deepseek" in actual_path and "qwen3" in actual_path:
                return "deepseek_r1_qwen3"
            elif "deepseek" in actual_path and "qwen" in actual_path:
                return "deepseek_r1_qwen"
            elif "qwen" in actual_path:
                return "qwen"
            else:
                return "unknown"
        except Exception as e:
            logger.error(f"检测模型类型失败: {str(e)}")
            return "unknown"

    @classmethod
    def load_tokenizer(cls, model_path: str) -> AutoTokenizer:
        """加载原始tokenizer"""
        logger.info(f"从原始模型加载tokenizer: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        # 确保有pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @classmethod
    def load_base_model(cls, model_path: str, max_seq_length: int, supports_unsloth: bool):
        """加载基础模型 - 关键修改：添加CPU offload支持"""
        logger.info(f"加载基础模型: {model_path}")
        # 配置量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # 关键修改：添加显存优化配置 - 仅在非Unsloth模式下应用
        device_map = "auto"
        max_memory = None
        # 针对14B模型添加特殊处理 - 用于评估/验证/聊天
        if "14b" in model_path.lower():
            logger.info("检测到14B大模型，启用CPU offload (非SFT模式)")
            # 根据实际情况调整分配
            max_memory = {0: "14GiB", "cpu": "32GiB"}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=device_map,
                max_memory=max_memory, # 仅在此模式下使用
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
        return model

    @classmethod
    def merge_adapter(cls, model, adapter_path: str):
        """合并适配器到基础模型"""
        if not adapter_path or not os.path.exists(adapter_path):
            logger.warning(f"适配器路径无效或不存在: {adapter_path}")
            return model

        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            raise FileNotFoundError(f"适配器配置文件不存在: {adapter_config_path}")

        logger.info(f"正在合并适配器: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        model = model.merge_and_unload()
        logger.info("适配器已合并到基础模型中")
        return model

    @classmethod
    def check_unsloth_support(cls, model_type: str) -> bool:
        """检查模型是否支持Unsloth"""
        # 明确排除 deepseek-r1-0528-qwen3 和 deepseek-r1-qwen3
        unsupported = ["deepseek_r1_0528_qwen3", "deepseek_r1_qwen3", "deepseek_r1_qwen"]
        return model_type not in unsupported

    @classmethod
    def get_target_modules(cls, model_type: str):
        """获取模型对应的LoRA目标模块"""
        module_map = {
            "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen3_14b": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 14B使用相同模块
            "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "deepseek_r1_qwen": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "deepseek_r1_qwen3": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "deepseek_r1_0528_qwen3": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "unknown": ["q_proj", "k_proj", "v_proj", "o_proj"]
        }
        return module_map.get(model_type, module_map["unknown"])

# 警告提示
import transformers, peft
if not hasattr(transformers, "models") or not hasattr(peft, "tuners"):
    logger.warning("WARNING: Unsloth should be imported before transformers, peft to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations. Please restructure your imports with 'import unsloth' at the top of your file.")