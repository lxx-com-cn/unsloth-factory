# src/core/model_factory.py - 最终修复版本
import os
import json
import logging
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)

# 全局变量，用于延迟导入
_unsloth_module = None
_fast_language_model = None

def _get_unsloth():
    """延迟导入 Unsloth，避免不必要的 patch"""
    global _unsloth_module, _fast_language_model
    if _unsloth_module is None:
        try:
            import unsloth as _unsloth
            from unsloth import FastLanguageModel as _flm
            _fast_language_model = _flm
            logger.info("Unsloth 延迟导入成功")
        except ImportError as e:
            logger.warning(f"Unsloth 导入失败: {e}")
            _unsloth_module = False
    return _fast_language_model


class ModelFactory:
    """创建和管理大语言模型实例的工厂类"""

    @classmethod
    def create_model(cls, model_path, max_seq_length, adapter_path=None, use_unsloth=True):
        """创建模型实例 - 修复Qwen3延迟导入问题"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"基础模型路径不存在: {model_path}")

        model_type = cls.detect_model_type(model_path)
        logger.info(f"检测到模型类型: {model_type}")

        # 检查Unsloth兼容性
        supports_unsloth = cls.check_unsloth_support(model_type) and use_unsloth
        
        # Qwen3 特殊处理：不导入 Unsloth，避免 patch
        if "qwen3" in model_type.lower():
            logger.info(f"检测到 {model_type}，使用标准Transformers（避免Unsloth patch）")
            supports_unsloth = False

        # 只在需要时导入 Unsloth
        FastLanguageModel = None
        if supports_unsloth:
            FastLanguageModel = _get_unsloth()
            if FastLanguageModel is None:
                logger.warning("Unsloth 不可用，回退到标准方式")
                supports_unsloth = False

        tokenizer = cls.load_tokenizer(model_path)

        # 根据支持情况选择加载方式
        if supports_unsloth and FastLanguageModel is not None:
            try:
                # 针对14B模型添加特殊处理
                if "14b" in model_path.lower():
                    logger.info("检测到14B大模型，应用特殊优化配置 (SFT模式)")
                    model, _ = FastLanguageModel.from_pretrained(
                        model_name=model_path,
                        max_seq_length=max_seq_length,
                        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                        load_in_4bit=True,
                        token=os.environ.get("HF_TOKEN", None),
                    )
                else:
                    model, _ = FastLanguageModel.from_pretrained(
                        model_name=model_path,
                        max_seq_length=max_seq_length,
                        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                        load_in_4bit=True,
                        token=os.environ.get("HF_TOKEN", None),
                    )
                logger.info("使用Unsloth优化加载")
                target_modules = cls.get_target_modules(model_type)
                
                # 应用优化层配置（如果可用）
                if hasattr(FastLanguageModel, 'configure_optimized_parameters'):
                    model = FastLanguageModel.configure_optimized_parameters(model)
                    logger.info("已应用优化层配置")
                
                # 合并适配器（如果提供了路径且用于非训练场景）
                if adapter_path:
                    model = cls.merge_adapter(model, adapter_path)
                return model, tokenizer, target_modules, True
                
            except Exception as e:
                logger.warning(f"Unsloth加载失败: {str(e)}，回退到标准方式")
                # 清理已导入的模块，避免残留影响
                import gc
                gc.collect()
                torch.cuda.empty_cache()

        # 标准加载方式（回退 或 use_unsloth=False）
        logger.info("使用标准Transformers加载方式")
        model = cls.load_base_model(model_path, max_seq_length)
        
        # 合并适配器（如果提供了路径）
        if adapter_path:
            model = cls.merge_adapter(model, adapter_path)
        target_modules = cls.get_target_modules(model_type)
        return model, tokenizer, target_modules, False

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
            actual_path = model_path.lower()

            # 添加对 deepseek-r1-0528-qwen3 的特殊检测
            if "deepseek" in actual_path and "0528" in actual_path and "qwen3" in actual_path:
                return "deepseek_r1_0528_qwen3"
            elif "qwen3" in actual_path or "qwen3" in model_name:
                if "14b" in actual_path:
                    return "qwen3_14b"
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
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @classmethod
    def load_base_model(cls, model_path: str, max_seq_length: int):
        """加载基础模型 - 添加CPU offload支持"""
        logger.info(f"加载基础模型: {model_path}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        device_map = "auto"
        max_memory = None
        
        # 针对14B模型添加特殊处理
        if "14b" in model_path.lower():
            logger.info("检测到14B大模型，启用CPU offload")
            max_memory = {0: "14GiB", "cpu": "32GiB"}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=device_map,
                max_memory=max_memory,
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
        # 明确排除 qwen3 系列
        unsupported = ["qwen3", "qwen3_14b", "deepseek_r1_0528_qwen3", "deepseek_r1_qwen3", "deepseek_r1_qwen"]
        return model_type not in unsupported

    @classmethod
    def get_target_modules(cls, model_type: str):
        """获取模型对应的LoRA目标模块"""
        module_map = {
            "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen3_14b": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "deepseek_r1_qwen": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "deepseek_r1_qwen3": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "deepseek_r1_0528_qwen3": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "unknown": ["q_proj", "k_proj", "v_proj", "o_proj"]
        }
        return module_map.get(model_type, module_map["unknown"])