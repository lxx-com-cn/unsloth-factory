# src/core/model_factory.py
import os
import json
import logging
import warnings
import torch
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import unsloth
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)

class ModelFactory:
    """创建和管理大语言模型实例的工厂类 - 修复Qwen3检测和性能问题"""
    
    @classmethod
    def create_model(cls, model_path, max_seq_length, adapter_path=None, use_unsloth=True):
        """创建模型实例 - 修复Qwen3检测"""
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
                model, _ = FastLanguageModel.from_pretrained(
                    model_name=model_path,
                    max_seq_length=max_seq_length,
                    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    load_in_4bit=True,
                )
                logger.info("使用Unsloth优化加载")
                target_modules = cls.get_target_modules(model_type)
                return model, tokenizer, target_modules, True
            except Exception as e:
                logger.warning(f"Unsloth加载失败: {str(e)}，回退到标准方式")
                supports_unsloth = False
        
        # 标准加载方式（回退）
        model = cls.load_base_model(model_path, max_seq_length, supports_unsloth)
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
            tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
            logger.info(f"设置pad token为: {tokenizer.pad_token}")
        
        return tokenizer
    
    @classmethod
    def load_base_model(cls, model_path: str, max_seq_length: int, use_unsloth: bool):
        """加载基础模型"""
        logger.info(f"加载基础模型: {model_path}")
        
        # 配置量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else None,
            )
        
        model.config.use_cache = False
        return model
    
    @classmethod
    def load_adapter(cls, model, adapter_path: str):
        """加载适配器"""
        logger.info(f"加载适配器: {adapter_path}")
        
        # 验证适配器配置
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            raise FileNotFoundError(f"适配器配置文件不存在: {adapter_config_path}")
        
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False
        )
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
    def get_target_modules(cls, model_type: str) -> list:
        """根据模型类型获取目标模块"""
        module_map = {
            "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "deepseek_r1_qwen": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "deepseek_r1_qwen3": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "deepseek_r1_0528_qwen3": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "unknown": ["q_proj", "k_proj", "v_proj", "o_proj"]
        }
        return module_map.get(model_type, module_map["unknown"])