# src/core/dataset_factory.py
import json
import os
import logging
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

class DatasetFactory:
    def __init__(self, dataset_info_path=None):
        if dataset_info_path is None:
            # 计算默认路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(os.path.dirname(current_dir))
            dataset_info_path = os.path.join(base_dir, "datasets", "dataset_info.json")
        self.dataset_info = self.load_dataset_info(dataset_info_path)
    
    def load_dataset_info(self, path):
        """加载数据集格式定义"""
        if not os.path.exists(path):
            logger.warning(f"数据集信息文件不存在: {path}，使用默认配置")
            return {
                "alpaca": {"description": "Alpaca指令微调格式"},
                "sharegpt": {"description": "ShareGPT对话格式"},
                "preference": {"description": "偏好数据集格式"}
            }
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("formats", {})
        except Exception as e:
            logger.error(f"加载数据集信息失败: {str(e)}")
            return {}
    
    def create_dataset(self, file_path, format_name, data_limit=None):
        """创建数据集实例"""
        if format_name not in self.dataset_info:
            logger.warning(f"不支持的数据集格式: {format_name}，尝试使用alpaca格式")
            format_name = "alpaca"
        
        format_info = self.dataset_info[format_name]
        loader_name = f"load_{format_name}_dataset"
        
        if not hasattr(self, loader_name):
            logger.warning(f"加载器 {loader_name} 未实现，使用默认的Alpaca加载器")
            loader = self.load_alpaca_dataset
        else:
            loader = getattr(self, loader_name)
        
        try:
            dataset = loader(file_path, data_limit)
            logger.info(f"成功加载 {format_name} 数据集，样本数: {len(dataset)}")
            return dataset
        except Exception as e:
            logger.error(f"加载数据集失败: {str(e)}")
            raise
    
    def load_alpaca_dataset(self, file_path, data_limit=None):
        """加载Alpaca格式数据集 - 修复数据限制问题"""
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")
        
        # 支持多种文件格式
        if file_path.endswith('.json'):
            # 加载整个数据集
            full_dataset = load_dataset("json", data_files=file_path)["train"]
            
            # 应用数据限制
            if data_limit is not None and data_limit > 0:
                # 确保不超过数据集大小
                actual_limit = min(data_limit, len(full_dataset))
                dataset = full_dataset.select(range(actual_limit))
            else:
                dataset = full_dataset
        elif file_path.endswith('.jsonl'):
            # 加载整个数据集
            full_dataset = load_dataset("json", data_files=file_path, split="train")
            
            # 应用数据限制
            if data_limit is not None and data_limit > 0:
                actual_limit = min(data_limit, len(full_dataset))
                dataset = full_dataset.select(range(actual_limit))
            else:
                dataset = full_dataset
        else:
            raise ValueError(f"不支持的格式: {file_path}")
        
        # 应用模板转换
        processed = []
        for item in dataset:
            # 跳过空样本
            if not item.get("instruction") or not item.get("output"):
                continue
                
            text = self.apply_alpaca_template(item)
            if text:  # 确保文本不为空
                processed.append({"text": text})
        
        # 确保不超过数据限制
        if data_limit is not None and len(processed) > data_limit:
            processed = processed[:data_limit]
        
        dataset = Dataset.from_list(processed)
        
        return dataset
    
    def apply_alpaca_template(self, item):
        """应用Alpaca模板"""
        text = ""
        
        # 系统提示
        system_prompt = item.get("system", "")
        if system_prompt and system_prompt.strip():
            text += f"<|system|>\n{system_prompt.strip()}</s>\n"
        
        # 历史对话
        history = item.get("history", [])
        for hist in history:
            if len(hist) == 2 and hist[0].strip() and hist[1].strip():
                text += f"<|user|>\n{hist[0].strip()}</s>\n"
                text += f"<|assistant|>\n{hist[1].strip()}</s>\n"
        
        # 当前指令
        instruction = item.get("instruction", "").strip()
        if not instruction:
            return ""
        
        text += f"<|user|>\n{instruction}"
        
        # 输入信息
        input_text = item.get("input", "").strip()
        if input_text:
            text += f" {input_text}"
        
        text += "</s>\n"
        
        # 助手回复
        output = item.get("output", "").strip()
        if not output:
            return ""
        
        text += f"<|assistant|>\n{output}</s>"
        return text
    
    def load_sharegpt_dataset(self, file_path, data_limit=None):
        """加载ShareGPT格式数据集"""
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")
        
        # 支持多种文件格式
        if file_path.endswith('.json'):
            dataset = load_dataset("json", data_files=file_path)["train"]
        elif file_path.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=file_path, split="train")
        else:
            raise ValueError(f"不支持的格式: {file_path}")
        
        processed = []
        for item in dataset:
            text = self.apply_sharegpt_template(item)
            if text:  # 确保文本不为空
                processed.append({"text": text})
        
        dataset = Dataset.from_list(processed)
        
        # 数据限制
        if data_limit and len(dataset) > data_limit:
            dataset = dataset.select(range(data_limit))
        
        return dataset
    
    def apply_sharegpt_template(self, item):
        """应用ShareGPT模板"""
        text = ""
        
        # 系统提示
        if "system" in item and item["system"]:
            text += f"<|system|>\n{item['system']}</s>\n"
        
        # 对话内容
        if "conversations" in item:
            for conv in item["conversations"]:
                if "from" in conv and "value" in conv:
                    role = "user" if conv["from"] == "human" else "assistant"
                    conv_text = conv["value"].strip()
                    if conv_text:
                        text += f"<|{role}|>\n{conv_text}</s>\n"
        
        # 工具信息
        if "tools" in item and item["tools"]:
            text += f"<|tools|>\n{item['tools']}</s>\n"
        
        return text.strip()
    
    def load_preference_dataset(self, file_path, data_limit=None):
        """加载偏好数据集格式"""
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")
        
        # 支持多种文件格式
        if file_path.endswith('.json'):
            dataset = load_dataset("json", data_files=file_path)["train"]
        elif file_path.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=file_path, split="train")
        else:
            raise ValueError(f"不支持的格式: {file_path}")
        
        processed = []
        for item in dataset:
            processed_item = self.apply_preference_template(item)
            if processed_item:  # 确保项目有效
                processed.append(processed_item)
        
        dataset = Dataset.from_list(processed)
        
        # 数据限制
        if data_limit and len(dataset) > data_limit:
            dataset = dataset.select(range(data_limit))
        
        return dataset
    
    def apply_preference_template(self, item):
        """应用偏好数据集模板"""
        # 确保必要字段存在
        required_fields = ["prompt", "chosen", "rejected"]
        for field in required_fields:
            if field not in item or not item[field].strip():
                return None
        
        return {
            "prompt": item["prompt"].strip(),
            "chosen": item["chosen"].strip(),
            "rejected": item["rejected"].strip()
        }