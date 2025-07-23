# src/validators/validator.py
import os
import json
import logging
import torch
import re
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import PeftModel
from src.core.model_factory import ModelFactory
from src.core.dataset_factory import DatasetFactory
from src.utils.helpers import (
    setup_logging,
    validate_domain_response,
    clean_response
)

logger = logging.getLogger(__name__)

class ModelValidator:
    def __init__(self, args):
        self.args = args
        self.dataset_factory = DatasetFactory()
        self.domain = args.domain

    def load_model(self, adapter_path=None):
        """加载基础模型和适配器 - 强制使用原始tokenizer"""
        logger.info(f"加载基础模型: {self.args.model}")
        
        # 使用ModelFactory加载模型，始终使用原始tokenizer
        model, tokenizer, _, _ = ModelFactory.create_model(
            model_path=self.args.model,
            max_seq_length=self.args.max_seq_length,
            adapter_path=adapter_path,  # 传递适配器路径
            use_unsloth=False
        )
        
        return model, tokenizer
    
    def load_dataset(self):
        """加载验证数据集"""
        return self.dataset_factory.create_dataset(
            file_path=self.args.dataset,
            format_name=self.args.dataset_format,
            data_limit=self.args.max_samples
        )
    
    def generate_response(self, model, tokenizer, prompt):
        """生成模型响应 - 优化格式控制"""
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.args.max_seq_length
        ).to(model.device)
        
        # 生成参数优化
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.1  # 防止重复
        }
        
        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)
        
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # 清理响应格式
        response = self.clean_response_format(response)
        
        # 领域验证
        validated = validate_domain_response(response, prompt, self.domain)
        
        return clean_response(validated)
    
    def clean_response_format(self, response: str) -> str:
        """清理响应格式问题"""
        # 移除乱码和特殊标记
        response = re.sub(r'<\|begin_of_text\|>|<\|end_of_text\|>|<\|start_of_text\|>|<\|end_of_start_of_text\|>|<\|begin\|>|<\|end\|>', '', response)
        response = re.sub(r'using the following.*?\n', '', response, flags=re.DOTALL)
        response = re.sub(r'```json.*?\n```', '', response, flags=re.DOTALL)
        
        # 提取有效内容
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)$', response, re.DOTALL)
        
        if think_match and answer_match:
            think = think_match.group(1).strip()
            answer = answer_match.group(1).strip()
            return f"<think>\n{think}\n</think>\n\n{answer}"
        
        return response.strip()
    
    def validate(self):
        """执行验证 - 修复CLI报错"""
        # 加载基础模型
        base_model, base_tokenizer = self.load_model()
        
        # 加载数据集
        dataset = self.load_dataset()
        logger.info(f"加载验证数据集: {len(dataset)} 条样本")
        
        # 随机选择样本
        if len(dataset) > self.args.max_samples:
            indices = random.sample(range(len(dataset)), self.args.max_samples)
            samples = [dataset[i] for i in indices]
        else:
            samples = dataset
        
        # 验证结果存储
        results = {
            "base_model": [],
            "adapter_model": []
        }
        
        # 验证基础模型
        base_model.eval()
        with torch.no_grad():
            for sample in tqdm(samples, desc="验证基础模型"):
                text = sample["text"]
                prompt = text.split("<|assistant|>")[0] + "<|assistant|>"
                
                response = self.generate_response(base_model, base_tokenizer, prompt)
                
                results["base_model"].append({
                    "prompt": prompt,
                    "expected_output": text.split("<|assistant|>")[-1].replace("</s>", "").strip(),
                    "model_response": response,
                })
        
        # 验证适配器模型（如果提供）
        if self.args.adapter:
            adapter_model, adapter_tokenizer = self.load_model(self.args.adapter)
            adapter_model.eval()
            with torch.no_grad():
                for sample in tqdm(samples, desc="验证微调模型"):
                    text = sample["text"]
                    prompt = text.split("<|assistant|>")[0] + "<|assistant|>"
                    
                    response = self.generate_response(adapter_model, adapter_tokenizer, prompt)
                    
                    results["adapter_model"].append({
                        "prompt": prompt,
                        "expected_output": text.split("<|assistant|>")[-1].replace("</s>", "").strip(),
                        "model_response": response,
                    })
        
        # 计算统计信息
        return self.save_results(results)
    
    def save_results(self, results):
        """保存验证结果 - 修复CLI报错"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        output_path = os.path.join(self.args.output_dir, "validation_results.json")
        
        # 计算统计信息
        summary = {
            "total_samples": len(results.get("base_model", [])),
            "model_info": {
                "base_model": self.args.model,
                "adapter": self.args.adapter,
                "model_name": self.args.model
            }
        }
        
        # 基础模型统计
        if results["base_model"]:
            base_lengths = [len(r["model_response"]) for r in results["base_model"]]
            base_issues = sum(1 for r in results["base_model"] if "警告" in r["model_response"])
            summary["base_model"] = {
                "avg_response_length": sum(base_lengths) / len(base_lengths),
                "domain_issues": base_issues
            }
        
        # 适配器模型统计（如果存在）
        if results["adapter_model"]:
            adapter_lengths = [len(r["model_response"]) for r in results["adapter_model"]]
            adapter_issues = sum(1 for r in results["adapter_model"] if "警告" in r["model_response"])
            summary["adapter_model"] = {
                "avg_response_length": sum(adapter_lengths) / len(adapter_lengths),
                "domain_issues": adapter_issues
            }
        
        # 保存完整结果
        full_results = {
            "summary": summary,
            "detailed_results": results,
            "validation_config": {
                "model": self.args.model,
                "adapter": self.args.adapter,
                "dataset": self.args.dataset,
                "max_samples": self.args.max_samples,
                "max_seq_length": self.args.max_seq_length,
                "domain": self.args.domain
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"验证完成! 结果保存至 {output_path}")
        
        # 打印摘要 - 修复CLI报错
        print("\n" + "="*50)
        print("验证摘要:")
        print(f"总样本数: {summary['total_samples']}")
        
        if "base_model" in summary:
            print(f"基础模型平均响应长度: {summary['base_model']['avg_response_length']:.0f} 字符")
            print(f"基础模型领域警告数: {summary['base_model']['domain_issues']}/{summary['total_samples']}")
        
        if "adapter_model" in summary:
            print(f"微调模型平均响应长度: {summary['adapter_model']['avg_response_length']:.0f} 字符")
            print(f"微调模型领域警告数: {summary['adapter_model']['domain_issues']}/{summary['total_samples']}")
        
        print("="*50)
        print(f"完整验证结果已保存至: {output_path}")
        
        return {
            "summary": summary,
            "output_path": output_path
        }