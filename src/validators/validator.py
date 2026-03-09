# src/validators/validator.py - 完整修改版（返回结构包含 detailed_results）
import os
import json
import logging
import torch
import re
import gc
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import PeftModel
from src.core.model_factory import ModelFactory
from src.core.dataset_factory import DatasetFactory
from src.utils.helpers import (
    setup_logging,
    validate_domain_response,
    clean_response,
    log_memory_usage
)
from src.validators.cot_validator import CotValidator

logger = logging.getLogger(__name__)

class ModelValidator:
    def __init__(self, args):
        self.args = args
        self.dataset_factory = DatasetFactory()
        self.domain = args.domain
        self.cot_validator = CotValidator()

        self.output_dir = self._resolve_output_dir()
        logger.info(f"验证结果将保存至: {self.output_dir}")

    def _resolve_output_dir(self) -> str:
        if hasattr(self.args, 'output_dir') and self.args.output_dir:
            if self.args.output_dir != "validation_results":
                return self.args.output_dir
            if hasattr(self.args, 'adapter') and self.args.adapter:
                adapter_path = self.args.adapter
                if "experiments" in adapter_path:
                    parts = adapter_path.split("experiments/")
                    if len(parts) > 1:
                        exp_part = parts[1].split("/")[0]
                        exp_dir = os.path.join("output/experiments", exp_part)
                        if os.path.exists(exp_dir):
                            validation_dir = os.path.join(exp_dir, "validation")
                            os.makedirs(validation_dir, exist_ok=True)
                            return validation_dir
        return self.args.output_dir

    # ========== 强制CoT系统提示（与训练/chat完全一致） ==========
    def _get_forced_system_prompt(self) -> str:
        return (
            "你是一个专业医疗助手。\n"
            "【重要】你的所有回答都必须遵守以下格式：\n"
            "1. 首先，在 <think> 标签内写出你的详细思考过程。\n"
            "2. 然后，在 </think> 标签后写出正式回答。\n"
            "3. 严禁输出自然语言思考而不使用 <think> 标签。\n"
            "示例：\n"
            "<think>\n"
            "用户询问高血压的定义，这是一种常见的慢性病...\n"
            "</think>\n"
            "高血压是指..."
        )

    def _build_validation_prompt(self, instruction: str, input_text: str = "") -> str:
        """构建与训练完全一致的Prompt"""
        user_content = instruction
        if input_text:
            user_content += " " + input_text
        return f"<|system|>\n{self._get_forced_system_prompt()}</s>\n<|user|>\n{user_content}</s>\n<|assistant|>\n"

    # ========== 直接从JSON文件加载原始样本 ==========
    def _load_raw_validation_samples(self) -> List[Dict]:
        file_path = self.args.dataset
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if self.args.max_samples:
            data = data[:self.args.max_samples]

        samples = []
        for item in data:
            instruction = item.get('instruction', '').strip()
            input_text = item.get('input', '').strip() if item.get('input') else ''
            output = item.get('output', '').strip()
            if not instruction or not output:
                continue
            samples.append({
                'instruction': instruction,
                'input': input_text,
                'expected_output': output
            })
        logger.info(f"从 {file_path} 加载 {len(samples)} 条原始验证样本")
        return samples

    def load_model(self, adapter_path=None):
        logger.info(f"加载基础模型: {self.args.model}")
        model, tokenizer, _, _ = ModelFactory.create_model(
            model_path=self.args.model,
            max_seq_length=self.args.max_seq_length,
            adapter_path=adapter_path,
            use_unsloth=False
        )
        return model, tokenizer

    def generate_response(self, model, tokenizer, sample):
        prompt = self._build_validation_prompt(sample['instruction'], sample['input'])
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.args.max_seq_length
        ).to(model.device)

        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.1
        }

        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response

    def validate(self):
        samples = self._load_raw_validation_samples()

        base_model, base_tokenizer = self.load_model()
        base_model.eval()

        results = {"base_model": [], "adapter_model": []}

        with torch.no_grad():
            for sample in tqdm(samples, desc="验证基础模型"):
                response = self.generate_response(base_model, base_tokenizer, sample)
                results["base_model"].append({
                    "prompt": self._build_validation_prompt(sample['instruction'], sample['input']),
                    "expected_output": sample["expected_output"],
                    "model_response": response,
                })

        del base_model, base_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        if self.args.adapter:
            adapter_model, adapter_tokenizer = self.load_model(self.args.adapter)
            adapter_model.eval()
            with torch.no_grad():
                for sample in tqdm(samples, desc="验证微调模型"):
                    response = self.generate_response(adapter_model, adapter_tokenizer, sample)
                    results["adapter_model"].append({
                        "prompt": self._build_validation_prompt(sample['instruction'], sample['input']),
                        "expected_output": sample["expected_output"],
                        "model_response": response,
                    })
            del adapter_model, adapter_tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("开始思维链质量验证...")
        cot_results = {}
        if results["base_model"]:
            cot_results["base_model"] = self.cot_validator.validate_batch(results["base_model"], show_progress=True)
        if results["adapter_model"]:
            cot_results["adapter_model"] = self.cot_validator.validate_batch(results["adapter_model"], show_progress=True)

        # 修改点：直接返回完整结果，而不仅仅是摘要
        return self.save_results(results, cot_results)

    def clean_response_format(self, response: str) -> str:
        response = re.sub(r'<\|begin_of_text\|>|<\|end_of_text\|>|<\|start_of_text\|>|<\|end_of_start_of_text\|>|<\|begin\|>|<\|end\|>', '', response)
        response = re.sub(r'using the following.*?\n', '', response, flags=re.DOTALL)
        response = re.sub(r'```json.*?\n```', '', response, flags=re.DOTALL)
        return response.strip()

    def _cot_result_to_dict(self, result) -> Dict[str, Any]:
        if hasattr(result, '__dataclass_fields__'):
            return {
                "has_think_block": result.has_think_block,
                "has_answer_block": result.has_answer_block,
                "think_content": result.think_content,
                "answer_content": result.answer_content,
                "reasoning_quality": result.reasoning_quality,
                "answer_consistency": result.answer_consistency,
                "think_length": result.think_length,
                "answer_length": result.answer_length,
                "issues": result.issues
            }
        elif isinstance(result, dict):
            return result
        else:
            return {"value": str(result)}

    def save_results(self, results, cot_results=None):
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "validation_results.json")

        summary = {
            "total_samples": len(results.get("base_model", [])),
            "model_info": {
                "base_model": self.args.model,
                "adapter": self.args.adapter,
                "model_name": self.args.model
            }
        }

        if results["base_model"]:
            base_lengths = [len(r["model_response"]) for r in results["base_model"]]
            base_issues = sum(1 for r in results["base_model"] if "警告" in r["model_response"])
            summary["base_model"] = {
                "avg_response_length": sum(base_lengths) / len(base_lengths),
                "domain_issues": base_issues
            }
            if cot_results and "base_model" in cot_results:
                cot_stats = cot_results["base_model"]["statistics"]
                summary["base_model"]["cot_quality"] = {
                    "has_think_block_ratio": cot_stats.get("has_think_block_ratio", 0),
                    "avg_reasoning_quality": cot_stats.get("avg_reasoning_quality", 0),
                    "avg_answer_consistency": cot_stats.get("avg_answer_consistency", 0)
                }

        if results["adapter_model"]:
            adapter_lengths = [len(r["model_response"]) for r in results["adapter_model"]]
            adapter_issues = sum(1 for r in results["adapter_model"] if "警告" in r["model_response"])
            summary["adapter_model"] = {
                "avg_response_length": sum(adapter_lengths) / len(adapter_lengths),
                "domain_issues": adapter_issues
            }
            if cot_results and "adapter_model" in cot_results:
                cot_stats = cot_results["adapter_model"]["statistics"]
                summary["adapter_model"]["cot_quality"] = {
                    "has_think_block_ratio": cot_stats.get("has_think_block_ratio", 0),
                    "avg_reasoning_quality": cot_stats.get("avg_reasoning_quality", 0),
                    "avg_answer_consistency": cot_stats.get("avg_answer_consistency", 0)
                }

        serializable_cot_results = {}
        if cot_results:
            for model_type in ["base_model", "adapter_model"]:
                if model_type in cot_results:
                    cot_data = cot_results[model_type]
                    serializable_cot_results[model_type] = {
                        "statistics": cot_data.get("statistics", {}),
                        "individual_results": []
                    }
                    for item in cot_data.get("individual_results", []):
                        serializable_item = {
                            "prompt": item.get("prompt", ""),
                            "raw_response": item.get("raw_response", "")
                        }
                        if "validation" in item:
                            serializable_item["validation"] = self._cot_result_to_dict(item["validation"])
                        serializable_cot_results[model_type]["individual_results"].append(serializable_item)

        # 构建完整结果字典
        full_results = {
            "summary": summary,
            "detailed_results": results,          # 关键：包含详细样本结果
            "cot_validation": serializable_cot_results,
            "validation_config": {
                "model": self.args.model,
                "adapter": self.args.adapter,
                "dataset": self.args.dataset,
                "max_samples": self.args.max_samples,
                "max_seq_length": self.args.max_seq_length,
                "domain": self.args.domain
            },
            "output_path": output_path
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)

        if cot_results:
            for model_type in ["base_model", "adapter_model"]:
                if model_type in cot_results:
                    report_path = os.path.join(self.output_dir, f"cot_report_{model_type}.txt")
                    self.cot_validator.generate_report(cot_results[model_type], output_path=report_path)

        self._print_summary_report(summary, output_path)
        return full_results   # 修改：返回包含 detailed_results 的完整字典

    def _print_summary_report(self, summary: Dict[str, Any], output_path: str):
        print("\n" + "="*70)
        print(" "*20 + "模型验证摘要报告")
        print("="*70)
        print(f"\n【基本信息】")
        print(f"  验证样本数: {summary['total_samples']}")
        print(f"  基础模型: {summary['model_info']['base_model']}")
        if summary['model_info']['adapter']:
            print(f"  微调适配器: {summary['model_info']['adapter']}")
        print(f"\n【基础模型表现】")
        if "base_model" in summary:
            base = summary["base_model"]
            print(f"  平均响应长度: {base['avg_response_length']:.0f} 字符")
            print(f"  领域警告数: {base['domain_issues']}/{summary['total_samples']}")
            if "cot_quality" in base:
                cot = base["cot_quality"]
                print(f"  思维链包含率: {cot['has_think_block_ratio']*100:.1f}%")
                print(f"  推理质量评分: {cot['avg_reasoning_quality']*100:.1f}%")
                print(f"  答案一致性: {cot['avg_answer_consistency']*100:.1f}%")
        else:
            print("  未测试基础模型")
        if "adapter_model" in summary:
            print(f"\n【微调模型表现】")
            adapter = summary["adapter_model"]
            print(f"  平均响应长度: {adapter['avg_response_length']:.0f} 字符")
            print(f"  领域警告数: {adapter['domain_issues']}/{summary['total_samples']}")
            if "cot_quality" in adapter:
                cot = adapter["cot_quality"]
                print(f"  思维链包含率: {cot['has_think_block_ratio']*100:.1f}%")
                print(f"  推理质量评分: {cot['avg_reasoning_quality']*100:.1f}%")
                print(f"  答案一致性: {cot['avg_answer_consistency']*100:.1f}%")
            if "base_model" in summary:
                print(f"\n【微调效果对比】")
                base_len = summary["base_model"]["avg_response_length"]
                adapter_len = adapter["avg_response_length"]
                len_change = ((adapter_len - base_len) / base_len * 100) if base_len > 0 else 0
                print(f"  响应长度变化: {len_change:+.1f}% ({base_len:.0f} -> {adapter_len:.0f})")
                if "cot_quality" in summary["base_model"] and "cot_quality" in adapter:
                    base_cot = summary["base_model"]["cot_quality"]["avg_reasoning_quality"]
                    adapter_cot = adapter["cot_quality"]["avg_reasoning_quality"]
                    cot_change = (adapter_cot - base_cot) * 100
                    print(f"  推理质量变化: {cot_change:+.1f}% ({base_cot*100:.1f}% -> {adapter_cot*100:.1f}%)")
        print(f"\n【输出文件】")
        print(f"  详细结果: {output_path}")
        cot_report_base = os.path.join(self.output_dir, "cot_report_base_model.txt")
        cot_report_adapter = os.path.join(self.output_dir, "cot_report_adapter_model.txt")
        if os.path.exists(cot_report_base):
            print(f"  基础模型CoT报告: {cot_report_base}")
        if os.path.exists(cot_report_adapter):
            print(f"  微调模型CoT报告: {cot_report_adapter}")
        print("\n" + "="*70)
        print(f"\n【快速评估】")
        if "adapter_model" in summary and "base_model" in summary:
            adapter_cot = summary["adapter_model"].get("cot_quality", {}).get("avg_reasoning_quality", 0)
            base_cot = summary["base_model"].get("cot_quality", {}).get("avg_reasoning_quality", 0)
            if adapter_cot > base_cot * 1.1:
                print("  ✓ 微调模型推理质量明显提升")
            elif adapter_cot > base_cot * 0.9:
                print("  ≈ 微调模型推理质量与基础模型相当")
            else:
                print("  ✗ 微调模型推理质量下降，建议检查训练数据")
            adapter_len = summary["adapter_model"]["avg_response_length"]
            if adapter_len > 500:
                print("  ✓ 响应详细程度良好")
            elif adapter_len > 200:
                print("  ~ 响应详细程度一般")
            else:
                print("  ! 响应过于简短，可能存在生成问题")
        print("\n" + "="*70 + "\n")