# src/core/dataset_factory.py - 完整修改版
import os
import json
import logging
import random
import re
from typing import List, Union, Optional, Dict
from datasets import load_dataset, Dataset, concatenate_datasets

logger = logging.getLogger(__name__)

class DatasetFactory:
    def __init__(self, dataset_info_path=None):
        if dataset_info_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(os.path.dirname(current_dir))
            dataset_info_path = os.path.join(base_dir, "datasets", "dataset_info.json")
        self.dataset_info = self.load_dataset_info(dataset_info_path)

    def load_dataset_info(self, path):
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

    def create_dataset(self, file_path, format_name, data_limit=None,
                       mixing_strategy="concat", weights=None, domain=None, return_raw=False):
        file_paths = self._parse_file_paths(file_path)

        if len(file_paths) == 1:
            dataset = self._load_single_dataset(file_paths[0], format_name, data_limit, domain=domain)
            logger.info(f"单数据集加载完成: {len(dataset)} 样本")
            if return_raw:
                return dataset, self._raw_samples
            return dataset

        datasets = []
        for i, single_path in enumerate(file_paths):
            try:
                individual_limit = None
                if data_limit and weights and mixing_strategy == "weighted":
                    individual_limit = int(data_limit * weights[i] / sum(weights))
                elif data_limit and mixing_strategy == "concat":
                    individual_limit = data_limit // len(file_paths)

                dataset = self._load_single_dataset(single_path, format_name, individual_limit, domain=domain)
                if dataset is not None and len(dataset) > 0:
                    datasets.append(dataset)
                    logger.info(f"成功加载: {single_path} ({len(dataset)} 样本)")
                else:
                    logger.warning(f"数据集为空或加载失败: {single_path}")
            except Exception as e:
                logger.error(f"加载失败 {single_path}: {str(e)}")
                continue

        if not datasets:
            raise ValueError("所有数据集文件加载失败")

        combined = self._mix_datasets(datasets, mixing_strategy, weights, data_limit)
        logger.info(f"多数据集混合完成: {len(file_paths)}个文件 -> {len(combined)}总样本")
        if return_raw:
            return combined, self._raw_samples
        return combined

    def _parse_file_paths(self, file_path):
        if isinstance(file_path, str):
            if ',' in file_path:
                return [p.strip() for p in file_path.split(',') if p.strip()]
            else:
                return [file_path.strip()]
        elif isinstance(file_path, list):
            return [str(p).strip() for p in file_path if str(p).strip()]
        else:
            return [str(file_path).strip()]

    def _mix_datasets(self, datasets: List[Dataset], strategy: str,
                      weights: Optional[List[float]], total_limit: Optional[int]) -> Dataset:
        if strategy == "concat":
            combined = concatenate_datasets(datasets)
        elif strategy == "interleave":
            combined = self._interleave_datasets(datasets)
        elif strategy == "weighted":
            combined = self._weighted_mix_datasets(datasets, weights, total_limit)
        else:
            logger.warning(f"未知混合策略 {strategy}，使用concat")
            combined = concatenate_datasets(datasets)

        if total_limit and len(combined) > total_limit:
            logger.info(f"应用总体数据限制: {len(combined)} -> {total_limit}")
            indices = random.sample(range(len(combined)), total_limit)
            combined = combined.select(indices)

        return combined

    def _interleave_datasets(self, datasets: List[Dataset]) -> Dataset:
        min_len = min(len(d) for d in datasets)
        interleaved = []
        for i in range(min_len):
            for dataset in datasets:
                interleaved.append(dataset[i])
        for dataset in datasets:
            if len(dataset) > min_len:
                for i in range(min_len, len(dataset)):
                    interleaved.append(dataset[i])
        return Dataset.from_list(interleaved)

    def _weighted_mix_datasets(self, datasets: List[Dataset],
                                weights: Optional[List[float]],
                                total_limit: Optional[int]) -> Dataset:
        if weights is None or len(weights) != len(datasets):
            logger.warning("权重不匹配，使用平均权重")
            weights = [1.0] * len(datasets)
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        if total_limit:
            sample_counts = [int(total_limit * w) for w in normalized_weights]
            diff = total_limit - sum(sample_counts)
            sample_counts[0] += diff
        else:
            base_size = min(len(d) for d in datasets)
            sample_counts = [int(base_size * w / min(normalized_weights)) for w in normalized_weights]

        sampled_datasets = []
        for dataset, count in zip(datasets, sample_counts):
            if count >= len(dataset):
                sampled_datasets.append(dataset)
            else:
                indices = random.sample(range(len(dataset)), count)
                sampled_datasets.append(dataset.select(indices))
        return concatenate_datasets(sampled_datasets)

    def _load_single_dataset(self, file_path, format_name, data_limit=None, domain=None):
        file_path = str(file_path).strip()
        if not file_path:
            raise ValueError("文件路径为空")

        if format_name not in self.dataset_info:
            logger.warning(f"不支持的数据集格式: {format_name}，尝试使用alpaca格式")
            format_name = "alpaca"

        loader_name = f"load_{format_name}_dataset"
        if not hasattr(self, loader_name):
            logger.warning(f"加载器 {loader_name} 未实现，使用默认的Alpaca加载器")
            loader = self.load_alpaca_dataset
        else:
            loader = getattr(self, loader_name)

        try:
            dataset = loader(file_path, data_limit, domain=domain)
            if dataset is None:
                raise ValueError("加载器返回 None")
            logger.info(f"成功加载 {format_name} 数据集: {file_path}，样本数: {len(dataset)}")
            return dataset
        except Exception as e:
            logger.error(f"加载数据集失败: {str(e)}")
            raise

    # ---------- 新增：用于验证时获取原始样本 ----------
    _raw_samples = None

    def load_alpaca_dataset(self, file_path, data_limit=None, domain=None):
        self._raw_samples = []  # 重置

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")

        if file_path.endswith('.json') or file_path.endswith('.jsonl'):
            try:
                if file_path.endswith('.json'):
                    full_dataset = load_dataset("json", data_files=file_path)["train"]
                else:
                    full_dataset = load_dataset("json", data_files=file_path, split="train")

                if data_limit is not None and data_limit > 0:
                    actual_limit = min(data_limit, len(full_dataset))
                    dataset = full_dataset.select(range(actual_limit))
                else:
                    dataset = full_dataset
            except Exception as e:
                logger.error(f"加载数据集文件失败: {str(e)}")
                raise
        else:
            raise ValueError(f"不支持的格式: {file_path}")

        processed = []
        skipped_count = 0

        for i, item in enumerate(dataset):
            try:
                if not item:
                    skipped_count += 1
                    continue

                instruction = item.get("instruction")
                output = item.get("output")

                if not instruction or not output:
                    skipped_count += 1
                    continue

                if not isinstance(instruction, str):
                    instruction = str(instruction)
                if not isinstance(output, str):
                    output = str(output)

                if not instruction.strip() or not output.strip():
                    skipped_count += 1
                    continue

                # 保存原始样本（用于验证）
                self._raw_samples.append({
                    "instruction": instruction.strip(),
                    "input": item.get("input", "").strip() if item.get("input") else "",
                    "output": output.strip()
                })

                text = self.apply_alpaca_template(item, domain=domain)
                if text and text.strip():
                    processed.append({"text": text})
                else:
                    skipped_count += 1

            except Exception as e:
                logger.warning(f"处理样本 {i} 时出错: {str(e)}")
                skipped_count += 1
                continue

        if skipped_count > 0:
            logger.warning(f"跳过了 {skipped_count} 个无效样本")

        if not processed:
            raise ValueError(f"数据集 {file_path} 中没有有效样本")

        if data_limit is not None and len(processed) > data_limit:
            processed = processed[:data_limit]
            self._raw_samples = self._raw_samples[:data_limit]

        return Dataset.from_list(processed)

    # ========== 核心修改：医学领域强制系统提示，与chat.py完全一致 ==========
    def _get_forced_system_prompt(self) -> str:
        """返回与chat.py完全相同的强制CoT系统提示"""
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

    def apply_alpaca_template(self, item, domain=None):
        text = ""

        if domain == "medical":
            # 强制使用与chat.py完全相同的系统提示
            text += f"<|system|>\n{self._get_forced_system_prompt()}</s>\n"
        else:
            system_prompt = item.get("system", "")
            if system_prompt and isinstance(system_prompt, str) and system_prompt.strip():
                text += f"<|system|>\n{system_prompt.strip()}</s>\n"

        history = item.get("history", [])
        if history and isinstance(history, list):
            for hist in history:
                if (isinstance(hist, list) and len(hist) == 2 and
                    isinstance(hist[0], str) and hist[0].strip() and
                    isinstance(hist[1], str) and hist[1].strip()):
                    text += f"<|user|>\n{hist[0].strip()}</s>\n"
                    text += f"<|assistant|>\n{hist[1].strip()}</s>\n"

        instruction = item.get("instruction", "")
        if not isinstance(instruction, str):
            instruction = str(instruction)
        instruction = instruction.strip()
        if not instruction:
            return ""

        text += f"<|user|>\n{instruction}"

        input_text = item.get("input")
        if input_text is None:
            input_text = ""
        elif not isinstance(input_text, str):
            input_text = str(input_text)

        input_text = input_text.strip()
        if input_text:
            text += f" {input_text}"

        text += "</s>\n"

        output = item.get("output", "")
        if not isinstance(output, str):
            output = str(output)
        output = output.strip()
        if not output:
            return ""

        if "<think>" in output:
            output = re.sub(r'<\s*think\s*>', '<think>', output, flags=re.IGNORECASE)
            output = re.sub(r'<\s*/\s*think\s*>', '</think>', output, flags=re.IGNORECASE)
            output = re.sub(r'<think>\s*', '<think>\n', output)
            output = re.sub(r'\s*</think>', '\n</think>\n', output)

        text += f"<|assistant|>\n{output}</s>"
        return text

    # ---------- 其他数据集加载方法（保留占位，完整实现请按需补充）----------
    def load_sharegpt_dataset(self, file_path, data_limit=None, domain=None):
        raise NotImplementedError("ShareGPT格式暂未实现，请使用alpaca格式")

    def load_preference_dataset(self, file_path, data_limit=None, domain=None):
        raise NotImplementedError("偏好数据集格式暂未实现，请使用alpaca格式")