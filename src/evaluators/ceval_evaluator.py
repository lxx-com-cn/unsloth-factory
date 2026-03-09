# src/evaluators/ceval_evaluator.py
import os
import csv
import json
import logging
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from src.utils.helpers import extract_answer_letter

logger = logging.getLogger(__name__)

def normalize_column_name(name: str) -> str:
    """标准化列名：去除空格、转换为小写、处理中文"""
    name = name.strip().lower()
    name_mapping = {
        "问题": "question",
        "答案": "answer",
        "正确选项": "answer",
        "选项a": "a",
        "选项b": "b",
        "选项c": "c",
        "选项d": "d",
    }
    return name_mapping.get(name, name)

def load_ceval_dataset(task_dir: str, subject: str, split: str = "val") -> List[Dict]:
    """加载 C-Eval 数据集，支持多种列名格式"""
    file_path = os.path.join(task_dir, split, f"{subject}_{split}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        # 尝试检测编码
        sample = f.read(1024)
        f.seek(0)
        if "�" in sample:
            logger.warning(f"检测到编码问题，尝试使用 GBK 编码: {file_path}")
            f = open(file_path, "r", encoding="gbk")
        
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV 文件没有列名: {file_path}")
        
        # 标准化列名
        fieldnames = [normalize_column_name(name) for name in reader.fieldnames]
        reader.fieldnames = fieldnames
        
        # 验证必要列
        required_columns = {"question", "a", "b", "c", "d", "answer"}
        missing_columns = required_columns - set(fieldnames)
        if missing_columns:
            raise ValueError(f"文件 {file_path} 缺少必要列: {', '.join(missing_columns)}")
        
        for i, row in enumerate(reader):
            try:
                question = row.get("question", "").strip()
                if not question:
                    logger.warning(f"第 {i+1} 行问题为空")
                    continue
                
                answer = row.get("answer", "").strip().upper()
                if answer not in ["A", "B", "C", "D"]:
                    logger.warning(f"第 {i+1} 行答案无效: {answer}")
                    continue
                
                data.append({
                    "id": row.get("id", f"{subject}_{split}_{i}"),
                    "question": question,
                    "choices": [
                        row.get("a", "").strip(),
                        row.get("b", "").strip(),
                        row.get("c", "").strip(),
                        row.get("d", "").strip()
                    ],
                    "answer": answer
                })
            except Exception as e:
                logger.error(f"处理第 {i+1} 行数据时出错: {e}")
                continue
    
    if not data:
        raise ValueError(f"文件 {file_path} 没有有效数据")
    
    logger.info(f"成功加载 {len(data)} 条样本: {file_path}")
    return data

def build_ceval_prompt(item: Dict, examples: List[Dict], lang: str = "zh") -> str:
    """构建 C-Eval 提示模板"""
    prompt = ""
    
    # 添加 few-shot 示例
    for i, ex in enumerate(examples):
        prompt += f"问题 {i+1}: {ex['question']}\n"
        for j, choice in enumerate(ex["choices"]):
            prompt += f"{chr(65+j)}. {choice}\n"
        if ex.get("answer"):
            prompt += f"答案: {ex['answer']}\n\n"
        else:
            prompt += "答案:\n\n"
    
    # 当前问题
    prompt += f"问题: {item['question']}\n"
    for i, choice in enumerate(item["choices"]):
        prompt += f"{chr(65+i)}. {choice}\n"
    
    prompt += "答案:" if lang == "zh" else "Answer:"
    return prompt

def evaluate_ceval(
    model,
    tokenizer,
    task_dir: str,
    n_shot: int,
    lang: str,
    save_dir: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 10,
) -> Dict[str, Any]:
    """评估模型在 C-Eval 验证集上的表现"""
    model.eval()
    
    # 获取所有学科
    subjects = []
    val_dir = os.path.join(task_dir, "val")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"验证目录不存在: {val_dir}")
    
    for filename in os.listdir(val_dir):
        if filename.endswith("_val.csv"):
            subject = filename.replace("_val.csv", "")
            subjects.append(subject)
    
    if not subjects:
        raise ValueError(f"未找到验证文件: {val_dir}")
    
    logger.info(f"找到 {len(subjects)} 个学科: {', '.join(subjects[:5])}...")
    
    results = {}
    accs = []
    subject_stats = {}
    detailed_results = []  # 存储每个样本的详细信息
    
    progress_bar = tqdm(subjects, desc="评估 C-Eval 学科")
    for subject in progress_bar:
        try:
            logger.info(f"开始评估学科: {subject}")
            val_data = load_ceval_dataset(task_dir, subject, "val")
            
            # 选择 few-shot 示例（从验证集中取前 n_shot 个有答案的样本）
            few_shot_examples = []
            if n_shot > 0:
                valid_examples = [item for item in val_data if item.get("answer")]
                if valid_examples:
                    few_shot_examples = valid_examples[:min(n_shot, len(valid_examples))]
                    logger.info(f"使用 {len(few_shot_examples)} 条 few-shot 示例")
            
            predictions = []
            references = []
            subject_details = []
            
            for idx, item in enumerate(val_data):
                try:
                    prompt = build_ceval_prompt(item, few_shot_examples, lang)
                    
                    inputs = tokenizer(
                        [prompt],
                        return_tensors="pt",
                        truncation=True,
                        max_length=tokenizer.model_max_length,
                    )
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    generation_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "num_return_sequences": 1,
                        "pad_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "do_sample": True,
                    }
                    
                    with torch.no_grad():
                        outputs = model.generate(**inputs, **generation_kwargs)
                    
                    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    pred = extract_answer_letter(output_text)
                    
                    predictions.append(pred)
                    references.append(item["answer"])
                    
                    subject_details.append({
                        "id": item["id"],
                        "question": item["question"],
                        "choices": item["choices"],
                        "reference_answer": item["answer"],
                        "model_prediction": pred,
                        "correct": pred == item["answer"],
                        "model_output": output_text,
                    })
                except Exception as e:
                    logger.error(f"处理样本 {idx} 失败: {e}")
                    predictions.append("")
                    references.append(item["answer"])
                    subject_details.append({
                        "id": item.get("id", f"{subject}_{idx}"),
                        "question": item.get("question", ""),
                        "choices": item.get("choices", ["", "", "", ""]),
                        "reference_answer": item.get("answer", ""),
                        "model_prediction": "",
                        "correct": False,
                        "model_output": "",
                        "error": str(e)
                    })
            
            # 计算准确率
            correct = sum(1 for p, r in zip(predictions, references) if p == r)
            acc = correct / len(val_data) if val_data else 0
            accs.append(acc)
            results[subject] = acc
            
            subject_stats[subject] = {
                "accuracy": acc,
                "total": len(val_data),
                "correct": correct,
            }
            
            detailed_results.extend(subject_details)
            logger.info(f"学科 {subject} 评估完成 | 准确率: {acc:.4f}")
            progress_bar.set_description(f"评估 {subject} | 准确率: {acc:.4f}")
            
        except Exception as e:
            logger.error(f"评估学科 {subject} 失败: {e}", exc_info=True)
            results[subject] = 0.0
            subject_stats[subject] = {"accuracy": 0.0, "total": 0, "correct": 0, "error": str(e)}
            continue
    
    # 计算平均准确率
    avg_acc = np.mean(accs) if accs else 0.0
    results["average"] = avg_acc
    
    # STEM 学科平均（示例，可根据实际分类调整）
    stem_keywords = ["physics", "chemistry", "biology", "math", "computer", "engineering"]
    stem_accs = [acc for sub, acc in results.items() if any(k in sub.lower() for k in stem_keywords) and sub not in ["average", "stem_average"]]
    results["stem_average"] = np.mean(stem_accs) if stem_accs else 0.0
    
    # 性能评级
    if avg_acc >= 0.8:
        results["performance_rating"] = "优秀"
    elif avg_acc >= 0.6:
        results["performance_rating"] = "良好"
    elif avg_acc >= 0.4:
        results["performance_rating"] = "一般"
    else:
        results["performance_rating"] = "需改进"
    
    # 保存详细结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 学科结果
        subject_path = os.path.join(save_dir, "ceval_subject_results.json")
        with open(subject_path, "w", encoding="utf-8") as f:
            json.dump(subject_stats, f, ensure_ascii=False, indent=2)
        
        # 详细样本结果
        detailed_path = os.path.join(save_dir, "ceval_detailed_results.json")
        with open(detailed_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        # 总体摘要
        summary_path = os.path.join(save_dir, "ceval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "average_accuracy": avg_acc,
                "stem_average": results["stem_average"],
                "performance_rating": results["performance_rating"],
                "subject_count": len(subjects),
                "evaluated_subjects": len(accs),
                "total_questions": len(detailed_results),
                "correct_answers": sum(1 for r in detailed_results if r.get("correct")),
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存至: {save_dir}")
    
    return results