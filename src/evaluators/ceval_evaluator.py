# src/evaluators/ceval_evaluator.py
import os
import csv
import json
import logging
import numpy as np
from tqdm import tqdm
from src.utils.helpers import extract_answer_letter, log_sample_debug
import inspect  # 新增导入

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_column_name(name):
    """标准化列名：去除空格、转换为小写、处理中文"""
    name = name.strip().lower()
    # 处理常见中文列名
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

def load_ceval_dataset(task_dir, subject, split="val"):
    """加载C-Eval数据集，支持多种列名格式"""
    file_path = os.path.join(task_dir, split, f"{subject}_{split}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        # 检测文件编码
        try:
            sample = f.read(1024)
            f.seek(0)
            if "�" in sample:
                logger.warning(f"检测到编码问题，尝试使用GBK编码: {file_path}")
                f = open(file_path, "r", encoding="gbk")
        except:
            pass
        
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV文件没有列名: {file_path}")
        
        # 标准化列名
        fieldnames = [normalize_column_name(name) for name in reader.fieldnames]
        reader.fieldnames = fieldnames
        
        # 验证必要列是否存在
        required_columns = {"question", "a", "b", "c", "d", "answer"}  # 验证集必须包含答案
        
        missing_columns = required_columns - set(fieldnames)
        if missing_columns:
            raise ValueError(f"文件 {file_path} 缺少必要列: {', '.join(missing_columns)}")
        
        for i, row in enumerate(reader):
            try:
                # 确保问题列存在
                question = row.get("question", "").strip()
                if not question:
                    logger.warning(f"第 {i+1} 行问题为空")
                    continue
                
                # 获取答案
                answer = row.get("answer", "").strip().upper()
                
                # 验证答案格式
                if answer not in ["A", "B", "C", "D"]:
                    logger.warning(f"第 {i+1} 行答案无效: {answer}")
                    continue  # 跳过无效答案样本
                
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
                logger.error(f"处理第 {i+1} 行数据时出错: {str(e)}")
                logger.debug(f"问题行数据: {row}")
    
    if not data:
        raise ValueError(f"文件 {file_path} 没有有效数据")
    
    logger.info(f"成功加载 {len(data)} 条样本: {file_path}")
    return data

def build_ceval_prompt(item, examples, lang="zh", template=None):
    """构建C-Eval提示模板"""
    prompt = ""
    
    # 添加few-shot示例
    for i, example in enumerate(examples):
        prompt += f"问题 {i+1}: {example['question']}\n"
        for j, choice in enumerate(example["choices"]):
            prompt += f"{chr(65+j)}. {choice}\n"
        
        # 只有验证集示例才有答案
        if example.get("answer"):
            prompt += f"答案: {example['answer']}\n\n"
        else:
            prompt += "答案:\n\n"
    
    # 添加当前问题
    prompt += f"问题: {item['question']}\n"
    for i, choice in enumerate(item["choices"]):
        prompt += f"{chr(65+i)}. {choice}\n"
    
    # 中文提示
    if lang == "zh":
        prompt += "答案:"
    else:
        prompt += "Answer:"
    
    return prompt

def calculate_accuracy(predictions, references):
    """手动计算准确率"""
    correct = 0
    for pred, ref in zip(predictions, references):
        if pred == ref:
            correct += 1
    return correct / len(predictions) if predictions else 0

def get_supported_generation_args(model):
    """获取模型支持的生成参数"""
    try:
        # 检查模型是否定义了支持的参数
        if hasattr(model, "supported_generation_args"):
            return model.supported_generation_args
        
        # 检查模型配置
        if hasattr(model, "config") and hasattr(model.config, "supported_generation_args"):
            return model.config.supported_generation_args
        
        # 尝试获取生成函数的参数
        generate_params = inspect.signature(model.generate).parameters.keys()
        return list(generate_params)
    except Exception as e:
        logger.warning(f"获取模型支持的生成参数失败: {str(e)}")
        return ["max_new_tokens", "num_return_sequences", "pad_token_id"]  # 返回基本参数

def evaluate_ceval(model, tokenizer, task_dir, n_shot, lang, save_dir=None):
    """评估模型在C-Eval验证集上的表现"""
    # 设置模型为评估模式
    model.eval()
    
    # 获取模型支持的生成参数
    supported_args = get_supported_generation_args(model)
    logger.info(f"模型支持的生成参数: {', '.join(supported_args)}")
    
    # 获取所有学科
    subjects = []
    val_dir = os.path.join(task_dir, "val")  # 使用验证集目录
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"验证目录不存在: {val_dir}")
    
    for filename in os.listdir(val_dir):
        if filename.endswith("_val.csv"):
            subject = filename.replace("_val.csv", "")
            subjects.append(subject)
    
    if not subjects:
        raise ValueError(f"未找到验证文件: {val_dir}")
    
    logger.info(f"找到 {len(subjects)} 个学科: {', '.join(subjects[:3])}...")
    
    results = {}
    accs = []
    subject_stats = {}
    detailed_results = []  # 存储详细结果
    
    # 创建进度条
    progress_bar = tqdm(subjects, desc="评估CEval学科")
    
    for subject in progress_bar:
        try:
            logger.info(f"开始评估学科: {subject}")
            
            # 加载验证集（带答案）
            val_data = load_ceval_dataset(task_dir, subject, "val")
            logger.info(f"加载 {len(val_data)} 条验证集样本")
            
            # 随机选择few-shot示例
            few_shot_examples = []
            if n_shot > 0:
                # 确保只选择有答案的样本
                valid_val_data = [item for item in val_data if item.get("answer")]
                if not valid_val_data:
                    logger.warning("验证集中没有有效答案，将不使用few-shot")
                else:
                    # 选择前n_shot个有效样本
                    few_shot_examples = valid_val_data[:min(n_shot, len(valid_val_data))]
                    logger.info(f"使用 {len(few_shot_examples)} 条few-shot示例")
            
            # 评估该学科
            predictions = []
            references = []
            sample_debug = []
            
            for idx, item in enumerate(val_data):
                try:
                    # 构建提示
                    prompt = build_ceval_prompt(item, few_shot_examples, lang)
                    
                    # 生成答案
                    inputs = tokenizer(
                        [prompt],
                        return_tensors="pt",
                        truncation=True,
                        max_length=tokenizer.model_max_length,
                    )
                    
                    # 将输入移动到模型所在的设备
                    device = model.device if hasattr(model, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # 基础生成参数
                    generation_kwargs = {
                        "max_new_tokens": 10,
                        "num_return_sequences": 1,
                        "pad_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
                    }
                    
                    # 可选参数 - 只在模型支持时添加
                    if "temperature" in supported_args:
                        generation_kwargs["temperature"] = 0.7
                    if "top_p" in supported_args:
                        generation_kwargs["top_p"] = 0.9
                    if "top_k" in supported_args:
                        generation_kwargs["top_k"] = 50
                    if "do_sample" in supported_args:
                        generation_kwargs["do_sample"] = True
                    
                    # 生成响应 - 使用智能参数设置
                    outputs = model.generate(
                        **inputs,
                        **generation_kwargs
                    )
                    
                    # 解码输出
                    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # 提取预测答案
                    pred = extract_answer_letter(output)
                    
                    # 记录结果
                    predictions.append(pred)
                    references.append(item["answer"])
                    
                    # 保存详细结果
                    detailed_results.append({
                        "subject": subject,
                        "question": item["question"],
                        "choices": item["choices"],
                        "reference_answer": item["answer"],
                        "model_prediction": pred,
                        "correct": pred == item["answer"],
                        "model_output": output
                    })
                    
                    # 保存前5个样本的调试信息
                    if idx < 5:
                        sample_debug.append({
                            "prompt": prompt,
                            "output": output,
                            "prediction": pred,
                            "reference": item["answer"]
                        })
                except Exception as e:
                    logger.error(f"处理样本 {idx} 失败: {str(e)}")
                    predictions.append("")
                    references.append(item["answer"])
            
            # 计算准确率
            acc = calculate_accuracy(predictions, references)
            accs.append(acc)
            results[subject] = acc
            subject_stats[subject] = {
                "accuracy": acc,
                "total": len(val_data),
                "correct": sum(1 for p, r in zip(predictions, references) if p == r),
                "sample_debug": sample_debug
            }
            
            logger.info(f"学科 {subject} 评估完成 | 准确率: {acc:.4f}")
            progress_bar.set_description(f"评估 {subject} | 准确率: {acc:.4f}")
            
        except Exception as e:
            logger.error(f"评估学科 {subject} 失败: {str(e)}", exc_info=True)
            results[subject] = 0
            continue
    
    # 计算平均准确率
    results["average"] = np.mean(accs) if accs else 0
    
    # STEM学科平均（示例，可根据实际学科分类调整）
    stem_subjects = ["physics", "chemistry", "biology", "math"]
    stem_accs = [acc for sub, acc in results.items() if any(s in sub.lower() for s in stem_subjects)]
    results["stem_average"] = np.mean(stem_accs) if stem_accs else 0
    
    # 性能评级
    avg_acc = results["average"]
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
        # 保存学科结果
        subject_path = os.path.join(save_dir, "ceval_subject_results.json")
        with open(subject_path, "w", encoding="utf-8") as f:
            json.dump(subject_stats, f, ensure_ascii=False, indent=2)
        
        # 保存详细结果
        detailed_path = os.path.join(save_dir, "ceval_detailed_results.json")
        with open(detailed_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        # 保存总体结果
        summary_path = os.path.join(save_dir, "ceval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "average_accuracy": results["average"],
                "stem_average": results["stem_average"] if "stem_average" in results else 0,
                "performance_rating": results["performance_rating"],
                "subject_count": len(subjects),
                "evaluated_subjects": len(accs),
                "total_questions": len(detailed_results),
                "correct_answers": sum(1 for r in detailed_results if r["correct"])
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果保存至: {save_dir}")
    
    return results