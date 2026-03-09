#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert_quant.py - 使用 llm-compressor 进行模型量化（T4 16GB 深度优化版）
核心策略：FP16加载 + CPU卸载 + 使用正确的API参数 + 逐层量化避免OOM
"""

import os
import sys
import json
import argparse
import logging
import torch
import gc
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_project_root(start_path: str) -> str:
    """查找项目根目录"""
    current = os.path.abspath(start_path)
    if os.path.isfile(current):
        current = os.path.dirname(current)
    
    markers = ["output", "datasets", "cli.py", "pytree.py", "src"]
    for _ in range(10):
        if any(os.path.exists(os.path.join(current, m)) for m in markers):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    
    cwd = os.getcwd()
    for m in markers:
        if os.path.exists(os.path.join(cwd, m)):
            return cwd
    
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def check_llmcompressor() -> Tuple[bool, Dict[str, bool]]:
    """检查 llm-compressor 可用性及支持的量化方案"""
    result = {"available": False, "GPTQ": False, "AWQ": False, "string_recipe": False}
    
    try:
        from llmcompressor import oneshot
        result["available"] = True
        logger.info("llm-compressor 核心功能可用")
    except ImportError as e:
        logger.error(f"llm-compressor 未安装: {e}")
        return False, result
    
    try:
        from llmcompressor.modifiers.quantization import GPTQModifier
        result["GPTQ"] = True
        logger.info("GPTQModifier 可用")
    except ImportError:
        logger.warning("GPTQModifier 不可用，将使用字符串 recipe")
    
    try:
        from llmcompressor.modifiers.quantization import AWQModifier
        result["AWQ"] = True
        logger.warning("AWQModifier 可用，但AWQ比GPTQ更耗显存，T4上建议用GPTQ")
    except ImportError:
        logger.info("AWQModifier 不可用（对T4来说是好事，AWQ更耗显存）")
    
    try:
        import llmcompressor
        version = getattr(llmcompressor, "__version__", "unknown")
        logger.info(f"llm-compressor 版本: {version}")
        result["string_recipe"] = True
    except:
        pass
    
    return True, result


def generate_medical_calibration_data(num_samples: int = 16) -> List[Dict[str, str]]:
    """
    自动生成医学领域校准数据（T4优化：默认16条，减少内存压力）
    返回格式: [{"text": "..."}, {"text": "..."}]
    """
    templates = [
        "什么是{disease}？{disease}是一种{description}，主要表现为{symptom}。",
        "{disease}的主要病因包括{cause}，此外{risk_factor}也是重要危险因素。",
        "{disease}的诊断需要依靠{diagnosis}，治疗方法包括{treatment}。",
        "患者主诉{symptom}，检查发现{disease}，建议{treatment}。",
        "{disease}常用药物包括{medication}，需遵医嘱使用，注意{side_effect}。",
        "预防{disease}应做到{prevention}，定期{diagnosis}有助于早期发现。",
        "患者{age}岁，{gender}，主诉{symptom}，诊断为{disease}，予{treatment}。",
        "{disease}可能并发{complication}，需警惕{symptom}，及时{treatment}。",
    ]
    
    diseases = [
        "高血压", "糖尿病", "冠心病", "脑卒中", "心肌梗死", "慢性阻塞性肺疾病",
        "肺炎", "哮喘", "慢性胃炎", "胃溃疡", "肝硬化", "脂肪肝",
        "肾炎", "肾结石", "前列腺增生", "早泄", "抑郁症", "焦虑症",
    ]
    
    descriptions = [
        "常见的慢性疾病", "急性炎症性疾病", "退行性病变", "代谢性疾病",
        "心血管疾病", "消化系统疾病", "呼吸系统疾病", "内分泌疾病",
    ]
    
    symptoms = [
        "头晕头痛", "胸闷心悸", "呼吸困难", "咳嗽咳痰", "恶心呕吐",
        "腹痛腹泻", "尿频尿急", "焦虑抑郁", "失眠多梦", "乏力消瘦",
    ]
    
    causes = [
        "遗传因素", "环境因素", "生活方式", "感染因素", "免疫异常",
        "代谢紊乱", "血管病变", "长期吸烟饮酒", "高盐高脂饮食",
    ]
    
    risk_factors = [
        "年龄增长", "家族史", "肥胖", "高血压", "糖尿病", "高血脂",
    ]
    
    diagnoses = [
        "体格检查", "血液检查", "影像学检查", "心电图", "超声检查",
    ]
    
    treatments = [
        "药物治疗", "手术治疗", "介入治疗", "放射治疗", "化学治疗",
    ]
    
    medications = [
        "降压药", "降糖药", "降脂药", "抗凝药", "抗生素",
    ]
    
    side_effects = [
        "胃肠道反应", "肝肾功能损害", "骨髓抑制", "过敏反应",
    ]
    
    preventions = [
        "戒烟限酒", "低盐低脂饮食", "规律运动", "控制体重", "定期体检",
    ]
    
    complications = [
        "心力衰竭", "肾功能衰竭", "脑卒中", "心肌梗死",
    ]
    
    ages = ["35", "55", "75"]
    genders = ["男性", "女性"]
    
    import random
    random.seed(42)
    
    generated = []
    while len(generated) < num_samples:
        template = random.choice(templates)
        try:
            text = template.format(
                disease=random.choice(diseases),
                description=random.choice(descriptions),
                symptom=random.choice(symptoms),
                cause=random.choice(causes),
                risk_factor=random.choice(risk_factors),
                diagnosis=random.choice(diagnoses),
                treatment=random.choice(treatments),
                medication=random.choice(medications),
                side_effect=random.choice(side_effects),
                prevention=random.choice(preventions),
                complication=random.choice(complications),
                age=random.choice(ages),
                gender=random.choice(genders),
            )
            if len(text) > 20:
                generated.append({"text": text})
        except KeyError:
            continue
    
    logger.info(f"自动生成 {len(generated)} 条医学校准数据")
    return generated[:num_samples]


def load_or_generate_calibration_data(
    calib_file: Optional[str],
    num_samples: int,
    auto_generate: bool = True
) -> Tuple[List[Dict[str, str]], bool]:
    """加载或生成校准数据，返回格式: [{"text": "..."}, ...]"""
    data_list = []
    is_generated = False
    
    if calib_file and os.path.exists(calib_file):
        logger.info(f"从文件加载校准数据: {calib_file}")
        try:
            with open(calib_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num_samples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        if 'text' not in data:
                            text = (data.get('instruction') or 
                                    data.get('input') or 
                                    data.get('content') or 
                                    data.get('prompt') or '')
                            if text:
                                data = {"text": text}
                            else:
                                continue
                        data_list.append(data)
                    except json.JSONDecodeError:
                        data_list.append({"text": line})
            
            logger.info(f"成功加载 {len(data_list)} 条校准数据")
            
            if len(data_list) < num_samples // 2 and auto_generate:
                logger.warning(f"加载数据不足({len(data_list)}条)，将自动生成补充")
                needed = num_samples - len(data_list)
                data_list.extend(generate_medical_calibration_data(needed))
                is_generated = True
                
        except Exception as e:
            logger.error(f"加载校准文件失败: {e}")
            data_list = []
    
    if not data_list and auto_generate:
        logger.info("自动生成校准数据...")
        data_list = generate_medical_calibration_data(num_samples)
        is_generated = True
        
        default_calib_path = os.path.join(find_project_root(__file__), "calibration.jsonl")
        if not os.path.exists(default_calib_path):
            try:
                with open(default_calib_path, 'w', encoding='utf-8') as f:
                    for item in data_list:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                logger.info(f"已保存自动生成的校准数据到: {default_calib_path}")
            except Exception as e:
                logger.warning(f"保存校准数据失败: {e}")
    
    if not data_list:
        raise RuntimeError("无法获取校准数据")
    
    return data_list[:num_samples], is_generated


def create_recipe(
    quant_scheme: str,
    bits: int,
    group_size: int,
    supports_gptq_modifier: bool,
    supports_awq_modifier: bool,
    supports_string_recipe: bool,
    sequential_targets: str = "Linear"  # T4关键优化：使用"Linear"而不是默认的"TransformerBlock"
) -> Union[Any, str]:
    """
    创建量化配置（recipe），兼容不同版本的 llmcompressor
    关键优化：使用 sequential_targets="Linear" 避免OOM
    """
    scheme_str = f"W{bits}A16"
    
    if quant_scheme.upper() == "GPTQ" and supports_gptq_modifier:
        try:
            from llmcompressor.modifiers.quantization import GPTQModifier
            logger.info(f"使用 GPTQModifier 创建 {scheme_str} 配置")
            logger.info(f"关键优化：sequential_targets='{sequential_targets}'（逐层量化避免OOM）")
            return GPTQModifier(
                scheme=scheme_str,
                targets="Linear",
                ignore=["lm_head"],
                sequential_targets=sequential_targets,  # 关键优化
            )
        except Exception as e:
            logger.warning(f"GPTQModifier 创建失败: {e}")
    
    if quant_scheme.upper() == "AWQ" and supports_awq_modifier:
        try:
            from llmcompressor.modifiers.quantization import AWQModifier
            logger.info(f"使用 AWQModifier 创建 {scheme_str} 配置")
            return AWQModifier(
                scheme=scheme_str,
                targets="Linear",
                ignore=["lm_head"],
                sequential_targets=sequential_targets,
            )
        except Exception as e:
            logger.warning(f"AWQModifier 创建失败: {e}")
    
    if supports_string_recipe:
        logger.info(f"使用字符串 recipe 创建 {quant_scheme} {scheme_str} 配置")
        logger.info(f"关键优化：sequential_targets='{sequential_targets}'")
        
        if quant_scheme.upper() == "GPTQ":
            return f"""
quant_stage:
  quant_modifiers:
    GPTQModifier:
      scheme: "{scheme_str}"
      targets: ["Linear"]
      ignore: ["lm_head"]
      sequential_targets: "{sequential_targets}"
"""
        elif quant_scheme.upper() == "AWQ":
            return f"""
quant_stage:
  quant_modifiers:
    AWQModifier:
      scheme: "{scheme_str}"
      targets: ["Linear"]
      ignore: ["lm_head"]
      sequential_targets: "{sequential_targets}"
"""
    
    logger.info(f"使用 YAML 格式创建 {quant_scheme} 配置")
    return f"""
modifiers:
  - !{quant_scheme}Modifier
      scheme: {scheme_str}
      targets: [Linear]
      ignore: [lm_head]
      sequential_targets: {sequential_targets}
"""


def load_model_with_cpu_offload(model_path: str, gpu_memory_fraction: float = 0.70):  # T4优化：降低到0.70
    """
    关键优化：使用 FP16 + device_map="auto" + 自定义 max_memory 实现CPU卸载
    禁止使用 BitsAndBytes（会导致量化时解压爆显存）
    T4优化：降低 gpu_memory_fraction 到 0.70，给 Hessian 计算留更多空间
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    import accelerate
    
    logger.info("使用 FP16 + CPU卸载加载模型（禁用BNB避免解压爆显存）...")
    logger.info(f"T4优化：GPU内存比例设置为 {gpu_memory_fraction}，为Hessian计算预留空间")
    
    # 计算GPU可用内存（保留更多缓冲给Hessian计算）
    if torch.cuda.is_available():
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_bytes = int(total_gpu_memory * gpu_memory_fraction)
        gpu_memory_gb = gpu_memory_bytes / (1024**3)
        logger.info(f"GPU总内存: {total_gpu_memory / (1024**3):.2f} GB")
        logger.info(f"分配给模型: {gpu_memory_gb:.2f} GB")
        logger.info(f"保留显存缓冲: {(1-gpu_memory_fraction)*100:.0f}%（用于Hessian计算）")
    else:
        gpu_memory_bytes = 0
        logger.warning("未检测到CUDA，将使用CPU")
    
    # 关键：max_memory 字典控制层分布
    max_memory = {
        0: f"{int(gpu_memory_bytes / (1024**3))}GiB",
        "cpu": "100GiB"  # CPU内存充足
    }
    
    logger.info(f"内存分配策略: GPU={max_memory[0]}, CPU={max_memory['cpu']}")
    
    # 先加载到meta设备，再dispatch
    logger.info("正在加载模型配置...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    logger.info("正在加载模型权重（此过程可能需要几分钟）...")
    
    # 使用accelerate的load_checkpoint_and_dispatch实现真正的逐层加载
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
    
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # 推断最优设备映射
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        dtype=torch.float16,
        no_split_module_classes=["Qwen2DecoderLayer", "LlamaDecoderLayer", "MistralDecoderLayer"]
    )
    
    # 统计层分布
    gpu_layers = sum(1 for d in device_map.values() if isinstance(d, int) or str(d).startswith('cuda'))
    cpu_layers = sum(1 for d in device_map.values() if str(d) == 'cpu')
    disk_layers = sum(1 for d in device_map.values() if str(d) == 'disk')
    logger.info(f"层分布规划: GPU={gpu_layers}, CPU={cpu_layers}, Disk={disk_layers}")
    
    # 实际加载并分配
    model = load_checkpoint_and_dispatch(
        model,
        model_path,
        device_map=device_map,
        offload_folder=None,
        dtype=torch.float16,
        offload_state_dict=True,
        no_split_module_classes=["Qwen2DecoderLayer", "LlamaDecoderLayer", "MistralDecoderLayer"]
    )
    
    # 打印实际分布
    if hasattr(model, 'hf_device_map'):
        actual_gpu = sum(1 for d in model.hf_device_map.values() if str(d).startswith('cuda'))
        actual_cpu = sum(1 for d in model.hf_device_map.values() if str(d) == 'cpu')
        logger.info(f"实际层分布: GPU={actual_gpu}, CPU={actual_cpu}")
    
    # 强制垃圾回收
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        logger.info(f"加载后显存: 已分配={allocated:.2f}GB, 预留={reserved:.2f}GB")
    
    return model


def quantize_model(
    model_path: str,
    output_dir: str,
    calib_data: List[Dict[str, str]],
    quant_scheme: str,
    bits: int,
    group_size: int,
    max_seq_length: int,
    device: str,
    supports: Dict[str, bool],
    gpu_memory_fraction: float = 0.70,  # T4优化：默认0.70
    sequential_targets: str = "Linear",  # T4关键优化
) -> bool:
    """
    执行模型量化 - T4 16GB 深度优化版本
    
    核心策略：
    1. 不使用BNB加载（避免量化时解压爆显存）
    2. 使用 FP16 + CPU卸载加载，自动将部分层放CPU
    3. 使用 sequential_targets="Linear" 逐层量化避免OOM
    4. 降低 max_seq_length 到 64 或更低
    5. 激进的显存清理
    """
    try:
        from llmcompressor import oneshot
        from transformers import AutoTokenizer
        from datasets import Dataset
    except ImportError as e:
        logger.error(f"导入失败: {e}")
        return False
    
    if not os.path.exists(model_path):
        logger.error(f"模型路径不存在: {model_path}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    quant_scheme = quant_scheme.upper()
    if quant_scheme == "AWQ":
        logger.warning("AWQ在T4上容易OOM，建议改用GPTQ")
    
    logger.info(f"开始 {quant_scheme} {bits}bit 量化...")
    logger.info(f"  输入模型: {model_path}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  分组大小: {group_size}")
    logger.info(f"  T4关键优化: sequential_targets='{sequential_targets}'")
    
    try:
        # 加载分词器
        logger.info("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 关键：使用CPU卸载加载模型，不使用BNB
        model = load_model_with_cpu_offload(model_path, gpu_memory_fraction)
        
        logger.info("模型加载完成")
        
        # 准备校准数据 - 使用 datasets.Dataset
        logger.info(f"准备 {len(calib_data)} 条校准数据...")
        
        # 创建 HuggingFace Dataset
        calib_dataset = Dataset.from_list(calib_data)
        logger.info(f"成功创建 Dataset，包含 {len(calib_dataset)} 条数据")
        
        # 激进清理显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"Dataset创建后显存: {allocated:.2f} GB")
        
        # 创建量化配置 - 关键优化：使用 sequential_targets="Linear"
        recipe = create_recipe(
            quant_scheme=quant_scheme,
            bits=bits,
            group_size=group_size,
            supports_gptq_modifier=supports["GPTQ"],
            supports_awq_modifier=supports["AWQ"],
            supports_string_recipe=supports["string_recipe"],
            sequential_targets=sequential_targets,
        )
        
        # 设置环境变量优化显存分配
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
        
        # 执行量化
        logger.info(f"开始量化（T4 约需 30-60 分钟，请耐心等待）...")
        logger.info("注意：量化过程中会频繁进行CPU/GPU数据传输，速度较慢但能保证不OOM")
        logger.info(f"关键优化：使用 sequential_targets='{sequential_targets}' 逐层量化")
        
        # 关键修复：使用正确的参数名
        oneshot_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "recipe": recipe,
            "output_dir": output_dir,
            "max_seq_length": max_seq_length,
            "num_calibration_samples": len(calib_data),
            "dataset": calib_dataset,
        }
        
        # 在量化前再次清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 执行量化
        oneshot(**oneshot_kwargs)
        
        # 保存量化信息
        quant_info = {
            "quantization_method": quant_scheme,
            "bits": bits,
            "group_size": group_size,
            "scheme": f"W{bits}A16",
            "original_model_path": model_path,
            "output_path": output_dir,
            "calibration_samples": len(calib_data),
            "max_seq_length": max_seq_length,
            "device": device,
            "torch_dtype": "float16",
            "loading_strategy": "fp16_with_cpu_offload",
            "gpu_memory_fraction": gpu_memory_fraction,
            "sequential_targets": sequential_targets,
        }
        
        info_path = os.path.join(output_dir, "quantization_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(quant_info, f, indent=2, ensure_ascii=False)
        
        compare_model_sizes(model_path, output_dir)
        
        logger.info(f"{quant_scheme} 量化完成！")
        logger.info(f"使用 vLLM 加载命令:")
        logger.info(f'  llm = LLM("{output_dir}", quantization="{quant_scheme.lower()}", dtype="float16")')
        
        return True
        
    except Exception as e:
        logger.error(f"量化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_model_sizes(original_path: str, quantized_path: str) -> Dict[str, float]:
    """比较模型大小"""
    def get_dir_size(path: str) -> float:
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total / (1024**3)
    
    try:
        orig_size = get_dir_size(original_path)
        quant_size = get_dir_size(quantized_path)
        
        result = {
            "original_size_gb": round(orig_size, 2),
            "quantized_size_gb": round(quant_size, 2),
            "compression_ratio": round(orig_size / quant_size, 2) if quant_size > 0 else 0,
            "size_reduction_percent": round((1 - quant_size/orig_size) * 100, 1) if orig_size > 0 else 0
        }
        
        logger.info("模型大小对比:")
        logger.info(f"  原始模型: {result['original_size_gb']} GB")
        logger.info(f"  量化模型: {result['quantized_size_gb']} GB")
        logger.info(f"  压缩比:   {result['compression_ratio']}x")
        logger.info(f"  体积减少: {result['size_reduction_percent']}%")
        
        return result
    except Exception as e:
        logger.error(f"计算模型大小失败: {e}")
        return {}


def find_latest_merged_model(project_root: str) -> Optional[str]:
    """自动查找最新的合并模型"""
    output_dir = os.path.join(project_root, "output")
    if not os.path.exists(output_dir):
        return None
    
    candidates = []
    for d in os.listdir(output_dir):
        if d.startswith(("sft-", "experiments")):
            merged_path = os.path.join(output_dir, d, "merged_model")
            if os.path.exists(merged_path):
                has_weights = any(
                    f.startswith(("model-", "pytorch_model", "model.safetensors"))
                    for f in os.listdir(merged_path)
                )
                if has_weights:
                    mtime = os.path.getmtime(merged_path)
                    candidates.append((mtime, merged_path))
    
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="使用 llm-compressor 进行 GPTQ/AWQ 量化（T4 16GB 深度优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
T4 16GB 关键优化说明：
  1. 禁止使用 --load_bits（BNB加载会导致量化时解压爆显存）
  2. 使用 FP16 + CPU卸载加载，自动将部分层放CPU
  3. 使用 sequential_targets="Linear" 逐层量化（关键优化，避免OOM）
  4. 降低 --calib_samples 到 16（默认）
  5. 降低 --max_seq_len 到 64（默认）
  6. 使用 --gpu_fraction 0.70 为Hessian计算预留显存（默认）

正确示例（T4 16GB）:
  # 推荐配置（GPTQ 4bit + 16样本 + 64长度 + Linear逐层）
  python src/merger/convert_quant.py \
    --model_path output/sft-qwen3-14b/merged_model \
    --output_dir output/sft-qwen3-14b/gptq4_model \
    --quant_scheme GPTQ \
    --bits 4 \
    --calib_samples 16 \
    --max_seq_len 64 \
    --gpu_fraction 0.70 \
    --sequential_targets Linear

  # 如果仍OOM，进一步降低参数
  python src/merger/convert_quant.py \
    --model_path output/sft-qwen3-14b/merged_model \
    --output_dir output/sft-qwen3-14b/gptq4_model \
    --quant_scheme GPTQ \
    --bits 4 \
    --calib_samples 8 \
    --max_seq_len 32 \
    --gpu_fraction 0.65 \
    --sequential_targets Linear

  # 错误示范（不要使用）：
  # --load_bits 4  # 这会导致OOM！
  # --sequential_targets TransformerBlock  # 这会导致OOM！
        """
    )
    
    parser.add_argument("--model_path", type=str, default=None,
                        help="合并后的 HF 模型路径（默认自动查找最新）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="量化模型输出目录（默认基于模型名和量化方案生成）")
    parser.add_argument("--quant_scheme", type=str, default="GPTQ",
                        choices=["GPTQ", "AWQ"],
                        help="量化方案（默认 GPTQ，T4上比AWQ更稳定）")
    parser.add_argument("--bits", type=int, default=4,
                        choices=[4, 8],
                        help="目标量化位数（默认 4）")
    parser.add_argument("--group_size", type=int, default=128,
                        help="分组大小（默认 128）")
    parser.add_argument("--calib_file", type=str, default=None,
                        help="校准数据文件路径（JSONL格式，默认自动生成）")
    parser.add_argument("--calib_samples", type=int, default=16,  # T4优化：默认16
                        help="校准样本数量（默认 16，T4建议8-16）")
    parser.add_argument("--max_seq_len", type=int, default=64,  # T4优化：默认64
                        help="最大序列长度（默认 64，T4建议32-64）")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="计算设备（默认 cuda:0）")
    parser.add_argument("--no_auto_generate", action="store_true",
                        help="禁止自动生成校准数据（若指定则必须提供 calib_file）")
    parser.add_argument("--gpu_fraction", type=float, default=0.70,  # T4优化：默认0.70
                        help="GPU内存使用比例（默认0.70，为Hessian计算预留空间）")
    parser.add_argument("--sequential_targets", type=str, default="Linear",  # T4关键优化
                        choices=["Linear", "TransformerBlock"],
                        help="序列化目标（默认 Linear，T4必须用Linear避免OOM）")
    
    args = parser.parse_args()
    
    # 检查 llmcompressor
    available, supports = check_llmcompressor()
    if not available:
        logger.error("llm-compressor 不可用，请安装: pip install llmcompressor")
        sys.exit(1)
    
    # 解析路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(script_dir)
    logger.info(f"项目根目录: {project_root}")
    
    # 自动查找模型
    if args.model_path is None:
        args.model_path = find_latest_merged_model(project_root)
        if args.model_path:
            logger.info(f"自动找到合并模型: {args.model_path}")
        else:
            logger.error("未找到合并模型，请手动指定 --model_path")
            sys.exit(1)
    
    # 生成输出路径
    if args.output_dir is None:
        model_name = os.path.basename(os.path.normpath(os.path.dirname(args.model_path)))
        scheme_lower = args.quant_scheme.lower()
        args.output_dir = os.path.join(
            project_root, "output", 
            f"{model_name}-{scheme_lower}{args.bits}"
        )
    
    # 转换为绝对路径
    args.model_path = os.path.abspath(args.model_path)
    args.output_dir = os.path.abspath(args.output_dir)
    if args.calib_file:
        args.calib_file = os.path.abspath(args.calib_file)
    
    # 验证输入
    if not os.path.exists(args.model_path):
        logger.error(f"模型路径不存在: {args.model_path}")
        sys.exit(1)
    
    # 检查模型文件
    model_files = os.listdir(args.model_path)
    has_weights = any(
        f.startswith(("model-", "pytorch_model", "model.safetensors"))
        for f in model_files
    )
    if not has_weights:
        logger.error(f"模型目录缺少权重文件: {args.model_path}")
        sys.exit(1)
    
    logger.info(f"输入模型: {args.model_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"量化方案: {args.quant_scheme} {args.bits}bit")
    logger.info(f"校准样本: {args.calib_samples}")
    logger.info(f"最大序列长度: {args.max_seq_len}")
    logger.info(f"GPU内存比例: {args.gpu_fraction}")
    logger.info(f"序列化目标: {args.sequential_targets}（关键优化）")
    logger.info(f"校准文件: {args.calib_file or '自动生成'}")
    logger.info("关键：使用FP16+CPU卸载加载（禁用BNB避免解压爆显存）")
    
    # T4 关键警告
    if args.sequential_targets == "TransformerBlock":
        logger.warning("=" * 80)
        logger.warning("警告：sequential_targets='TransformerBlock' 在T4上可能导致OOM！")
        logger.warning("强烈建议使用 --sequential_targets Linear")
        logger.warning("=" * 80)
    
    # 加载/生成校准数据
    try:
        calib_data, is_generated = load_or_generate_calibration_data(
            args.calib_file,
            args.calib_samples,
            auto_generate=not args.no_auto_generate
        )
        if is_generated:
            logger.info("使用自动生成的校准数据")
    except RuntimeError as e:
        logger.error(f"获取校准数据失败: {e}")
        sys.exit(1)
    
    # 执行量化
    success = quantize_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        calib_data=calib_data,
        quant_scheme=args.quant_scheme,
        bits=args.bits,
        group_size=args.group_size,
        max_seq_length=args.max_seq_len,
        device=args.device,
        supports=supports,
        gpu_memory_fraction=args.gpu_fraction,
        sequential_targets=args.sequential_targets,
    )
    
    if success:
        logger.info("量化成功完成！")
        sys.exit(0)
    else:
        logger.error("量化失败！")
        logger.error("如果OOM，请尝试：")
        logger.error("  1. 降低 --calib_samples 到 8")
        logger.error("  2. 降低 --max_seq_len 到 32")
        logger.error("  3. 降低 --gpu_fraction 到 0.65")
        logger.error("  4. 确保使用 --sequential_targets Linear")
        logger.error("  5. 确保没有使用 --load_bits")
        sys.exit(1)


if __name__ == "__main__":
    main()