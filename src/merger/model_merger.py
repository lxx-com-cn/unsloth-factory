#!/usr/bin/env python3
# src/merger/model_merger.py - 完整修复版（解决T4 16GB内存问题）

import os
import shutil
import logging
import json
import torch
import gc
import time
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.utils.helpers import copy_model_config_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_models(args):
    """
    合并 LoRA 适配器到基础模型，并保存完整模型。
    针对T4 16GB深度优化：使用CPU卸载、分块保存、激进内存管理
    """
    base_model_path = args.model
    adapter_path = args.adapter
    output_dir = args.output
    max_shard_size = getattr(args, 'max_shard_size', '2GB')
    dtype_str = getattr(args, 'dtype', 'auto')

    # 清理输出目录
    if os.path.exists(output_dir):
        logger.info(f"清理输出目录中的旧模型文件: {output_dir}")
        for f in os.listdir(output_dir):
            fpath = os.path.join(output_dir, f)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    logger.info(f"  删除旧文件: {f}")
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
                    logger.info(f"  删除旧目录: {f}")
            except Exception as e:
                logger.warning(f"  删除失败 {f}: {e}")
    else:
        os.makedirs(output_dir, exist_ok=True)

    # 确定数据类型 - T4上使用float16更稳定
    if dtype_str == "auto":
        dtype = torch.float16  # T4上强制使用float16，bfloat16可能有问题
    else:
        dtype = getattr(torch, dtype_str)

    logger.info(f"使用数据类型: {dtype}")
    logger.info(f"最大分片大小: {max_shard_size}")

    # 加载 tokenizer
    logger.info("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    logger.info("Tokenizer已保存")

    # 关键优化1：使用low_cpu_mem_usage和device_map=None加载到CPU
    logger.info("加载基础模型到 CPU（使用low_cpu_mem_usage优化）...")
    
    # 先清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 获取系统内存信息
    import psutil
    mem = psutil.virtual_memory()
    logger.info(f"系统内存: 总计{mem.total/1024**3:.1f}GB, 可用{mem.available/1024**3:.1f}GB")
    
    # 如果可用内存不足32GB，给出警告
    if mem.available < 32 * 1024**3:
        logger.warning("系统可用内存不足32GB，合并过程可能因OOM被杀死")
        logger.warning("建议：关闭其他程序，或增加swap空间")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map=None,  # 强制CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # 关键：减少内存峰值
            offload_state_dict=True,  # 关键：将状态字典卸载到磁盘
        )
        logger.info(f"基础模型加载完成，当前内存使用: {get_memory_info()}")
    except Exception as e:
        logger.error(f"基础模型加载失败: {e}")
        raise

    # 加载并合并 LoRA
    logger.info("加载并合并 LoRA 适配器...")
    try:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            torch_dtype=dtype,
            device_map=None,  # 强制CPU
        )
        logger.info("LoRA适配器加载完成")
        
        # 关键优化2：合并前再次清理
        gc.collect()
        
        logger.info("开始合并LoRA到基础模型...")
        model = model.merge_and_unload()
        logger.info(f"合并完成，当前内存使用: {get_memory_info()}")
    except Exception as e:
        logger.error(f"LoRA合并失败: {e}")
        raise

    # 关键优化3：删除旧的索引文件（保险起见）
    old_index_path = os.path.join(output_dir, "model.safetensors.index.json")
    if os.path.exists(old_index_path):
        logger.info(f"删除残留的旧索引文件: {old_index_path}")
        os.remove(old_index_path)

    # 关键优化4：分步骤保存，添加内存监控和强制清理
    logger.info(f"开始保存合并后的模型，分片大小: {max_shard_size}...")
    logger.info("注意：保存大模型需要较长时间，请耐心等待...")
    
    # 使用try-except捕获保存过程中的异常
    try:
        # 先尝试标准保存
        _save_model_with_retry(model, output_dir, max_shard_size)
    except Exception as e:
        logger.error(f"标准保存失败: {e}")
        logger.info("尝试备选保存方案（不使用safetensors）...")
        try:
            # 备选方案：不使用safetensors，使用pytorch原生格式
            model.save_pretrained(
                output_dir,
                safe_serialization=False,  # 使用pytorch格式，更稳定但文件更大
                max_shard_size=max_shard_size
            )
            logger.info("备选保存方案成功")
        except Exception as e2:
            logger.error(f"备选保存也失败: {e2}")
            raise RuntimeError("模型保存完全失败，请检查磁盘空间和内存")

    # 强制垃圾回收
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 复制辅助文件
    copy_auxiliary_files(base_model_path, output_dir)

    # 复制核心配置文件
    copy_model_config_files(
        src_dir=base_model_path,
        dst_dir=output_dir,
        config_files=[
            'tokenizer_config.json',
            'special_tokens_map.json',
            'generation_config.json',
            'config.json'
        ]
    )

    # 验证生成的模型文件
    if not verify_merged_model(output_dir):
        logger.warning("模型验证未通过，但文件可能仍然可用")

    logger.info(f"合并完成，模型已保存至: {output_dir}")
    logger.info(f"最终内存状态: {get_memory_info()}")


def _save_model_with_retry(model, output_dir: str, max_shard_size: str, max_retries: int = 3):
    """
    带重试机制的模型保存，每次重试前强制清理内存
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"保存尝试 {attempt + 1}/{max_retries}...")
            
            # 每次尝试前强制清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 添加延迟，让系统回收内存
            if attempt > 0:
                wait_time = 5 * attempt
                logger.info(f"等待 {wait_time} 秒让系统回收内存...")
                time.sleep(wait_time)
            
            # 尝试保存
            model.save_pretrained(
                output_dir,
                safe_serialization=True,
                max_shard_size=max_shard_size
            )
            logger.info("保存成功")
            return
            
        except Exception as e:
            logger.warning(f"保存尝试 {attempt + 1} 失败: {e}")
            if attempt == max_retries - 1:
                raise
            logger.info("准备重试...")


def get_memory_info() -> str:
    """获取当前内存使用信息"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return f"内存: {mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB ({mem.percent}%)"
    except:
        return "内存信息获取失败"


def copy_auxiliary_files(src_dir: str, dst_dir: str):
    """
    复制模型目录中的辅助文件，排除模型权重和索引文件
    """
    copied = 0
    skipped = 0
    
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        # 跳过模型权重文件
        if filename.startswith('model-') and filename.endswith('.safetensors'):
            skipped += 1
            continue
        if filename.startswith('pytorch_model') and filename.endswith('.bin'):
            skipped += 1
            continue
        # 跳过索引文件
        if filename == 'model.safetensors.index.json':
            skipped += 1
            continue
        # 跳过适配器文件
        if filename.startswith('adapter'):
            skipped += 1
            continue
        # 跳过子目录
        if os.path.isdir(src_path):
            continue
        # 跳过已复制的tokenizer文件
        if filename in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 
                       'vocab.json', 'merges.txt', 'added_tokens.json']:
            continue

        try:
            shutil.copy2(src_path, dst_path)
            copied += 1
            logger.info(f"复制辅助文件: {filename}")
        except Exception as e:
            logger.warning(f"复制 {filename} 失败: {e}")
    
    logger.info(f"辅助文件复制完成: {copied}个已复制, {skipped}个已跳过")


def verify_merged_model(output_dir: str) -> bool:
    """
    验证合并后的模型文件完整性
    """
    logger.info("验证合并后的模型...")
    
    # 检查索引文件
    index_file = os.path.join(output_dir, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        # 检查是否有pytorch格式的索引
        pytorch_index = os.path.join(output_dir, "pytorch_model.bin.index.json")
        if os.path.exists(pytorch_index):
            index_file = pytorch_index
            logger.info("检测到PyTorch格式索引文件")
        else:
            # 检查是否有单个模型文件
            model_files = [f for f in os.listdir(output_dir) 
                          if f.startswith('model-') or f.startswith('pytorch_model')]
            if not model_files:
                logger.error("未找到模型权重文件或索引文件")
                return False
            else:
                logger.info(f"找到模型文件（无索引）: {model_files}")
                return True
    
    # 读取索引
    try:
        with open(index_file, 'r') as f:
            index = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"索引文件 JSON 解析失败: {e}")
        return False
    except Exception as e:
        logger.error(f"读取索引文件失败: {e}")
        return False

    # 获取 weight_map
    weight_map = index.get('weight_map', {})
    if not weight_map:
        logger.error("索引文件缺少 weight_map 字段")
        return False

    # 从 weight_map 中提取分片文件名
    shards_in_index = set()
    for weight_name, shard_file in weight_map.items():
        shards_in_index.add(shard_file)

    # 获取实际存在的分片
    actual_shards = set()
    for f in os.listdir(output_dir):
        if f.startswith('model-') and f.endswith('.safetensors'):
            actual_shards.add(f)
        elif f.startswith('pytorch_model') and f.endswith('.bin'):
            actual_shards.add(f)

    logger.info(f"索引文件记录的分片数: {len(shards_in_index)}")
    logger.info(f"实际存在的分片数: {len(actual_shards)}")

    # 对比
    missing_in_dir = shards_in_index - actual_shards
    extra_in_dir = actual_shards - shards_in_index

    if missing_in_dir:
        logger.error(f"索引中有但实际缺失的分片: {missing_in_dir}")
        return False
    if extra_in_dir:
        logger.warning(f"实际存在但索引中未记录的分片: {extra_in_dir}")

    if not missing_in_dir:
        logger.info("验证通过：索引文件与实际分片一致")
        return True
    else:
        logger.error("验证失败：索引文件与实际分片不匹配！")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--model", type=str, required=True, help="Base model path")
    parser.add_argument("--adapter", type=str, required=True, help="Adapter path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--max_shard_size", type=str, default="2GB", help="Max shard size")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    args = parser.parse_args()
    merge_models(args)