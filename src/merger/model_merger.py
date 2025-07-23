#!/usr/bin/env python3
# src/merger/model_merger.py
import os
import shutil
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_models(args):
    base_model_path = args.model
    adapter_path = args.adapter
    output_dir = args.output
    max_shard_size = args.max_shard_size
    dtype_str = args.dtype

    os.makedirs(output_dir, exist_ok=True)

    # 自动选择 dtype
    if dtype_str == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = getattr(torch, dtype_str)

    logger.info(f"使用数据类型: {dtype}")

    # ✅ 强制加载 tokenizer（始终使用原始模型）
    logger.info("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # ✅ 强制加载基础模型到 CPU，避免显存不足
    logger.info("加载基础模型到 CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=None,  # 强制 CPU
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # ✅ 加载并合并 LoRA（也在 CPU 上）
    logger.info("加载并合并 LoRA 到 CPU...")
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        torch_dtype=dtype,
        device_map=None,  # 强制 CPU
    )
    model = model.merge_and_unload()

    # ✅ 保存合并后的模型（safetensors，分片）
    logger.info("保存合并后的模型到磁盘...")
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size
    )

    # ✅ 复制原始模型的辅助文件
    copy_auxiliary_files(base_model_path, output_dir)

    logger.info(f"✅ 合并完成，模型已保存至: {output_dir}")

def copy_auxiliary_files(src_dir: str, dst_dir: str):
    """复制原始模型的辅助文件（非权重）"""
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        # 跳过权重文件、目录、日志等
        if any(filename.startswith(prefix) for prefix in ["model-", "pytorch_model", "adapter"]):
            continue
        if os.path.isdir(src_path):
            continue

        shutil.copy2(src_path, dst_path)
        logger.info(f"复制辅助文件: {filename}")