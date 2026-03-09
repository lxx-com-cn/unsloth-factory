# src/utils/helpers.py - 优化资源监控函数

import os
import re
import logging
import psutil
import GPUtil
import json
import shutil
import tempfile
import torch
import inspect
from src.knowledge import get_knowledge_base  # 导入知识库接口

# 设置模块级 logger
logger = logging.getLogger(__name__)

def get_gpu_memory_usage():
    """获取GPU显存使用情况"""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            available = total - allocated
            
            return {
                'allocated_gb': round(allocated, 2),
                'reserved_gb': round(reserved, 2),
                'total_gb': round(total, 2),
                'available_gb': round(available, 2),
                'utilization_percent': round((allocated / total) * 100, 1)
            }
    except Exception as e:
        logger.error(f"获取GPU显存失败: {e}")
    return {'allocated_gb': 0, 'reserved_gb': 0, 'total_gb': 0, 'available_gb': 0, 'utilization_percent': 0}

def check_system_resources():
    """检查系统资源"""
    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # 内存使用
    memory = psutil.virtual_memory()
    memory_used_gb = memory.used / (1024**3)
    memory_total_gb = memory.total / (1024**3)
    
    # GPU使用
    gpu_info = get_gpu_memory_usage()
    
    return {
        'cpu_percent': cpu_percent,
        'memory_used_gb': round(memory_used_gb, 2),
        'memory_total_gb': round(memory_total_gb, 2),
        'memory_percent': memory.percent,
        'gpu_memory': gpu_info
    }

def can_handle_concurrent_request(current_concurrent=0):
    """检查是否可以处理并发请求 - 修复并发限制"""
    resources = check_system_resources()
    gpu_available = resources['gpu_memory']['available_gb']
    gpu_utilization = resources['gpu_memory']['utilization_percent']
    
    # 更宽松的并发条件：只要有2GB可用显存且利用率低于85%就可以处理并发
    can_handle = gpu_available >= 2.0 and gpu_utilization < 85
    
    logger.info(f"并发检查: 当前并发={current_concurrent}, 可用显存={gpu_available}GB, GPU利用率={gpu_utilization}%, 可处理={can_handle}")
    
    return can_handle

def extract_answer_letter(text):
    """从文本中提取选择题答案字母（增强版）"""
    # 方法1: 直接查找大写字母
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1)
    
    # 方法2: 查找"答案："后的字母
    match = re.search(r'[答答案案]：?\s*([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # 方法3: 查找括号内的字母
    match = re.search(r'[\(（]([A-D])[\)）]', text)
    if match:
        return match.group(1)
    
    # 方法4: 查找选项文本开头的字母
    match = re.search(r'^\s*([A-D])[\.、]', text)
    if match:
        return match.group(1)
    
    # 方法5: 查找类似"正确答案是A"的模式
    match = re.search(r'(正确答案?|正确选项?|正确选择?)\s*[:：]?\s*([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(2)
    
    # 最后尝试：返回文本中第一个大写字母
    for char in text:
        if char in "ABCD":
            return char
    
    return ""  # 未找到答案

def is_chinese(text):
    """检查文本是否主要为中文"""
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(chinese_chars) / max(1, len(text)) > 0.5

def clean_response(response):
    """清理响应中的特殊标记和无关内容"""
    # 移除<think>标签
    if response.startswith("<think>"):
        end_pos = response.find("</think>")
        if end_pos != -1:
            response = response[end_pos + 8:].strip()
    
    # 移除多余的选项分析
    if "(A)" in response and "(B)" in response:
        # 找到实际回答开始位置
        answer_start = max(
            response.find("答案是"),
            response.find("正确选项是"),
            response.find("The answer is"),
            response.find("Therefore")
        )
        if answer_start != -1:
            response = response[answer_start:]
    
    return response

def log_sample_debug(sample, idx, logger):
    """记录样本调试信息"""
    logger.debug(f"样本 {idx} 调试信息:")
    logger.debug(f"问题: {sample.get('question', 'N/A')}")
    logger.debug(f"选项: A.{sample.get('choices', [''])[0]} | B.{sample.get('choices', [''])[1]} | C.{sample.get('choices', [''])[2]} | D.{sample.get('choices', [''])[3]}")
    logger.debug(f"参考答案: {sample.get('answer', 'N/A')}")
    logger.debug(f"模型输出: {sample.get('output', 'N/A')}")
    logger.debug(f"预测答案: {sample.get('prediction', 'N/A')}")
    logger.debug("=" * 50)

def fix_unsloth_chat_template(template_str):
    """修复聊天模板以满足 Unsloth 要求"""
    # 确保聊天模板包含必要的生成提示
    if "{% if add_generation_prompt %}" not in template_str:
        logger.warning("模板缺少 {% if add_generation_prompt %}，自动修复")
        # 尝试在适当位置添加
        if template_str.strip().endswith("{% endif %}"):
            # 在最后endif之前插入
            template_str = template_str.replace(
                "{% endif %}",
                "{% if add_generation_prompt %}{{ '' }}{% endif %}{% endif %}"
            )
        else:
            # 直接追加
            template_str += "{% if add_generation_prompt %}{{ '' }}{% endif %}"
    
    # 添加 Unsloth 要求的特定标记
    required_string = "{{ '' }}"
    if required_string not in template_str:
        logger.warning("模板缺少 {{ '' }}，自动修复")
        # 添加在适当位置
        if "{% if add_generation_prompt %}" in template_str:
            template_str = template_str.replace(
                "{% if add_generation_prompt %}",
                f"{{% if add_generation_prompt %}}{required_string}"
            )
    
    # 确保模板有正确的结束标记
    if "{% endif %}" not in template_str:
        logger.warning("模板缺少 {% endif %}，自动修复")
        template_str += "{% endif %}"
    
    return template_str

def ensure_template_compatibility(template_path):
    """确保模板兼容 Unsloth"""
    try:
        # 确保路径存在
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"模板文件不存在: {template_path}")
        
        with open(template_path, "r", encoding="utf-8") as f:
            template = json.load(f)
        
        if "chat_template" in template and template["chat_template"]:
            logger.info("修复聊天模板兼容性")
            template["chat_template"] = fix_unsloth_chat_template(template["chat_template"])
            
            # 保存修复后的模板
            with open(template_path, "w", encoding="utf-8") as f:
                json.dump(template, f, ensure_ascii=False, indent=2)
            logger.info(f"模板已修复并保存: {template_path}")
        
        return template
    except Exception as e:
        logger.error(f"修复模板失败: {str(e)}")
        return None

def ensure_tokenizer_compatibility(model_path):
    """确保 tokenizer 配置兼容 Unsloth"""
    try:
        # 检查 tokenizer 配置文件是否存在
        tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
        if not os.path.exists(tokenizer_config_path):
            logger.warning(f"tokenizer_config.json 不存在: {model_path}")
            return model_path
        
        # 读取 tokenizer 配置
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
        
        # 检查是否需要修复
        if "chat_template" not in tokenizer_config:
            logger.info("tokenizer_config.json 中没有 chat_template 字段")
            return model_path
        
        original_template = tokenizer_config["chat_template"]
        if "add_generation_prompt" in original_template:
            logger.info("tokenizer 配置已兼容 Unsloth")
            return model_path
        
        # 修复模板
        logger.info("修复 tokenizer 聊天模板")
        fixed_template = fix_unsloth_chat_template(original_template)
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        logger.info(f"创建临时目录: {temp_dir}")
        
        # 复制整个模型目录到临时目录
        for item in os.listdir(model_path):
            src = os.path.join(model_path, item)
            dst = os.path.join(temp_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        
        # 更新 tokenizer 配置
        temp_tokenizer_config_path = os.path.join(temp_dir, "tokenizer_config.json")
        tokenizer_config["chat_template"] = fixed_template
        with open(temp_tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已修复 tokenizer 配置并保存到临时目录: {temp_dir}")
        return temp_dir
        
    except Exception as e:
        logger.error(f"修复 tokenizer 配置失败: {str(e)}")
        return model_path

def log_memory_usage():
    """记录内存和显存使用情况"""
    # 获取CPU内存使用情况
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_mem = mem_info.rss / (1024 ** 3)  # GB
    
    # 获取GPU显存使用情况
    gpu_mem = 0
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_mem = gpus[0].memoryUsed
    except Exception as e:
        logger.error(f"获取GPU信息失败: {str(e)}")
    
    return f"内存: {cpu_mem:.2f} GB, 显存: {gpu_mem:.2f} MB"

def calculate_dataset_stats(dataset):
    """计算数据集统计信息"""
    stats = {
        "total_samples": len(dataset),
        "min_length": float('inf'),
        "max_length": 0,
        "total_length": 0,
        "empty_samples": 0
    }

    try:
        for item in dataset:
            text = item.get("text", "")
            if not text.strip():
                stats["empty_samples"] += 1
                continue

            length = len(text)
            stats["min_length"] = min(stats["min_length"], length)
            stats["max_length"] = max(stats["max_length"], length)
            stats["total_length"] += length

        if stats["total_samples"] - stats["empty_samples"] > 0:
            stats["avg_length"] = stats["total_length"] / (stats["total_samples"] - stats["empty_samples"])
        else:
            stats["avg_length"] = 0
            
        logger.info(f"数据集统计: 总样本={stats['total_samples']}, 空样本={stats['empty_samples']}")
        logger.info(f"文本长度: 最小={stats['min_length']}, 最大={stats['max_length']}, 平均={stats['avg_length']:.2f}")
    except Exception as e:
        logger.error(f"计算数据集统计信息时出错: {str(e)}")
        stats["avg_length"] = 0

    return stats

def setup_logging(level=logging.INFO):
    """设置日志格式和级别"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level
    )
    logger.info("日志系统已初始化")

def deep_clean_response(response):
    """深度清理响应中的特殊标记和无关内容"""
    # 1. 移除所有XML风格标签
    response = re.sub(r'<[^>]+>', '', response)
    
    # 2. 移除结束标记
    response = response.replace('</s>', '').replace('<|endoftext|>', '')
    
    # 3. 移除选项分析标记
    response = re.sub(r'\([A-D]\)', '', response)
    
    # 4. 清理多余空格和换行
    response = re.sub(r'\s+', ' ', response).strip()
    
    # 5. 截断到第一个句号后的完整句子
    if '.' in response:
        response = response[:response.rfind('.')+1]
    
    return response

def validate_diagnosis(response, question, model_template, domain="medical"):
    """验证诊断与症状的匹配度 - 通用领域支持"""
    # 获取领域知识库
    knowledge_base, term_map, get_advice = get_knowledge_base(domain)
    
    # 术语替换
    for eng, chn in term_map.items():
        response = response.replace(eng, chn)
    
    # 特定病症增强
    for condition, info in knowledge_base.items():
        if condition in question:
            # 添加领域特定建议
            response = get_advice(condition, response)
    
    return response

def validate_domain_response(response, question, domain="medical"):
    """增强响应验证 - 通用领域支持"""
    # 确保输入是字符串
    if not response or not isinstance(response, str):
        return response
    
    if not question or not isinstance(question, str):
        return response
    
    try:
        # 获取领域知识库
        knowledge_base, term_map, get_advice = get_knowledge_base(domain)
        
        # 应用术语替换
        for eng, chn in term_map.items():
            response = response.replace(eng, chn)
            question = question.replace(eng, chn)
        
        # 应用领域特定建议
        response = get_advice(question, response)
        
    except Exception as e:
        logger.warning(f"知识库处理失败: {str(e)}，使用基础验证")
        response = _basic_domain_validation(response, question, domain)
    
    # 应用领域特定的错误修正
    try:
        response = _apply_domain_error_corrections(response, question, domain)
    except Exception as e:
        logger.error(f"领域错误修正失败: {str(e)}")
    
    return response


def _basic_domain_validation(response, question, domain):
    """基础领域验证（当知识库不可用时使用）"""
    # 基础的危险信号检测
    danger_signals = {
        "medical": [
            ("自行用药", "请勿自行用药，务必在医生指导下治疗"),
            ("催吐", "中毒情况下不要自行催吐，立即就医"),
        ],
        "legal": [
            ("自行维权", "建议咨询专业律师"),
        ],
        "psychology": [
            ("自行停药", "请不要自行停药，务必遵医嘱"),
        ],
        "exam": [
            ("抄袭", "必须提交原创内容"),
        ]
    }
    
    signals = danger_signals.get(domain, [])
    for signal, warning in signals:
        if signal in response and warning not in response:
            response += f"\n\n【重要提醒】{warning}"
    
    return response
    

def _apply_domain_error_corrections(response, question, domain):
    """应用领域特定的错误修正"""
    error_corrections = {
        "medical": [
            (r"闭经.*子宫内膜癌", "需排除妊娠后再评估", "闭经直接诊断为子宫内膜癌"),
            (r"麻疹.*(扁桃体炎|痄腮)", "麻疹", "麻疹误诊为扁桃体炎或痄腮"),
            (r"胃肠穿孔.*胰腺炎", "胃肠穿孔", "胃肠穿孔误诊为胰腺炎"),
            (r"霍奇金病.*(白血病|淋巴瘤)", "霍奇金淋巴瘤", "霍奇金病与其他淋巴瘤混淆"),
            (r"脑卒中.*自行用药", "立即就医，不要自行用药", "脑卒中自行用药风险"),
            (r"心肌梗死.*阿司匹林", "立即就医，不要自行用药", "心梗自行用药风险"),
            (r"中毒.*催吐", "不要自行催吐", "中毒自行催吐风险"),
        ],
        "legal": [
            (r"口头协议.*证据效力", "书面合同具有更高证据效力", "忽视书面证据风险"),
            (r"知识产权.*自行维权", "建议委托专业律师", "自行维权风险"),
            (r"刑事.*自首", "自首可以从轻或减轻处罚", "未提及自首法律效果"),
        ],
        "psychology": [
            (r"抑郁.*自行停药", "不要自行停药，遵医嘱", "自行停药风险"),
            (r"自杀意念.*保密", "有伤害自己或他人风险时应突破保密原则", "保密原则误解"),
        ],
        "exam": [
            (r"数学题.*近似值", "使用精确计算", "考试中不应使用近似值"),
            (r"作文.*抄袭", "原创内容得分更高", "抄袭风险"),
        ]
    }
    
    corrections = error_corrections.get(domain, [])
    for pattern, replacement, warning in corrections:
        if re.search(pattern, question) and re.search(pattern, response):
            logger.warning(f"检测到可能的领域错误: {warning}")
            response = re.sub(pattern, replacement, response)
    
    return response
    
def ensure_chinese_output(text):
    """确保输出为纯中文"""
    # 替换英文术语
    for eng, chn in MEDICAL_TERM_MAP.items():
        text = text.replace(eng, chn)
    
    # 移除残留英文
    text = re.sub(r'[a-zA-Z]{4,}', '', text)
    return text

def copy_model_config_files(src_dir: str, dst_dir: str, override=True):
    """
    复制模型核心配置文件，确保与原始模型一致
    
    参数:
        src_dir: 原始模型目录路径
        dst_dir: 目标目录路径
        override: 是否覆盖目标目录中的已有文件
    """
    # 关键配置文件清单（按优先级排序）
    config_files = [
        'tokenizer_config.json',  # Tokenizer 核心配置
        'special_tokens_map.json',  # 特殊token映射
        'generation_config.json',  # 生成参数配置
        'config.json',  # 模型结构定义
        'model.safetensors.index.json',  # 分片模型索引
        'tokenizer.json'  # Tokenizer 完整配置
    ]
    
    # 可选补充文件（存在则复制）
    supplementary_files = [
        'vocab.json',
        'merges.txt',
        'added_tokens.json',
        'preprocessor_config.json'
    ]
    
    copied_files = []
    skipped_files = []
    
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)
    
    # 复制核心配置文件
    for filename in config_files:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        
        # 检查源文件是否存在
        if not os.path.exists(src_path):
            skipped_files.append(filename)
            continue
            
        # 检查是否覆盖
        if os.path.exists(dst_path) and not override:
            skipped_files.append(filename)
            continue
            
        shutil.copy2(src_path, dst_path)
        copied_files.append(filename)
    
    # 复制补充文件（可选）
    for filename in supplementary_files:
        src_path = os.path.join(src_dir, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(dst_dir, filename))
            copied_files.append(filename)
    
    # 记录结果
    if copied_files:
        logger.info(f"已复制 {len(copied_files)} 个配置文件到 {dst_dir}")
    if skipped_files:
        logger.warning(f"跳过 {len(skipped_files)} 个文件: {', '.join(skipped_files)}")
    
    return copied_files


def copy_original_tokenizer_files(original_model_path: str, target_dir: str):
    """将原始模型的 tokenizer 相关配置文件复制到目标目录，确保一致性。"""
    files_to_copy = [
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]
    for filename in files_to_copy:
        src = os.path.join(original_model_path, filename)
        dst = os.path.join(target_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"已复制原始文件: {filename} -> {dst}")
        else:
            logger.warning(f"原始文件不存在，跳过: {src}")

def copy_model_config_files(src_dir: str, dst_dir: str, config_files=None):
    """复制模型的核心配置文件，如 config.json"""
    if config_files is None:
        config_files = ['config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json']
    for filename in config_files:
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dst_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"已复制配置文件: {filename} -> {dst}")
        else:
            logger.warning(f"配置文件不存在，跳过: {src}")
            
            
# 当直接运行此文件时进行测试
if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    logger.info("测试 helpers 模块")
    
    # 测试数据集统计
    test_dataset = [
        {"text": "这是一个测试样本"},
        {"text": "另一个更长的测试样本用于验证功能"},
        {"text": ""},  # 空样本
        {"text": "短文本"}
    ]
    stats = calculate_dataset_stats(test_dataset)
    logger.info(f"测试数据集统计结果: {stats}")
    
    # 测试内存监控
    logger.info(f"当前内存使用: {log_memory_usage()}")
    
    # 测试医学验证
    test_response = "患者闭经，诊断为子宫内膜癌"
    test_question = "闭经的诊断"
    fixed_response = validate_domain_response(test_response, test_question, "medical")
    logger.info(f"医学验证前: {test_response}")
    logger.info(f"医学验证后: {fixed_response}")
    
    logger.info("helpers 模块测试完成")