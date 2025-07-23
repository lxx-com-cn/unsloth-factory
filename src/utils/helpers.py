# src/utils/helpers.py
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
    
    # 获取领域知识库
    knowledge_base, term_map, get_advice = get_knowledge_base(domain)
    
    try:
        # 关键错误检查
        for condition, terms in knowledge_base.items():
            if condition in question:
                for term in terms:
                    if term not in response:
                        logger.warning(f"可能遗漏领域术语: {term}")
        
        # 领域特定错误修正
        if domain == "medical":
            error_corrections = [
                (r"闭经.*子宫内膜癌", "需排除妊娠后再评估", "闭经直接诊断为子宫内膜癌"),
                (r"麻疹.*(扁桃体炎|痄腮)", "麻疹", "麻疹误诊为扁桃体炎或痄腮"),
                (r"胃肠穿孔.*胰腺炎", "胃肠穿孔", "胃肠穿孔误诊为胰腺炎"),
                (r"霍奇金病.*(白血病|淋巴瘤)", "霍奇金淋巴瘤", "霍奇金病与其他淋巴瘤混淆"),
                (r"脑卒中.*自行用药", "立即就医，不要自行用药", "脑卒中自行用药风险"),
                (r"心肌梗死.*阿司匹林", "立即就医，不要自行用药", "心梗自行用药风险"),
                (r"中毒.*催吐", "不要自行催吐", "中毒自行催吐风险"),
            ]
        elif domain == "legal":
            error_corrections = [
                (r"口头协议.*证据效力", "书面合同具有更高证据效力", "忽视书面证据风险"),
                (r"知识产权.*自行维权", "建议委托专业律师", "自行维权风险"),
                (r"刑事.*自首", "自首可以从轻或减轻处罚", "未提及自首法律效果"),
            ]
        elif domain == "psychology":
            error_corrections = [
                (r"抑郁.*自行停药", "不要自行停药，遵医嘱", "自行停药风险"),
                (r"自杀意念.*保密", "有伤害自己或他人风险时应突破保密原则", "保密原则误解"),
            ]
        elif domain == "exam":
            error_corrections = [
                (r"数学题.*近似值", "使用精确计算", "考试中不应使用近似值"),
                (r"作文.*抄袭", "原创内容得分更高", "抄袭风险"),
            ]
        else:
            error_corrections = []
        
        # 应用修正
        for pattern, replacement, warning in error_corrections:
            if re.search(pattern, question) and re.search(pattern, response):
                logger.warning(f"检测到可能的领域错误: {warning}")
                response = re.sub(pattern, replacement, response)
    except Exception as e:
        logger.error(f"领域验证失败: {str(e)}")
    
    return response

def ensure_chinese_output(text):
    """确保输出为纯中文"""
    # 替换英文术语
    for eng, chn in MEDICAL_TERM_MAP.items():
        text = text.replace(eng, chn)
    
    # 移除残留英文
    text = re.sub(r'[a-zA-Z]{4,}', '', text)
    return text

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