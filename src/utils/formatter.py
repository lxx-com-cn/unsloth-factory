# src/utils/formatter.py
import re
from typing import Dict

class ResponseFormatter:
    """响应格式化工具类"""
    
    @staticmethod
    def format_medical_response(text: str, think_chain: bool = True) -> str:
        """格式化医学相关响应"""
        # 根据think_chain参数决定是否保留think内容
        if not think_chain:
            text = ResponseFormatter._remove_think_tags(text)
        
        # 清理内部标记
        text = ResponseFormatter._clean_internal_tokens(text)
        
        # 应用markdown格式化
        text = ResponseFormatter._apply_markdown_format(text)
        
        # 医学特定格式化
        text = ResponseFormatter._apply_medical_format(text)
        
        # 最终清理
        text = ResponseFormatter._final_cleanup(text)
        
        return text.strip()
    
    @staticmethod
    def format_finance_response(text: str, think_chain: bool = True) -> str:
        """格式化金融相关响应"""
        if not think_chain:
            text = ResponseFormatter._remove_think_tags(text)
        
        text = ResponseFormatter._clean_internal_tokens(text)
        text = ResponseFormatter._apply_markdown_format(text)
        text = ResponseFormatter._apply_finance_format(text)
        text = ResponseFormatter._final_cleanup(text)
        return text.strip()
    
    @staticmethod
    def format_legal_response(text: str, think_chain: bool = True) -> str:
        """格式化法律相关响应"""
        if not think_chain:
            text = ResponseFormatter._remove_think_tags(text)
        
        text = ResponseFormatter._clean_internal_tokens(text)
        text = ResponseFormatter._apply_markdown_format(text)
        text = ResponseFormatter._apply_legal_format(text)
        text = ResponseFormatter._final_cleanup(text)
        return text.strip()
    
    @staticmethod
    def format_education_response(text: str, think_chain: bool = True) -> str:
        """格式化教育相关响应"""
        if not think_chain:
            text = ResponseFormatter._remove_think_tags(text)
        
        text = ResponseFormatter._clean_internal_tokens(text)
        text = ResponseFormatter._apply_markdown_format(text)
        text = ResponseFormatter._apply_education_format(text)
        text = ResponseFormatter._final_cleanup(text)
        return text.strip()
    
    @staticmethod
    def format_psychology_response(text: str, think_chain: bool = True) -> str:
        """格式化心理相关响应"""
        if not think_chain:
            text = ResponseFormatter._remove_think_tags(text)
        
        text = ResponseFormatter._clean_internal_tokens(text)
        text = ResponseFormatter._apply_markdown_format(text)
        text = ResponseFormatter._apply_psychology_format(text)
        text = ResponseFormatter._final_cleanup(text)
        return text.strip()
    
    @staticmethod
    def format_general_response(text: str, think_chain: bool = True) -> str:
        """通用响应格式化"""
        if not think_chain:
            text = ResponseFormatter._remove_think_tags(text)
        
        text = ResponseFormatter._clean_internal_tokens(text)
        text = ResponseFormatter._apply_markdown_format(text)
        text = ResponseFormatter._final_cleanup(text)
        return text.strip()
    
    @staticmethod
    def _remove_think_tags(text: str) -> str:
        """移除think标签"""
        think_pattern = r'<think>(.*?)</think>'
        return re.sub(think_pattern, '', text, flags=re.DOTALL)
    
    @staticmethod
    def _clean_internal_tokens(text: str) -> str:
        """清理内部标记"""
        internal_tokens = [
            '<|im_end|>', '<|endoftext|>', '<|fim_prefix|>', 
            '<|fim_middle|>', '<|fim_suffix|>', '<|fim_pad|>',
            '<|repo_name|>', '<|file_sep|>', '<|vision_start|>',
            '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>',
            '<|video_pad|>', '<|object_ref_start|>', '<|object_ref_end|>',
            '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>'
        ]
        
        for token in internal_tokens:
            text = text.replace(token, '')
        
        return text
    
    @staticmethod
    def _apply_markdown_format(text: str) -> str:
        """应用markdown格式化"""
        # 处理三级标题
        text = re.sub(r'###\s*(.+?)(?:\n|$)', r'\n### \1\n', text)
        # 处理二级标题
        text = re.sub(r'##\s*(.+?)(?:\n|$)', r'\n## \1\n', text)
        # 处理一级标题
        text = re.sub(r'#\s*(.+?)(?:\n|$)', r'\n# \1\n', text)
        
        # 处理分割线
        text = re.sub(r'---+\s*', '\n' + '-' * 50 + '\n', text)
        
        # 处理无序列表
        text = re.sub(r'^\s*-\s+', '• ', text, flags=re.MULTILINE)
        # 处理有序列表
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # 处理加粗文本
        text = re.sub(r'\*\*(.+?)\*\*', r'**\1**', text)
        
        # 处理代码块
        text = re.sub(r'```(?:\w+)?\n?(.*?)\n?```', r'\n\1\n', text, flags=re.DOTALL)
        
        return text
    
    @staticmethod
    def _apply_medical_format(text: str) -> str:
        """医学特定格式化"""
        # 医学关键词高亮
        medical_terms = {
            r'\b(高血压|糖尿病|冠心病|脑卒中|心肌梗死|肺炎|哮喘)\b': '**\\1**',
            r'\b(病因|症状|诊断|治疗|预防|并发症)\b': '**\\1**',
            r'\b(建议|注意|警告|危险)\b': '**\\1**',
            r'\b(正常值|参考范围|指标)\b': '**\\1**',
            r'\b(药物|用药|剂量)\b': '**\\1**',
            r'\b(检查|检验|检测)\b': '**\\1**',
            r'\b(手术|操作|治疗)\b': '**\\1**',
        }
        
        for pattern, replacement in medical_terms.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    @staticmethod
    def _apply_finance_format(text: str) -> str:
        """金融特定格式化"""
        finance_terms = {
            r'\b(股票|基金|债券|投资|理财|保险)\b': '**\\1**',
            r'\b(收益率|风险|回报|资产|负债)\b': '**\\1**',
            r'\b(经济|市场|政策|监管|合规)\b': '**\\1**',
        }
        
        for pattern, replacement in finance_terms.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    @staticmethod
    def _apply_legal_format(text: str) -> str:
        """法律特定格式化"""
        legal_terms = {
            r'\b(法律|法规|条例|合同|协议)\b': '**\\1**',
            r'\b(权利|义务|责任|诉讼|仲裁)\b': '**\\1**',
            r'\b(证据|证人|法院|法官|律师)\b': '**\\1**',
        }
        
        for pattern, replacement in legal_terms.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    @staticmethod
    def _apply_education_format(text: str) -> str:
        """教育特定格式化"""
        education_terms = {
            r'\b(教育|学习|教学|课程|考试)\b': '**\\1**',
            r'\b(学生|教师|学校|教材|作业)\b': '**\\1**',
            r'\b(知识|技能|能力|素质|发展)\b': '**\\1**',
        }
        
        for pattern, replacement in education_terms.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    @staticmethod
    def _apply_psychology_format(text: str) -> str:
        """心理特定格式化"""
        psychology_terms = {
            r'\b(心理|情绪|情感|认知|行为)\b': '**\\1**',
            r'\b(压力|焦虑|抑郁|治疗|咨询)\b': '**\\1**',
            r'\b(心理健康|心理咨询|心理治疗)\b': '**\\1**',
        }
        
        for pattern, replacement in psychology_terms.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    @staticmethod
    def _final_cleanup(text: str) -> str:
        """最终清理"""
        # 清理多余空行
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # 清理多余空格
        text = re.sub(r'[ \t]+', ' ', text)
        # 移除行尾空格
        text = re.sub(r' +\n', '\n', text)
        
        return text
    
    @staticmethod
    def format_for_api(text: str, domain: str = "medical", think_chain: bool = True) -> Dict:
        """为API响应格式化文本"""
        if domain == "medical":
            formatted_text = ResponseFormatter.format_medical_response(text, think_chain)
        elif domain == "finance":
            formatted_text = ResponseFormatter.format_finance_response(text, think_chain)
        elif domain == "legal":
            formatted_text = ResponseFormatter.format_legal_response(text, think_chain)
        elif domain == "education":
            formatted_text = ResponseFormatter.format_education_response(text, think_chain)
        elif domain == "psychology":
            formatted_text = ResponseFormatter.format_psychology_response(text, think_chain)
        else:
            formatted_text = ResponseFormatter.format_general_response(text, think_chain)
        
        return {
            "original": text,
            "formatted": formatted_text,
            "domain": domain,
            "think_chain": think_chain,
            "length": len(formatted_text)
        }