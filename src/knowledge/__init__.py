# src/knowledge/__init__.py
from .medical import MEDICAL_KNOWLEDGE_BASE, MEDICAL_TERM_MAP, get_domain_specific_advice as medical_advice
from .legal import LEGAL_KNOWLEDGE_BASE, LEGAL_TERM_MAP, get_domain_specific_advice as legal_advice
from .psychology import PSYCHOLOGY_KNOWLEDGE_BASE, PSYCHOLOGY_TERM_MAP, get_domain_specific_advice as psychology_advice
from .exam import EXAM_KNOWLEDGE_BASE, EXAM_TERM_MAP, get_domain_specific_advice as exam_advice

def get_knowledge_base(domain):
    """获取指定领域的知识库"""
    if domain == "medical":
        return MEDICAL_KNOWLEDGE_BASE, MEDICAL_TERM_MAP, medical_advice
    elif domain == "legal":
        return LEGAL_KNOWLEDGE_BASE, LEGAL_TERM_MAP, legal_advice
    elif domain == "psychology":
        return PSYCHOLOGY_KNOWLEDGE_BASE, PSYCHOLOGY_TERM_MAP, psychology_advice
    elif domain == "exam":
        return EXAM_KNOWLEDGE_BASE, EXAM_TERM_MAP, exam_advice
    else:
        return {}, {}, lambda condition, response: response