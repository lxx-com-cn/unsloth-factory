# src/knowledge/exam.py
EXAM_KNOWLEDGE_BASE = {
    "数学": ["代数", "几何", "微积分", "概率统计", "线性代数"],
    "物理": ["力学", "电磁学", "热学", "光学", "量子物理"],
    "化学": ["无机化学", "有机化学", "物理化学", "分析化学", "生物化学"],
    "生物": ["细胞生物学", "遗传学", "生态学", "进化论", "生理学"],
    "历史": ["古代史", "近代史", "现代史", "世界史", "中国史"],
    "地理": ["自然地理", "人文地理", "区域地理", "地图学", "地质学"],
    "语文": ["文言文", "现代文", "作文", "阅读理解", "文学常识"],
    "英语": ["词汇", "语法", "阅读理解", "写作", "听力"],
}

EXAM_TERM_MAP = {
    "Algebra": "代数",
    "Geometry": "几何",
    "Calculus": "微积分",
    "Probability": "概率",
    "Mechanics": "力学",
    "Electromagnetism": "电磁学",
    "Inorganic Chemistry": "无机化学",
    "Organic Chemistry": "有机化学",
    "Cell Biology": "细胞生物学",
    "Genetics": "遗传学",
    "Ancient History": "古代史",
    "Modern History": "近代史",
    "Physical Geography": "自然地理",
    "Human Geography": "人文地理",
    "Classical Chinese": "文言文",
    "Composition": "作文",
}

def get_domain_specific_advice(condition, response):
    """获取考试领域特定建议"""
    if condition == "数学":
        return response + "\n\n建议: 注重解题步骤，检查计算过程"
    elif condition == "作文":
        return response + "\n\n建议: 明确中心思想，合理分段，注意开头和结尾"
    elif condition == "英语":
        return response + "\n\n建议: 注意时态和语态，使用高级词汇"
    elif condition == "物理":
        return response + "\n\n建议: 画图辅助分析，注意单位转换"
    return response