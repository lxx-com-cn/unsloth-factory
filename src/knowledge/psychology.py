# src/knowledge/psychology.py
PSYCHOLOGY_KNOWLEDGE_BASE = {
    "抑郁": ["情绪低落", "兴趣减退", "睡眠障碍", "食欲改变", "自杀意念"],
    "焦虑": ["过度担心", "紧张不安", "心悸出汗", "回避行为"],
    "强迫症": ["强迫思维", "强迫行为", "重复动作", "仪式化行为"],
    "社交恐惧": ["社交回避", "脸红出汗", "心跳加速", "评价焦虑"],
    "创伤后应激": ["闪回", "噩梦", "警觉增高", "回避相关刺激"],
    "人格障碍": ["人际关系困难", "情绪不稳定", "自我认同问题", "行为模式固定"],
    "儿童心理": ["注意力缺陷", "多动", "学习困难", "分离焦虑"],
    "婚姻咨询": ["沟通困难", "信任问题", "亲密关系", "冲突解决"],
}

PSYCHOLOGY_TERM_MAP = {
    "Depression": "抑郁症",
    "Anxiety": "焦虑症",
    "OCD": "强迫症",
    "PTSD": "创伤后应激障碍",
    "CBT": "认知行为疗法",
    "DBT": "辩证行为疗法",
    "Psychodynamic": "心理动力学",
    "Self-esteem": "自尊",
    "Attachment": "依恋",
    "Resilience": "心理韧性",
}

def get_domain_specific_advice(condition, response):
    """获取心理咨询领域特定建议"""
    if condition == "抑郁":
        return response + "\n\n建议: 鼓励寻求专业帮助，提供支持性环境"
    elif condition == "焦虑":
        return response + "\n\n建议: 教授放松技巧，如深呼吸和渐进式肌肉放松"
    elif condition == "自杀意念":
        return response + "\n\n紧急建议: 立即联系心理危机干预热线或专业机构"
    elif condition == "儿童心理":
        return response + "\n\n建议: 家庭参与治疗，父母教育很重要"
    return response