# src/knowledge/__init__.py - 修复后的完整文件

from .assembler import KnowledgeAssembler

# 全局装配器缓存
_domain_assemblers = {}

def get_knowledge_base(domain, repo_root=None):
    """获取指定领域的知识库 - 使用装配器"""
    if domain not in _domain_assemblers:
        # 创建新的装配器
        assembler = KnowledgeAssembler(domain, repo_root)
        _domain_assemblers[domain] = assembler
    else:
        assembler = _domain_assemblers[domain]
    
    # 从装配器获取建议规则
    advice_rules = assembler.get_advice_rules()
    
    # 创建基于装配器的建议函数
    def advice_func(condition, response):
        """基于装配器规则的建议函数"""
        # 检查条件是否匹配任何建议规则
        for pattern, advice in advice_rules.items():
            if pattern in condition:
                if advice not in response:
                    response += f"\n\n【{domain.capitalize()}建议】{advice}"
        
        # 添加领域特定的通用建议
        domain_general_advice = {
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
        
        general_advices = domain_general_advice.get(domain, [])
        for signal, warning in general_advices:
            if signal in response and warning not in response:
                response += f"\n\n【重要提醒】{warning}"
        
        return response
    
    return assembler.get_knowledge_base(), assembler.get_term_map(), advice_func

def get_domain_assembler(domain, repo_root=None):
    """直接获取领域装配器实例"""
    if domain not in _domain_assemblers:
        _domain_assemblers[domain] = KnowledgeAssembler(domain, repo_root)
    return _domain_assemblers[domain]

def clear_assembler_cache():
    """清空装配器缓存"""
    _domain_assemblers.clear()

# 保持向后兼容的原有函数
def get_domain_specific_advice(domain, condition, response):
    """获取领域特定建议"""
    knowledge_base, term_map, advice_func = get_knowledge_base(domain)
    return advice_func(condition, response)