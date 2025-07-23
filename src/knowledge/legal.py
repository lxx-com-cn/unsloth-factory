# src/knowledge/legal.py
LEGAL_KNOWLEDGE_BASE = {
    "合同纠纷": ["违约责任", "合同解除", "赔偿损失", "继续履行"],
    "知识产权": ["著作权", "专利权", "商标权", "侵权赔偿"],
    "劳动争议": ["劳动合同", "工资支付", "加班费", "解除劳动合同"],
    "婚姻家庭": ["离婚", "财产分割", "抚养权", "赡养费"],
    "交通事故": ["责任认定", "赔偿标准", "保险理赔", "伤残鉴定"],
    "房产纠纷": ["房屋买卖", "租赁合同", "物业管理", "产权登记"],
    "刑事辩护": ["取保候审", "辩护策略", "量刑建议", "证据收集"],
    "公司事务": ["股权转让", "公司章程", "公司治理", "股东权利"],
}

LEGAL_TERM_MAP = {
    "Plaintiff": "原告",
    "Defendant": "被告",
    "Tort": "侵权",
    "Lawsuit": "诉讼",
    "Litigation": "诉讼",
    "Contract": "合同",
    "Intellectual Property": "知识产权",
    "Labor Dispute": "劳动争议",
    "Divorce": "离婚",
    "Alimony": "赡养费",
    "Traffic Accident": "交通事故",
    "Real Estate": "房产",
    "Criminal Defense": "刑事辩护",
    "Corporate Affairs": "公司事务",
}

def get_domain_specific_advice(condition, response):
    """获取法律领域特定建议"""
    if condition == "合同纠纷":
        return response + "\n\n建议: 收集书面合同、沟通记录等证据"
    elif condition == "知识产权":
        return response + "\n\n建议: 咨询专业知识产权律师，准备权属证明"
    elif condition == "劳动争议":
        return response + "\n\n建议: 收集劳动合同、工资单、考勤记录等证据"
    elif condition == "刑事辩护":
        return response + "\n\n建议: 尽快联系专业刑事辩护律师"
    return response