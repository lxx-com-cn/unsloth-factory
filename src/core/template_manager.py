# src/core/template_manager.py
import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TemplateManager:
    """管理模型模板"""
    
    TEMPLATE_DIR = "templates"
    
    def __init__(self, template_dir=None):
        self.template_dir = template_dir or self.TEMPLATE_DIR
        self.templates = self.load_all_templates()
    
    def load_all_templates(self):
        """加载所有可用模板"""
        templates = {}
        for file_path in Path(self.template_dir).glob("*.json"):
            template_name = file_path.stem
            templates[template_name] = self.load_template(template_name)
        return templates
    
    def load_template(self, template_name):
        """加载指定模板"""
        # 检查是否是绝对路径
        if os.path.isabs(template_name) and os.path.exists(template_name):
            template_path = template_name
        else:
            # 处理可能的扩展名
            if not template_name.endswith(".json"):
                template_name += ".json"
            
            template_path = os.path.join(self.template_dir, template_name)
            
            if not os.path.exists(template_path):
                # 尝试不带扩展名
                template_path = os.path.join(self.template_dir, template_name.replace(".json", ""))
                if not os.path.exists(template_path):
                    raise FileNotFoundError(f"Template not found: {template_name}")
        
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template = json.load(f)
            
            # 添加模板名称
            template["name"] = template_name.replace(".json", "")
            return template
        except Exception as e:
            logger.error(f"Error loading template {template_path}: {str(e)}")
            raise
    
    def get_template(self, template_name):
        """获取模板"""
        # 尝试从缓存中获取
        if template_name in self.templates:
            return self.templates[template_name]
        
        # 动态加载
        return self.load_template(template_name)
    
    def list_templates(self):
        """列出所有可用模板"""
        return list(self.templates.keys())