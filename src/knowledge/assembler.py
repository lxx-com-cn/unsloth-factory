# src/knowledge/assembler.py - 完整修改版（含自动创建）

import os
import re
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Union

logger = logging.getLogger(__name__)

class KnowledgeAssembler:
    SUPPORTED_FORMATS = {
        '.txt': '_load_txt_file',
        '.json': '_load_json_file',
        '.yaml': '_load_yaml_file',
        '.yml': '_load_yaml_file'
    }

    def __init__(self, domain: str, repo_root: str = None):
        self.domain = domain
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).parents[2] / "knowledge_repo"
        self.knowledge_base = {}
        self.term_map = {}
        self.advice_rules = {}
        self.file_stats = {}
        self.config = {}

        # ========== 自动创建领域目录并生成默认知识 ==========
        self._ensure_domain_knowledge()

        self._load_domain_knowledge()

    def _ensure_domain_knowledge(self):
        """如果领域目录不存在，自动创建并填充默认知识"""
        domain_path = self.repo_root / self.domain
        if not domain_path.exists():
            logger.info(f"领域知识目录不存在，自动创建: {domain_path}")
            domain_path.mkdir(parents=True, exist_ok=True)

            # 创建默认配置文件
            default_config = {
                "domain": self.domain,
                "description": f"自动生成的{self.domain}领域知识库",
                "domain_categories": {
                    "disease_categories": "疾病分类",
                    "symptoms": "症状"
                },
                "file_structure": {
                    "disease_categories": ["diseases.json"],
                    "general_files": ["medical_terms.json", "medical_advice.json"]
                }
            }
            config_path = domain_path / "config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)

            # 创建默认疾病知识
            diseases = {
                "高血压": "高血压是一种以动脉血压持续升高为特征的疾病，常引起心、脑、肾等器官损害。",
                "糖尿病": "糖尿病是一种以高血糖为特征的代谢性疾病，由胰岛素分泌缺陷或作用受损引起。",
                "冠心病": "冠心病是冠状动脉粥样硬化导致心肌缺血、缺氧或坏死的心脏病。"
            }
            disease_path = domain_path / "diseases.json"
            with open(disease_path, 'w', encoding='utf-8') as f:
                json.dump(diseases, f, ensure_ascii=False, indent=2)

            # 创建默认术语映射
            terms = {
                "hypertension": "高血压",
                "diabetes": "糖尿病",
                "coronary heart disease": "冠心病"
            }
            term_path = domain_path / "medical_terms.json"
            with open(term_path, 'w', encoding='utf-8') as f:
                json.dump(terms, f, ensure_ascii=False, indent=2)

            # 创建默认建议规则
            advice = {
                "自行用药": "请勿自行用药，务必在医生指导下治疗",
                "催吐": "中毒情况下不要自行催吐，立即就医",
                "心梗": "立即就医，不要自行用药"
            }
            advice_path = domain_path / "medical_advice.json"
            with open(advice_path, 'w', encoding='utf-8') as f:
                json.dump(advice, f, ensure_ascii=False, indent=2)

            logger.info(f"已自动生成默认医学知识库: {domain_path}")

    def _load_domain_knowledge(self):
        domain_path = self.repo_root / self.domain
        if not domain_path.exists():
            logger.warning(f"领域知识目录不存在: {domain_path}")
            return

        logger.info(f"开始加载领域知识: {self.domain}")
        self.config = self._load_config(domain_path)
        if not self.config:
            logger.warning("配置文件加载失败，使用自动扫描模式")
            self._auto_scan_files(domain_path)
            return

        self._load_structured_files(domain_path)
        logger.info(f"领域知识加载完成: {self.domain}")
        self._print_loading_stats()

    def _load_config(self, domain_path: Path) -> Dict:
        config_path = domain_path / "config.json"
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_path}")
            return {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败 {config_path}: {e}")
            return {}

    def _load_structured_files(self, domain_path: Path):
        config = self.config
        if 'file_structure' in config and 'disease_categories' in config['file_structure']:
            for disease_file in config['file_structure']['disease_categories']:
                file_path = domain_path / disease_file
                if file_path.exists():
                    self._load_single_file(file_path, 'diseases')
                else:
                    logger.warning(f"疾病分类文件不存在: {file_path}")
        if 'file_structure' in config and 'general_files' in config['file_structure']:
            for general_file in config['file_structure']['general_files']:
                file_path = domain_path / general_file
                if file_path.exists():
                    if 'term' in general_file.lower():
                        self._load_single_file(file_path, 'terms')
                    elif 'advice' in general_file.lower():
                        self._load_single_file(file_path, 'advice')
                    else:
                        self._load_single_file(file_path, 'general')
                else:
                    logger.warning(f"通用文件不存在: {file_path}")
        self._load_remaining_files(domain_path)

    def _load_single_file(self, file_path: Path, file_type: str):
        loader_method = getattr(self, self.SUPPORTED_FORMATS[file_path.suffix])
        try:
            data = loader_method(file_path)
            filename = file_path.stem
            if file_type == 'terms':
                self.term_map.update(data)
            elif file_type == 'advice':
                self.advice_rules.update(data)
            elif file_type == 'diseases':
                if isinstance(data, dict):
                    self.knowledge_base.update(data)
                else:
                    self.knowledge_base[filename] = data
            else:
                self.knowledge_base[filename] = data
            self.file_stats[filename] = {
                'path': str(file_path),
                'type': file_type,
                'format': file_path.suffix,
                'items': len(data) if isinstance(data, (dict, list)) else 1
            }
            logger.debug(f"加载文件: {file_path.name} -> {file_type}")
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")

    def _load_remaining_files(self, domain_path: Path):
        config_files = set()
        if 'file_structure' in self.config:
            for file_list in self.config['file_structure'].values():
                config_files.update(file_list)
        for file_path in domain_path.iterdir():
            if (file_path.is_file() and
                file_path.suffix in self.SUPPORTED_FORMATS and
                file_path.name not in config_files):
                logger.info(f"加载配置外文件: {file_path.name}")
                self._load_single_file(file_path, 'additional')

    def _auto_scan_files(self, domain_path: Path):
        logger.info("使用自动扫描模式加载文件")
        for file_path in domain_path.iterdir():
            if file_path.is_file() and file_path.suffix in self.SUPPORTED_FORMATS:
                self._load_single_file(file_path, 'auto')

    def _load_txt_file(self, file_path: Path) -> Union[Dict, List]:
        content = file_path.read_text(encoding='utf-8')
        if self._is_key_value_format(content):
            return self._parse_key_value_txt(content)
        elif self._is_category_format(content):
            return self._parse_category_txt(content)
        else:
            return self._parse_lines_txt(content)

    def _is_key_value_format(self, content: str) -> bool:
        lines = content.split('\n')
        key_value_lines = [line for line in lines if ':' in line and not line.strip().startswith('#')]
        return len(key_value_lines) > len(lines) * 0.5

    def _is_category_format(self, content: str) -> bool:
        return any(line.strip().endswith(':') for line in content.split('\n'))

    def _parse_key_value_txt(self, content: str) -> Dict:
        result = {}
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
        return result

    def _parse_category_txt(self, content: str) -> Dict:
        result = {}
        current_category = None
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.endswith(':'):
                current_category = line[:-1]
                result[current_category] = []
            elif current_category and line.startswith('- '):
                result[current_category].append(line[2:])
            elif current_category:
                result[current_category].append(line)
        return result

    def _parse_lines_txt(self, content: str) -> List:
        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
        return lines

    def _load_json_file(self, file_path: Path) -> Dict:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_yaml_file(self, file_path: Path) -> Dict:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _print_loading_stats(self):
        total_knowledge_items = 0
        for items in self.knowledge_base.values():
            if isinstance(items, (list, dict)):
                total_knowledge_items += len(items)
            else:
                total_knowledge_items += 1
        logger.info(f"知识库统计 - {self.domain}:")
        logger.info(f"  知识条目: {total_knowledge_items}")
        logger.info(f"  术语映射: {len(self.term_map)}")
        logger.info(f"  建议规则: {len(self.advice_rules)}")
        logger.info(f"  加载文件: {len(self.file_stats)}")
        if self.config:
            logger.info(f"  配置驱动: 是")
            if 'domain_categories' in self.config:
                loaded_categories = []
                for category_file in self.config.get('file_structure', {}).get('disease_categories', []):
                    if Path(category_file).stem in self.file_stats:
                        category_name = self.config['domain_categories'].get(Path(category_file).stem, '未知')
                        loaded_categories.append(category_name)
                logger.info(f"  已加载疾病分类: {', '.join(loaded_categories)}")

    def get_knowledge_base(self) -> Dict:
        return self.knowledge_base

    def get_term_map(self) -> Dict:
        return self.term_map

    def get_advice_rules(self) -> Dict:
        return self.advice_rules

    def get_domain_info(self) -> Dict:
        return {
            "domain": self.domain,
            "config_loaded": bool(self.config),
            "knowledge_base_size": sum(
                len(items) if isinstance(items, (list, dict)) else 1
                for items in self.knowledge_base.values()
            ),
            "term_map_size": len(self.term_map),
            "advice_rules_size": len(self.advice_rules),
            "loaded_files": self.file_stats,
            "domain_categories": self.config.get('domain_categories', {}) if self.config else {}
        }

    def get_disease_categories(self) -> Dict[str, str]:
        if self.config and 'domain_categories' in self.config:
            return self.config['domain_categories']
        return {}

    def search_knowledge(self, query: str, category: str = None) -> List[str]:
        results = []
        query_lower = query.lower()
        search_targets = {}
        if category and category in self.knowledge_base:
            search_targets[category] = self.knowledge_base[category]
        else:
            search_targets = self.knowledge_base
        for key, value in search_targets.items():
            if query_lower in key.lower():
                if isinstance(value, list):
                    results.extend(value)
                else:
                    results.append(str(value))
            if isinstance(value, list):
                for item in value:
                    if query_lower in item.lower():
                        results.append(item)
            elif isinstance(value, str) and query_lower in value.lower():
                results.append(value)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if query_lower in sub_key.lower():
                        results.append(sub_key)
                    if isinstance(sub_value, list):
                        for item in sub_value:
                            if query_lower in item.lower():
                                results.append(item)
        return list(set(results))