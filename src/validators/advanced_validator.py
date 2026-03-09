# src/validators/advanced_validator.py
import os
import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from src.knowledge import get_knowledge_base
from src.validators.cot_validator import CotValidator

logger = logging.getLogger(__name__)

# ================= 数据结构定义 =================
@dataclass
class AdvancedScore:
    completeness: float      # 完整性得分
    accuracy: float          # 准确性得分
    structure: float         # 结构性得分
    safety: float            # 安全性得分
    reasoning: float         # 思维链质量（从 CotValidator 继承）
    consistency: float       # 与期望输出的一致性
    total: float             # 综合总分

@dataclass
class DetailedResult:
    prompt: str
    expected: str
    response: str
    think_content: str
    answer_content: str
    scores: AdvancedScore
    issues: List[str]
    details: Dict[str, Any]  # 各维度详细匹配信息

# ================= 高级验证器类 =================
class AdvancedValidator:
    """
    多维度验证器，以训练数据中的期望输出为基准，对模型生成答案进行综合评分
    """
    def __init__(self, domain: str = "medical", knowledge_repo: Optional[str] = None):
        self.domain = domain
        self.cot_validator = CotValidator()
        
        # 加载领域知识库（用于准确性检测和安全关键词）
        self.knowledge_base, self.term_map, self.advice_func = get_knowledge_base(domain, knowledge_repo)
        
        # 各维度权重（可根据需要调整）
        self.weights = {
            "completeness": 0.25,   # 完整性最重要
            "accuracy": 0.25,        # 准确性同等重要
            "structure": 0.10,       # 结构性次要
            "safety": 0.15,          # 安全性
            "reasoning": 0.10,       # 思维链质量（权重降低，因为不要求长篇思考）
            "consistency": 0.15,     # 语义一致性
        }
        
        # 加载领域关键要点（以 medical 为例，可扩展）
        self._load_etiology_points()
        
        # 安全提示关键词
        self.safety_keywords = [
            "就医", "医生", "专科", "检查", "治疗",
            "不要自行用药", "请勿自行", "遵医嘱", "咨询专业"
        ]
        
        # 结构标记检测
        self.structure_patterns = [
            (r'#+ .+', '标题'),          # Markdown 标题
            (r'\*\*.*?\*\*', '粗体'),     # 粗体
            (r'^\s*[\-\*]\s+', '无序列表'), # 无序列表
            (r'^\s*\d+\.\s+', '有序列表'),  # 有序列表
            (r'---', '分割线'),           # 分割线
            (r'\n\n', '段落分隔'),        # 段落
        ]

    def _load_etiology_points(self):
        """加载该领域常见病因要点（以早泄为例，可根据领域扩展）"""
        # 这里可以从知识库动态构建，或硬编码典型要点
        # 当前以 medical 为例，硬编码早泄常见病因分类
        self.etiology_points = {
            "心理因素": ["焦虑", "压力", "抑郁", "紧张", "性经验不足", "伴侣关系"],
            "生理因素": ["前列腺炎", "激素水平", "睾酮", "神经敏感", "甲状腺", "糖尿病"],
            "生活习惯": ["手淫", "熬夜", "吸烟", "饮酒", "缺乏运动", "作息不规律"],
            "疾病关联": ["心血管疾病", "糖尿病", "泌尿系统感染", "神经系统疾病"],
            "建议措施": ["就医", "检查", "行为疗法", "药物治疗", "心理疏导", "生活方式调整"]
        }
        # 将所有要点展平用于匹配
        self.all_keywords = []
        for category, keywords in self.etiology_points.items():
            self.all_keywords.extend(keywords)

    def evaluate(self,
                 responses: List[Dict[str, Any]],
                 show_progress: bool = True) -> Dict[str, Any]:
        """
        对一批响应进行多维度评估
        responses: 每个元素包含 'prompt', 'expected_output', 'model_response'
        """
        results = []
        iterator = tqdm(responses) if show_progress else responses
        
        for item in iterator:
            prompt = item.get('prompt', '')
            expected = item.get('expected_output', '')
            response = item.get('model_response', '')
            
            # 使用 CotValidator 提取思维链信息和语义一致性
            cot_result = self.cot_validator.validate_response(response, expected)
            
            # 分离 think 和 answer 内容
            think_content = cot_result.think_content
            answer_content = cot_result.answer_content
            
            # 多维度评分
            scores, details = self._compute_scores(
                prompt, expected, response,
                think_content, answer_content,
                cot_result
            )
            
            # 收集问题
            issues = cot_result.issues.copy()
            if scores.completeness < 0.5:
                issues.append("完整性不足")
            if scores.accuracy < 0.8:
                issues.append("可能存在医学不准确信息")
            if scores.safety < 0.8:
                issues.append("缺少必要的安全提示")
            if scores.structure < 0.3:
                issues.append("结构混乱或缺乏组织")
            
            results.append(DetailedResult(
                prompt=prompt,
                expected=expected,
                response=response,
                think_content=think_content,
                answer_content=answer_content,
                scores=scores,
                issues=issues,
                details=details
            ))
        
        # 计算总体统计
        stats = self._compute_statistics(results)
        return {
            "individual_results": [self._to_dict(r) for r in results],
            "statistics": stats
        }

    def _compute_scores(self, prompt: str, expected: str, response: str,
                        think: str, answer: str,
                        cot_result) -> Tuple[AdvancedScore, Dict]:
        """计算各维度得分"""
        # 1. 完整性得分：基于预定义要点列表
        completeness = self._score_completeness(answer)
        
        # 2. 准确性得分：基于领域知识库检测错误
        accuracy = self._score_accuracy(answer)
        
        # 3. 结构性得分：检测 Markdown 标题、列表等
        structure = self._score_structure(answer)
        
        # 4. 安全性得分：检测安全关键词
        safety = self._score_safety(answer)
        
        # 5. 思维链质量（直接使用 CotValidator 的评分）
        reasoning = cot_result.reasoning_quality
        
        # 6. 语义一致性（CotValidator 已提供）
        consistency = cot_result.answer_consistency
        
        # 综合总分（加权平均）
        total = (
            completeness * self.weights["completeness"] +
            accuracy * self.weights["accuracy"] +
            structure * self.weights["structure"] +
            safety * self.weights["safety"] +
            reasoning * self.weights["reasoning"] +
            consistency * self.weights["consistency"]
        )
        
        scores = AdvancedScore(
            completeness=completeness,
            accuracy=accuracy,
            structure=structure,
            safety=safety,
            reasoning=reasoning,
            consistency=consistency,
            total=total
        )
        
        # 收集详细匹配信息
        details = {
            "completeness_matches": self._get_completeness_matches(answer),
            "accuracy_issues": self._detect_accuracy_issues(answer),
            "structure_markers": self._detect_structure_markers(answer),
            "safety_present": [kw for kw in self.safety_keywords if kw in answer]
        }
        return scores, details

    def _score_completeness(self, text: str) -> float:
        """完整性：计算命中预定义关键词的比例（按类别计分）"""
        if not text:
            return 0.0
        text_lower = text.lower()
        # 按类别计分，每个类别至少命中一个关键词即得该类别分
        category_score = 0.0
        total_categories = len(self.etiology_points)
        for category, keywords in self.etiology_points.items():
            if any(kw in text_lower for kw in keywords):
                category_score += 1.0 / total_categories
        return category_score

    def _get_completeness_matches(self, text: str) -> Dict[str, List[str]]:
        """返回每个分类下匹配到的关键词"""
        matches = {}
        text_lower = text.lower()
        for category, keywords in self.etiology_points.items():
            found = [kw for kw in keywords if kw in text_lower]
            if found:
                matches[category] = found
        return matches

    def _score_accuracy(self, text: str) -> float:
        """准确性：根据领域知识库检测错误（目前简单实现，可扩展）"""
        # 检测危险错误模式
        error_patterns = [
            (r"自行用药", -0.3),
            (r"催吐", -0.5),
            (r"自行停药", -0.3),
        ]
        score = 1.0
        for pattern, penalty in error_patterns:
            if re.search(pattern, text):
                score += penalty
        return max(score, 0.0)

    def _detect_accuracy_issues(self, text: str) -> List[str]:
        issues = []
        if re.search(r"自行用药", text):
            issues.append("建议自行用药")
        if re.search(r"催吐", text):
            issues.append("建议催吐")
        if re.search(r"自行停药", text):
            issues.append("建议自行停药")
        return issues

    def _score_structure(self, text: str) -> float:
        """结构性：检测是否使用标题、列表等结构元素"""
        if not text:
            return 0.0
        score = 0.0
        for pattern, name in self.structure_patterns:
            if re.search(pattern, text, re.MULTILINE):
                score += 0.2  # 每个结构元素加 0.2，最多 1.0
        return min(score, 1.0)

    def _detect_structure_markers(self, text: str) -> List[str]:
        markers = []
        for pattern, name in self.structure_patterns:
            if re.search(pattern, text, re.MULTILINE):
                markers.append(name)
        return markers

    def _score_safety(self, text: str) -> float:
        """安全性：检查是否包含就医建议等安全提示"""
        if not text:
            return 0.0
        safety_count = sum(1 for kw in self.safety_keywords if kw in text)
        # 根据命中数量打分，至少有一个得0.5，两个以上得1.0
        if safety_count >= 2:
            return 1.0
        elif safety_count == 1:
            return 0.5
        else:
            return 0.0

    def _compute_statistics(self, results: List[DetailedResult]) -> Dict[str, Any]:
        """计算总体统计"""
        n = len(results)
        if n == 0:
            return {}
        
        avg_completeness = sum(r.scores.completeness for r in results) / n
        avg_accuracy = sum(r.scores.accuracy for r in results) / n
        avg_structure = sum(r.scores.structure for r in results) / n
        avg_safety = sum(r.scores.safety for r in results) / n
        avg_reasoning = sum(r.scores.reasoning for r in results) / n
        avg_consistency = sum(r.scores.consistency for r in results) / n
        avg_total = sum(r.scores.total for r in results) / n
        
        # 常见问题统计
        all_issues = []
        for r in results:
            all_issues.extend(r.issues)
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return {
            "total_samples": n,
            "avg_completeness": avg_completeness,
            "avg_accuracy": avg_accuracy,
            "avg_structure": avg_structure,
            "avg_safety": avg_safety,
            "avg_reasoning": avg_reasoning,
            "avg_consistency": avg_consistency,
            "avg_total_score": avg_total,
            "common_issues": issue_counts
        }

    def _to_dict(self, result: DetailedResult) -> Dict:
        """将结果转换为可序列化的字典"""
        return {
            "prompt": result.prompt,
            "expected": result.expected,
            "response": result.response,
            "think_content": result.think_content,
            "answer_content": result.answer_content,
            "scores": {
                "completeness": result.scores.completeness,
                "accuracy": result.scores.accuracy,
                "structure": result.scores.structure,
                "safety": result.scores.safety,
                "reasoning": result.scores.reasoning,
                "consistency": result.scores.consistency,
                "total": result.scores.total
            },
            "issues": result.issues,
            "details": result.details
        }

    def generate_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """生成可读的报告"""
        stats = results.get("statistics", {})
        lines = [
            "=" * 70,
            "              高级验证报告（多维度评分）",
            "=" * 70,
            "",
            f"样本数: {stats.get('total_samples', 0)}",
            f"综合平均分: {stats.get('avg_total_score', 0):.3f}",
            "",
            "维度平均得分:",
            f"  完整性     : {stats.get('avg_completeness', 0):.3f}",
            f"  准确性     : {stats.get('avg_accuracy', 0):.3f}",
            f"  结构性     : {stats.get('avg_structure', 0):.3f}",
            f"  安全性     : {stats.get('avg_safety', 0):.3f}",
            f"  思维链质量 : {stats.get('avg_reasoning', 0):.3f}",
            f"  语义一致性 : {stats.get('avg_consistency', 0):.3f}",
            "",
            "常见问题统计:",
        ]
        issues = stats.get('common_issues', {})
        if issues:
            for issue, count in sorted(issues.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  - {issue}: {count} 次")
        else:
            lines.append("  无")
        
        lines.append("=" * 70)
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"高级验证报告已保存: {output_path}")
        
        return report