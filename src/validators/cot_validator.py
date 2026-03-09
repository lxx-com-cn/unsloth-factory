# src/validators/cot_validator.py - 优化版（修复逻辑连贯性误报）
import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

try:
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("jieba 或 scikit-learn 未安装，将使用简单的字符重叠评估一致性")

logger = logging.getLogger(__name__)

@dataclass
class CotValidationResult:
    has_think_block: bool
    has_answer_block: bool
    think_content: str
    answer_content: str
    reasoning_quality: float
    answer_consistency: float
    think_length: int
    answer_length: int
    issues: List[str]

class CotValidator:
    def __init__(self):
        self.think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        self.natural_think_patterns = [
            r'(.*?)(?:综上所述|因此|所以|答案是|建议如下|治疗方案|总结|总之)',
            r'(.*?)(?:Based on|Therefore|In conclusion|To summarize|In summary)',
            r'(嗯|让我|我需要|首先|考虑|分析).*?(?:综上所述|因此|所以)',
        ]
        self.reasoning_indicators = [
            "因为", "所以", "首先", "其次", "然后", "最后",
            "分析", "考虑", "根据", "由于", "因此", "结论",
            "可能", "或者", "比较", "需要", "应该",
            "because", "therefore", "first", "second", "then", "finally",
            "analyze", "consider", "according to", "conclusion", "need", "should"
        ]
        self.medical_terms = [
            "病因", "症状", "诊断", "治疗", "预防", "并发症",
            "药物", "手术", "检查", "体征", "预后", "病理",
            "高血压", "糖尿病", "冠心病", "早泄", "前列腺",
            "感染", "炎症", "肿瘤", "骨折"
        ]
        self.think_starters = [
            "嗯", "让我", "我需要", "首先", "考虑", "分析",
            "用户问的是", "这个问题", "从", "根据",
            "Hmm", "Let me", "I need to", "First", "Consider", "Analyze"
        ]
        self.think_enders = [
            "综上所述", "因此", "所以", "答案是", "建议如下",
            "治疗方案", "总结", "总之", "Based on", "Therefore",
            "In conclusion", "To summarize", "In summary", "The answer is"
        ]
        self.low_quality_markers = [
            "我不知道", "无法回答", "没有相关信息",
            "i don't know", "cannot answer", "no relevant information"
        ]

    def validate_response(self, response: str, expected_output: Optional[str] = None) -> CotValidationResult:
        issues = []
        think_match = self.think_pattern.search(response)
        has_think_block = think_match is not None
        think_content = think_match.group(1).strip() if think_match else ""
        answer_content = self.think_pattern.sub("", response).strip() if has_think_block else response

        if not has_think_block:
            think_content, answer_content, has_think_block = self._detect_natural_cot(response)
            if has_think_block:
                issues.append("使用自然语言CoT检测")

        think_length = len(think_content)
        answer_length = len(answer_content)

        if think_length < 20 and not has_think_block:
            issues.append("思维链过短或未检测到")
        elif think_length < 50 and has_think_block:
            issues.append("思维链较短")
        elif think_length > 2000:
            issues.append("思维链过长，可能包含冗余信息")

        reasoning_quality = self._evaluate_reasoning_quality_v3(think_content, answer_content, has_think_block)
        answer_consistency = self._check_answer_consistency_semantic(answer_content, expected_output) if expected_output else 1.0

        # ========== 修复逻辑连贯性误报 ==========
        logic_score = 1.0  # 默认连贯
        if has_think_block and answer_content:
            logic_score = self._check_logic_flow_improved(think_content, answer_content)
            if logic_score < 0.15:  # 降低阈值
                issues.append("思维链与答案逻辑不连贯")

        return CotValidationResult(
            has_think_block=has_think_block,
            has_answer_block=True,
            think_content=think_content,
            answer_content=answer_content,
            reasoning_quality=reasoning_quality,
            answer_consistency=answer_consistency,
            think_length=think_length,
            answer_length=answer_length,
            issues=issues
        )

    def _detect_natural_cot(self, response: str) -> Tuple[str, str, bool]:
        for pattern in self.natural_think_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                think_content = match.group(1).strip()
                answer_start = match.end(1)
                answer_content = response[answer_start:].strip()
                if len(think_content) > 50 and self._contains_reasoning(think_content):
                    return think_content, answer_content, True
        best_split = 0
        best_score = 0
        for i in range(100, min(len(response), 1000)):
            prefix = response[:i]
            suffix = response[i:]
            score = 0
            if any(starter in prefix[:50] for starter in self.think_starters):
                score += 2
            score += sum(1 for ind in self.reasoning_indicators if ind in prefix)
            if any(marker in suffix[:100] for marker in self.think_enders):
                score += 3
            if 100 < len(prefix) < 800:
                score += 1
            if score > best_score:
                best_score = score
                best_split = i
        if best_score >= 3 and best_split > 0:
            think_content = response[:best_split].strip()
            answer_content = response[best_split:].strip()
            return think_content, answer_content, True
        if len(response) > 300:
            split_pos = len(response) // 3
            think_part = response[:split_pos]
            if self._contains_reasoning(think_part):
                return think_part.strip(), response[split_pos:].strip(), True
        return "", response, False

    def _contains_reasoning(self, text: str) -> bool:
        indicators = ["因为", "所以", "首先", "其次", "分析", "考虑", "需要", "应该",
                     "because", "therefore", "first", "analyze", "consider"]
        return sum(1 for ind in indicators if ind in text.lower()) >= 2

    def _evaluate_reasoning_quality_v3(self, think_content: str, answer_content: str, has_think_block: bool) -> float:
        if not has_think_block:
            if self._contains_reasoning(answer_content) and len(answer_content) > 500:
                return 0.4
            return 0.0
        score = 0.0
        think_len = len(think_content)
        if 100 <= think_len <= 1000:
            score += 0.2
        elif think_len > 50:
            score += 0.1
        reasoning_count = sum(1 for ind in self.reasoning_indicators if ind in think_content.lower())
        score += min(reasoning_count * 0.05, 0.3)
        medical_term_count = sum(1 for term in self.medical_terms if term in think_content.lower())
        score += min(medical_term_count * 0.03, 0.2)
        if answer_content:
            think_words = set(re.findall(r'\b\w{2,}\b', think_content.lower()))
            answer_words = set(re.findall(r'\b\w{2,}\b', answer_content.lower()))
            if think_words and answer_words:
                overlap = len(think_words & answer_words) / len(think_words)
                score += min(overlap * 0.4, 0.3)
        if any(marker in think_content for marker in ["首先", "第一", "1.", "Step 1"]):
            score += 0.1
        if any(marker in think_content for marker in ["其次", "然后", "第二", "2."]):
            score += 0.1
        has_low_quality = any(marker in think_content.lower() for marker in self.low_quality_markers)
        if not has_low_quality:
            score += 0.2
        return min(score, 1.0)

    def _check_answer_consistency_semantic(self, generated_answer: str, expected_output: str) -> float:
        if not expected_output:
            return 1.0
        if not _SEMANTIC_AVAILABLE:
            return self._check_answer_consistency_fallback(generated_answer, expected_output)
        try:
            gen_seg = ' '.join(jieba.cut(generated_answer))
            exp_seg = ' '.join(jieba.cut(expected_output))
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([gen_seg, exp_seg])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.debug(f"语义相似度计算失败，回退到实体重叠: {e}")
            return self._check_answer_consistency_fallback(generated_answer, expected_output)

    def _check_answer_consistency_fallback(self, generated_answer: str, expected_output: str) -> float:
        gen_entities = set(self._extract_entities(generated_answer))
        exp_entities = set(self._extract_entities(expected_output))
        if not exp_entities:
            return 1.0 if not gen_entities else 0.5
        overlap = len(gen_entities & exp_entities)
        total = len(exp_entities)
        return overlap / total if total > 0 else 0.0

    def _extract_entities(self, text: str) -> List[str]:
        entities = []
        numbers = re.findall(r'\d+\.?\d*', text)
        entities.extend(numbers)
        chinese_entities = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        entities.extend(chinese_entities[:5])
        return entities

    # ========== 改进版逻辑流评分 ==========
    def _check_logic_flow_improved(self, think_content: str, answer_content: str) -> float:
        """改进的逻辑连贯性评估：结合关键词重叠、结论词检测和思考长度奖励"""
        if not think_content or not answer_content:
            return 0.0

        # 1. 基础词重叠率
        think_words = set(re.findall(r'\b\w{2,}\b', think_content.lower()))
        answer_words = set(re.findall(r'\b\w{2,}\b', answer_content.lower()))
        if think_words:
            overlap = len(think_words & answer_words) / len(think_words)
        else:
            overlap = 0.0

        # 2. 结论词奖励：如果思考结尾包含“因此”、“所以”等，认为逻辑连贯
        conclusion_bonus = 0.0
        think_last_200 = think_content[-200:].lower()
        if any(word in think_last_200 for word in ["因此", "所以", "综上所述", "综上", "总之", "hence", "therefore", "in summary"]):
            conclusion_bonus = 0.3

        # 3. 思考长度奖励：详细思考表明认真分析，加分
        length_bonus = min(len(think_content) / 500, 0.2)  # 最大0.2

        # 4. 计算总分，上限1.0
        score = overlap + conclusion_bonus + length_bonus
        return min(score, 1.0)

    def validate_batch(self, responses: List[Dict[str, Any]],
                       show_progress: bool = True) -> Dict[str, Any]:
        results = []
        iterator = tqdm(responses) if show_progress else responses
        for item in iterator:
            response = item.get("model_response", "")
            expected = item.get("expected_output", "")
            result = self.validate_response(response, expected)
            results.append({
                "prompt": item.get("prompt", ""),
                "validation": result,
                "raw_response": response
            })
        stats = self._compute_statistics(results)
        return {
            "individual_results": results,
            "statistics": stats
        }

    def _compute_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        total = len(results)
        if total == 0:
            return {}
        has_think = sum(1 for r in results if r["validation"].has_think_block)
        has_answer = sum(1 for r in results if r["validation"].has_answer_block)
        avg_think_length = sum(r["validation"].think_length for r in results) / total
        avg_answer_length = sum(r["validation"].answer_length for r in results) / total
        avg_reasoning_quality = sum(r["validation"].reasoning_quality for r in results) / total
        avg_consistency = sum(r["validation"].answer_consistency for r in results) / total
        all_issues = []
        for r in results:
            all_issues.extend(r["validation"].issues)
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        return {
            "total_samples": total,
            "has_think_block_ratio": has_think / total,
            "has_answer_block_ratio": has_answer / total,
            "avg_think_length": avg_think_length,
            "avg_answer_length": avg_answer_length,
            "avg_reasoning_quality": avg_reasoning_quality,
            "avg_answer_consistency": avg_consistency,
            "common_issues": issue_counts
        }

    def generate_report(self, validation_results: Dict[str, Any],
                        output_path: Optional[str] = None) -> str:
        stats = validation_results.get("statistics", {})
        report_lines = [
            "=" * 60,
            "思维链(CoT)验证报告",
            "=" * 60,
            "",
            "【整体统计】",
            f"总样本数: {stats.get('total_samples', 0)}",
            f"包含思维链比例: {stats.get('has_think_block_ratio', 0)*100:.1f}%",
            f"包含答案块比例: {stats.get('has_answer_block_ratio', 0)*100:.1f}%",
            f"平均思维链长度: {stats.get('avg_think_length', 0):.0f} 字符",
            f"平均答案长度: {stats.get('avg_answer_length', 0):.0f} 字符",
            f"平均推理质量: {stats.get('avg_reasoning_quality', 0)*100:.1f}%",
            f"平均答案一致性: {stats.get('avg_answer_consistency', 0)*100:.1f}%",
            "",
            "【常见问题】",
        ]
        issues = stats.get("common_issues", {})
        if issues:
            for issue, count in sorted(issues.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"  - {issue}: {count} 次")
        else:
            report_lines.append("  无")
        report_lines.extend([
            "",
            "【质量评估】",
        ])
        quality = stats.get('avg_reasoning_quality', 0)
        if quality >= 0.8:
            report_lines.append("推理质量: 优秀")
        elif quality >= 0.6:
            report_lines.append("推理质量: 良好")
        elif quality >= 0.4:
            report_lines.append("推理质量: 一般")
        else:
            report_lines.append("推理质量: 需改进")
        report_lines.append("=" * 60)
        report = "\n".join(report_lines)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"验证报告已保存: {output_path}")
        return report