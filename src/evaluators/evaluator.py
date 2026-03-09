# src/evaluators/evaluator.py
import os
import json
import logging
import torch
import gc
from typing import Dict, Any, Optional
from peft import PeftModel
from src.evaluators.ceval_evaluator import evaluate_ceval
from src.core.model_factory import ModelFactory
from src.utils.helpers import log_memory_usage

logger = logging.getLogger(__name__)

class Evaluator:
    """模型评估器，支持多种评测任务"""
    
    def __init__(self, args):
        self.args = args
        self.task = args.task
    
    def load_model(self, adapter_path: Optional[str] = None):
        """加载模型（始终使用原始 tokenizer）"""
        logger.info(f"加载基础模型: {self.args.model}")
        model, tokenizer, _, _ = ModelFactory.create_model(
            model_path=self.args.model,
            max_seq_length=self.args.max_seq_length,
            adapter_path=adapter_path,
            use_unsloth=False
        )
        return model, tokenizer
    
    def evaluate(self) -> Dict[str, Any]:
        """执行指定任务的评估"""
        if self.task == "ceval":
            return self._evaluate_ceval()
        else:
            raise ValueError(f"不支持的任务: {self.task}")
    
    def _evaluate_ceval(self) -> Dict[str, Any]:
        """C-Eval 评估"""
        results = {}
        
        # 评估基础模型
        logger.info("=" * 80)
        logger.info("评估基础模型")
        logger.info("=" * 80)
        base_model, base_tokenizer = self.load_model()
        logger.info(f"加载基础模型后内存使用: {log_memory_usage()}")
        
        base_results = evaluate_ceval(
            model=base_model,
            tokenizer=base_tokenizer,
            task_dir=self.args.task_dir,
            n_shot=self.args.n_shot,
            lang=self.args.lang,
            save_dir=os.path.join(self.args.save_dir, "base_model"),
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            max_new_tokens=self.args.max_new_tokens,
        )
        results["base_model"] = base_results
        
        # 释放基础模型资源
        logger.info(f"评估基础模型后内存使用: {log_memory_usage()}")
        del base_model, base_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"释放基础模型后内存使用: {log_memory_usage()}")
        
        # 评估微调模型（如果提供了适配器）
        if self.args.adapter:
            logger.info("=" * 80)
            logger.info("评估微调模型")
            logger.info("=" * 80)
            ft_model, ft_tokenizer = self.load_model(self.args.adapter)
            logger.info(f"加载微调模型后内存使用: {log_memory_usage()}")
            
            ft_results = evaluate_ceval(
                model=ft_model,
                tokenizer=ft_tokenizer,
                task_dir=self.args.task_dir,
                n_shot=self.args.n_shot,
                lang=self.args.lang,
                save_dir=os.path.join(self.args.save_dir, "finetuned_model"),
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                max_new_tokens=self.args.max_new_tokens,
            )
            results["finetuned_model"] = ft_results
            
            # 释放微调模型资源
            logger.info(f"评估微调模型后内存使用: {log_memory_usage()}")
            del ft_model, ft_tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"释放微调模型后内存使用: {log_memory_usage()}")
        
        # 保存完整评估结果并生成对比报告
        self._save_results(results)
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """保存评估结果并生成对比报告"""
        os.makedirs(self.args.save_dir, exist_ok=True)
        
        # 保存详细结果
        result_path = os.path.join(self.args.save_dir, "evaluation_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 生成对比报告
        comparison_path = os.path.join(self.args.save_dir, "comparison_report.txt")
        with open(comparison_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("模型评估对比报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"评估任务: {self.args.task}\n")
            f.write(f"基础模型: {self.args.model}\n")
            if self.args.adapter:
                f.write(f"适配器路径: {self.args.adapter}\n")
            f.write(f"评估数据集: {self.args.task_dir}\n")
            f.write(f"Few-shot数量: {self.args.n_shot}\n")
            f.write(f"语言: {self.args.lang}\n")
            f.write(f"最大序列长度: {self.args.max_seq_length}\n\n")
            
            # 基础模型结果
            f.write("-" * 80 + "\n")
            f.write("基础模型评估结果:\n")
            f.write("-" * 80 + "\n")
            base_results = results["base_model"]
            f.write(f"平均准确率: {base_results.get('average', 0):.4f}\n")
            if "stem_average" in base_results:
                f.write(f"STEM平均: {base_results.get('stem_average', 0):.4f}\n")
            if "performance_rating" in base_results:
                f.write(f"性能评级: {base_results.get('performance_rating', 'N/A')}\n")
            
            # 微调模型结果（如果有）
            if "finetuned_model" in results:
                f.write("\n" + "-" * 80 + "\n")
                f.write("微调模型评估结果:\n")
                f.write("-" * 80 + "\n")
                ft_results = results["finetuned_model"]
                f.write(f"平均准确率: {ft_results.get('average', 0):.4f}\n")
                if "stem_average" in ft_results:
                    f.write(f"STEM平均: {ft_results.get('stem_average', 0):.4f}\n")
                if "performance_rating" in ft_results:
                    f.write(f"性能评级: {ft_results.get('performance_rating', 'N/A')}\n")
                
                # 比较结果
                f.write("\n" + "-" * 80 + "\n")
                f.write("性能对比:\n")
                base_acc = base_results.get('average', 0)
                ft_acc = ft_results.get('average', 0)
                f.write(f"平均准确率变化: {ft_acc - base_acc:.4f} ({'提升' if ft_acc > base_acc else '下降'})\n")
                
                if "stem_average" in base_results and "stem_average" in ft_results:
                    base_stem = base_results.get('stem_average', 0)
                    ft_stem = ft_results.get('stem_average', 0)
                    f.write(f"STEM平均变化: {ft_stem - base_stem:.4f} ({'提升' if ft_stem > base_stem else '下降'})\n")
            
            # 详细结果路径
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"完整评估结果已保存至: {result_path}\n")
            f.write(f"对比报告已保存至: {comparison_path}\n")
        
        logger.info(f"评估结果保存至: {result_path}")
        logger.info(f"对比报告保存至: {comparison_path}")
        
        # 打印简要对比
        self._print_comparison(results)
    
    def _print_comparison(self, results: Dict[str, Any]):
        """打印对比报告"""
        print("\n" + "="*80)
        print("评估对比摘要:")
        print("="*80)
        
        base_results = results["base_model"]
        print(f"基础模型平均准确率: {base_results.get('average', 0):.4f}")
        if "stem_average" in base_results:
            print(f"基础模型STEM平均: {base_results.get('stem_average', 0):.4f}")
        if "performance_rating" in base_results:
            print(f"基础模型性能评级: {base_results.get('performance_rating', 'N/A')}")
        
        if "finetuned_model" in results:
            ft_results = results["finetuned_model"]
            print("\n" + "-"*80)
            print(f"微调模型平均准确率: {ft_results.get('average', 0):.4f}")
            if "stem_average" in ft_results:
                print(f"微调模型STEM平均: {ft_results.get('stem_average', 0):.4f}")
            if "performance_rating" in ft_results:
                print(f"微调模型性能评级: {ft_results.get('performance_rating', 'N/A')}")
            
            # 比较结果
            print("\n" + "-"*80)
            print("性能对比:")
            base_acc = base_results.get('average', 0)
            ft_acc = ft_results.get('average', 0)
            print(f"平均准确率变化: {ft_acc - base_acc:.4f} ({'提升' if ft_acc > base_acc else '下降'})")
            
            if "stem_average" in base_results and "stem_average" in ft_results:
                base_stem = base_results.get('stem_average', 0)
                ft_stem = ft_results.get('stem_average', 0)
                print(f"STEM平均变化: {ft_stem - base_stem:.4f} ({'提升' if ft_stem > base_stem else '下降'})")
        
        print("="*80)
        print(f"完整评估结果和对比报告已保存至: {self.args.save_dir}")