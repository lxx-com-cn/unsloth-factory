#!/usr/bin/env python3
# cli.py - 完整修复版（移除 awq_exporter 导入）
import os
import sys
import argparse
import logging
import json

logger = logging.getLogger(__name__)

import unsloth

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.trainers.trainer_factory import TrainerFactory
from src.utils.helpers import setup_logging
from src.merger.export import export_model
# from src.merger.awq_exporter import export_awq_cli  # 已删除，注释掉

def main():
    parser = argparse.ArgumentParser(description="Unsloth Training Framework", add_help=False)
    base_parser = argparse.ArgumentParser(add_help=False)

    base_parser.add_argument("--model", type=str, required=True, help="Base model path")
    base_parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (supports multiple files separated by comma)")
    base_parser.add_argument("--dataset_format", type=str, default="alpaca", help="Dataset format")
    base_parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    base_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    base_parser.add_argument("--max_seq_length", type=int, default=8192, help="Max sequence length")
    base_parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    base_parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    base_parser.add_argument("--learning_rate", type=float, default=2e-6, help="Learning rate")
    base_parser.add_argument("--data_limit", type=int, default=None, help="Limit number of samples")
    base_parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    base_parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    base_parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    base_parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
    base_parser.add_argument("--save_steps", type=int, default=500, help="Save steps interval")
    base_parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps interval")
    base_parser.add_argument("--no_packing", action="store_true", help="Disable packing")
    base_parser.add_argument("--dataloader_workers", type=int, default=4, help="Number of dataloader workers")
    base_parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint (path, 'auto', or experiment ID)")
    base_parser.add_argument("--domain", type=str, default="medical", choices=["medical", "finance", "legal", "education", "psychology"], help="Domain for fine-tuning")
    base_parser.add_argument("--mixing_strategy", type=str, default="concat", choices=["concat", "interleave", "weighted"], help="Dataset mixing strategy")
    base_parser.add_argument("--dataset_weights", type=str, default=None, help="Dataset weights for weighted mixing (comma separated)")

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands", required=True)

    sft_parser = subparsers.add_parser("sft", parents=[base_parser], help="Supervised Fine-Tuning")
    sft_parser.add_argument("--experiments_root", type=str, default="output/experiments", help="Root directory for experiments")
    sft_parser.add_argument("--experiment_id", type=str, default=None, help="Specific experiment ID to use")

    dpo_parser = subparsers.add_parser("dpo", parents=[base_parser], help="Direct Preference Optimization")
    dpo_parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    dpo_parser.add_argument("--experiments_root", type=str, default="output/experiments", help="Root directory for experiments")

    merge_parser = subparsers.add_parser("merge", help="Merge LoRA adapter into base model")
    merge_parser.add_argument("--model", type=str, required=True, help="Base model path")
    merge_parser.add_argument("--adapter", type=str, required=True, help="LoRA adapter path")
    merge_parser.add_argument("--output", type=str, required=True, help="Merged model output directory")
    merge_parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"], help="Model dtype")
    merge_parser.add_argument("--max_shard_size", type=str, default="5GB", help="Max shard size for safetensors (e.g., 2GB, 5GB, 100GB for single file)")

    export_parser = subparsers.add_parser("export", help="Export HF model to GGUF and optionally quantize using llama.cpp")
    export_parser.add_argument("--llama_cpp", type=str, required=True, help="Path to the llama.cpp root directory")
    export_parser.add_argument("--model", type=str, required=True, help="Path to the merged Hugging Face model directory")
    export_parser.add_argument("--gguf", type=str, help="Output path for the converted GGUF file (e.g., ./model.gguf)")
    export_parser.add_argument("--quant_method", type=str, help="Quantization method for llama-quantize (e.g., q4_0, q5_k_m)")
    export_parser.add_argument("--quant_gguf", type=str, help="Output path for the quantized GGUF file (e.g., ./model-q4_0.gguf)")

    # 移除 awq 命令，改用 convert_quant.py 独立脚本
    # awq_parser = subparsers.add_parser("awq", help="Export and quantize model to AWQ format")
    # ... awq 参数定义 ...

    # 添加 convert_awq 命令（调用独立脚本）
    convert_awq_parser = subparsers.add_parser("convert_awq", help="Convert merged model to AWQ/GPTQ using convert_quant.py (run in vllm env)")
    convert_awq_parser.add_argument("--model_path", type=str, default=None, help="Merged model path")
    convert_awq_parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    convert_awq_parser.add_argument("--quant_scheme", type=str, default="GPTQ", choices=["GPTQ", "AWQ"])
    convert_awq_parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    convert_awq_parser.add_argument("--calib_samples", type=int, default=128)

    validate_parser = subparsers.add_parser("validate", help="Validate fine-tuned model")
    validate_parser.add_argument("--model", type=str, required=True)
    validate_parser.add_argument("--adapter", type=str, help="Adapter path")
    validate_parser.add_argument("--dataset", type=str, required=True)
    validate_parser.add_argument("--dataset_format", type=str, default="alpaca")
    validate_parser.add_argument("--max_samples", type=int, default=10)
    validate_parser.add_argument("--max_seq_length", type=int, default=8192)
    validate_parser.add_argument("--output_dir", type=str, default="validation_results", help="Output directory")
    validate_parser.add_argument("--domain", type=str, default="medical", choices=["medical", "finance", "legal", "education", "psychology"], help="Domain for validation")
    validate_parser.add_argument("--advanced", action="store_true", help="Enable advanced multi-dimensional validation")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model on benchmarks")
    evaluate_parser.add_argument("--task", type=str, required=True, choices=["ceval"], help="Evaluation task")
    evaluate_parser.add_argument("--model", type=str, required=True, help="Base model path")
    evaluate_parser.add_argument("--adapter", type=str, help="Adapter path (if fine-tuned)")
    evaluate_parser.add_argument("--task_dir", type=str, required=True, help="Path to task data")
    evaluate_parser.add_argument("--n_shot", type=int, default=5, help="Number of few-shot examples")
    evaluate_parser.add_argument("--lang", type=str, default="zh", choices=["zh", "en"], help="Language for evaluation")
    evaluate_parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length")
    evaluate_parser.add_argument("--save_dir", type=str, default=None, help="Output directory for results (default: auto-detect)")
    evaluate_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    evaluate_parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    evaluate_parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    evaluate_parser.add_argument("--max_new_tokens", type=int, default=10, help="Maximum new tokens to generate")
    evaluate_parser.add_argument("--domain", type=str, default="medical", choices=["medical", "finance", "legal", "education", "psychology"], help="Domain for evaluation")

    chat_parser = subparsers.add_parser("chat", help="Interactive chat with the model")
    chat_parser.add_argument("--model", type=str, required=True, help="Base model path")
    chat_parser.add_argument("--adapter", type=str, help="Adapter path (if fine-tuned)")
    chat_parser.add_argument("--system", type=str, default="", help="System prompt for the chat")
    chat_parser.add_argument("--max_seq_length", type=int, default=8192, help="Max sequence length")
    chat_parser.add_argument("--think_chain", action="store_true", default=True, help="Enable chain-of-thought reasoning (default: True)")
    chat_parser.add_argument("--no_think_chain", action="store_false", dest="think_chain", help="Disable chain-of-thought reasoning")
    chat_parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max new tokens for generation")
    chat_parser.add_argument("--no_context", action="store_true", help="Disable chat history context")
    chat_parser.add_argument("--domain", type=str, default="medical",
                             choices=["medical", "finance", "legal", "education", "psychology"],
                             help="Domain for chat")

    exp_parser = subparsers.add_parser("experiments", help="Manage experiments")
    exp_parser.add_argument("--action", type=str, required=True, choices=["list", "cleanup"], help="Action to perform")
    exp_parser.add_argument("--task", type=str, default=None, help="Filter by task")
    exp_parser.add_argument("--domain", type=str, default=None, help="Filter by domain")
    exp_parser.add_argument("--keep_count", type=int, default=10, help="Number of experiments to keep when cleaning")

    help_parser = subparsers.add_parser("help", help="Show help for commands")
    help_parser.add_argument("command", nargs="?", help="Command to show help for")

    args = parser.parse_args()

    if args.command == "help":
        if args.command:
            if args.command in ["sft", "dpo", "validate", "evaluate", "chat", "merge", "export", "convert_awq", "experiments"]:
                subparsers.choices[args.command].print_help()
            else:
                parser.print_help()
        else:
            parser.print_help()
        sys.exit(0)

    setup_logging()

    if args.command in ["sft", "dpo"]:
        from src.core.experiment_manager import get_experiment_manager
        
        exp_manager = get_experiment_manager(getattr(args, 'experiments_root', 'output/experiments'))
        
        weights = None
        if args.dataset_weights:
            try:
                weights = [float(w) for w in args.dataset_weights.split(',')]
            except ValueError:
                logger.warning("数据集权重格式错误，使用默认权重")
        
        args.mixing_strategy = getattr(args, 'mixing_strategy', 'concat')
        args.weights = weights
        
        exp_id = exp_manager.create_experiment(
            task=args.command,
            domain=args.domain,
            config=args,
            resume=args.resume
        )
        
        if exp_id:
            args.output_dir = exp_manager.get_experiment_path(exp_id)
            args.experiment_id = exp_id
            
            if args.resume:
                checkpoint_path = exp_manager.get_latest_checkpoint(exp_id)
                if checkpoint_path:
                    args.resume_checkpoint = checkpoint_path
                    logger.info(f"将从检查点恢复: {checkpoint_path}")
        
        trainer = TrainerFactory.create_trainer(method=args.command, args=args)
        success = trainer.train()
        
        if success:
            exp_manager.update_experiment_status("completed")
        else:
            exp_manager.update_experiment_status("failed")
        
        if not success:
            logging.error(f"{args.command.upper()} training failed")
            sys.exit(1)

    elif args.command == "merge":
        from src.merger.model_merger import merge_models
        merge_models(args)

    elif args.command == "export":
        success = export_model(args)
        if not success:
            logging.error("Export process failed")
            sys.exit(1)
        else:
            logging.info("Export process completed successfully")

    elif args.command == "convert_awq":
        # 调用独立脚本，需要在 vllm 环境中运行
        import subprocess
        cmd = [
            "python", "src/merger/convert_quant.py",
            "--model_path", args.model_path or "output/sft-qwen3-14b/merged_model",
            "--output_dir", args.output_dir or "output/sft-qwen3-14b/gptq4_model",
            "--quant_scheme", args.quant_scheme,
            "--bits", str(args.bits),
            "--calib_samples", str(args.calib_samples),
        ]
        logger.info(f"执行: {' '.join(cmd)}")
        logger.info("注意：需要在 vllm 环境中运行此命令")
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    elif args.command == "experiments":
        from src.core.experiment_manager import get_experiment_manager
        
        exp_manager = get_experiment_manager("output/experiments")
        
        if args.action == "list":
            experiments = exp_manager.list_experiments(
                task=args.task,
                domain=args.domain
            )
            print("\n" + "="*80)
            print(f"{'Experiment ID':<40} {'Task':<10} {'Domain':<10} {'Status':<12} {'Adapter':<8}")
            print("="*80)
            for exp in experiments:
                adapter_mark = "Yes" if exp.get("has_adapter") else "No"
                print(f"{exp['experiment_id']:<40} {exp.get('task', 'N/A'):<10} "
                      f"{exp.get('domain', 'N/A'):<10} {exp.get('status', 'N/A'):<12} {adapter_mark:<8}")
            print("="*80)
            print(f"Total: {len(experiments)} experiments")
            
        elif args.action == "cleanup":
            exp_manager.cleanup_old_experiments(keep_count=args.keep_count)
            print(f"Cleaned up old experiments, keeping last {args.keep_count}")

    elif args.command == "validate":
        from src.validators.validator import ModelValidator
        validator = ModelValidator(args)
        results = validator.validate()
        
        print("\n" + "="*50)
        print("验证摘要:")
        print(f"总样本数: {results['summary']['total_samples']}")
        if "base_model" in results['summary']:
            print(f"基础模型平均响应长度: {results['summary']['base_model']['avg_response_length']:.0f} 字符")
            print(f"基础模型领域警告数: {results['summary']['base_model']['domain_issues']}/{results['summary']['total_samples']}")
        if "adapter_model" in results['summary']:
            print(f"微调模型平均响应长度: {results['summary']['adapter_model']['avg_response_length']:.0f} 字符")
            print(f"微调模型领域警告数: {results['summary']['adapter_model']['domain_issues']}/{results['summary']['total_samples']}")
        print("="*50)
        print(f"完整验证结果已保存至: {results['output_path']}")

        if args.advanced:
            try:
                from src.validators.advanced_validator import AdvancedValidator
                logger.info("开始高级多维度验证...")
                
                output_dir = os.path.dirname(results.get("output_path", args.output_dir))
                os.makedirs(output_dir, exist_ok=True)
                
                target = "adapter_model" if args.adapter else "base_model"
                if target not in results.get('detailed_results', {}):
                    logger.warning(f"未找到 {target} 的详细结果，跳过高级验证")
                else:
                    samples = []
                    for item in results['detailed_results'][target]:
                        samples.append({
                            "prompt": item.get("prompt", ""),
                            "expected_output": item.get("expected_output", ""),
                            "model_response": item.get("model_response", "")
                        })
                    
                    adv_validator = AdvancedValidator(domain=args.domain)
                    adv_results = adv_validator.evaluate(samples)
                    
                    adv_json_path = os.path.join(output_dir, "advanced_results.json")
                    with open(adv_json_path, 'w', encoding='utf-8') as f:
                        json.dump(adv_results, f, ensure_ascii=False, indent=2)
                    
                    adv_report_path = os.path.join(output_dir, "advanced_report.txt")
                    report = adv_validator.generate_report(adv_results, adv_report_path)
                    
                    print("\n" + report)
                    logger.info(f"高级验证结果已保存至: {adv_json_path}")
                    logger.info(f"高级验证报告已保存至: {adv_report_path}")
            except ImportError as e:
                logger.error(f"无法导入 AdvancedValidator: {e}")
            except Exception as e:
                logger.error(f"高级验证失败: {e}")

    elif args.command == "evaluate":
        from src.evaluators.evaluator import Evaluator
        
        if args.save_dir is None:
            if args.adapter and "experiments" in args.adapter:
                parts = args.adapter.split("experiments/")
                if len(parts) > 1:
                    exp_part = parts[1].split("/")[0]
                    args.save_dir = os.path.join("output/experiments", exp_part, "evaluation")
                else:
                    args.save_dir = "output/evaluation_results"
            else:
                args.save_dir = "output/evaluation_results"
        os.makedirs(args.save_dir, exist_ok=True)
        
        evaluator = Evaluator(args)
        results = evaluator.evaluate()

    elif args.command == "chat":
        from src.chat.chat import ChatSystem
        chat_system = ChatSystem(args)
        chat_system.start()

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()