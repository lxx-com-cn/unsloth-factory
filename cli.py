#!/usr/bin/env python3
# cli.py
import os
import sys
import argparse
import logging

# 添加环境变量设置解决 tokenizers 并行问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 导入核心模块
from src.trainers.trainer_factory import TrainerFactory
from src.utils.helpers import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Unsloth Training Framework", add_help=False)
    base_parser = argparse.ArgumentParser(add_help=False)

    # 通用参数
    base_parser.add_argument("--model", type=str, required=True, help="Base model path")
    base_parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
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
    base_parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint (path or 'auto')")
    base_parser.add_argument("--domain", type=str, default="medical", choices=["medical", "legal", "psychology", "exam"], help="Domain for fine-tuning")

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands", required=True)

    # SFT 训练命令
    sft_parser = subparsers.add_parser("sft", parents=[base_parser], help="Supervised Fine-Tuning")

    # DPO 训练命令
    dpo_parser = subparsers.add_parser("dpo", parents=[base_parser], help="Direct Preference Optimization")
    dpo_parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")

    # 合并模型命令
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA adapter into base model")
    merge_parser.add_argument("--model", type=str, required=True, help="Base model path")
    merge_parser.add_argument("--adapter", type=str, required=True, help="LoRA adapter path")
    merge_parser.add_argument("--output", type=str, required=True, help="Merged model output directory")
    merge_parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"], help="Model dtype")
    merge_parser.add_argument("--max_shard_size", type=str, default="2GB", help="Max shard size for safetensors")

    # 校验命令
    validate_parser = subparsers.add_parser("validate", help="Validate fine-tuned model")
    validate_parser.add_argument("--model", type=str, required=True)
    validate_parser.add_argument("--adapter", type=str, help="Adapter path")
    validate_parser.add_argument("--dataset", type=str, required=True)
    validate_parser.add_argument("--dataset_format", type=str, default="alpaca")
    validate_parser.add_argument("--max_samples", type=int, default=10)
    validate_parser.add_argument("--max_seq_length", type=int, default=8192)
    validate_parser.add_argument("--output_dir", type=str, default="validation_results", help="Output directory")
    validate_parser.add_argument("--domain", type=str, default="medical", choices=["medical", "legal", "psychology", "exam"], help="Domain for validation")

    # 评估命令
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model on benchmarks")
    evaluate_parser.add_argument("--task", type=str, required=True, choices=["ceval"], help="Evaluation task")
    evaluate_parser.add_argument("--model", type=str, required=True, help="Base model path")
    evaluate_parser.add_argument("--adapter", type=str, help="Adapter path (if fine-tuned)")
    evaluate_parser.add_argument("--task_dir", type=str, required=True, help="Path to task data")
    evaluate_parser.add_argument("--n_shot", type=int, default=5, help="Number of few-shot examples")
    evaluate_parser.add_argument("--lang", type=str, default="zh", choices=["zh", "en"], help="Language for evaluation")
    evaluate_parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length")
    evaluate_parser.add_argument("--save_dir", type=str, required=True, help="Output directory for results")
    evaluate_parser.add_argument("--domain", type=str, default="medical", choices=["medical", "legal", "psychology", "exam"], help="Domain for evaluation")

    # 聊天命令
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with the model")
    chat_parser.add_argument("--model", type=str, required=True, help="Base model path")
    chat_parser.add_argument("--adapter", type=str, help="Adapter path (if fine-tuned)")
    chat_parser.add_argument("--system", type=str, default="", help="System prompt for the chat")
    chat_parser.add_argument("--max_seq_length", type=int, default=8192, help="Max sequence length")
    chat_parser.add_argument("--think_chain", action="store_true", help="Enable chain-of-thought reasoning")
    chat_parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max new tokens for generation")
    chat_parser.add_argument("--no_context", action="store_true", help="Disable chat history context")
    chat_parser.add_argument("--domain", type=str, default="medical",
                             choices=["medical", "legal", "psychology", "exam"],
                             help="Domain for chat")

    # 添加帮助命令
    help_parser = subparsers.add_parser("help", help="Show help for commands")
    help_parser.add_argument("command", nargs="?", help="Command to show help for")

    args = parser.parse_args()

    # 处理帮助命令
    if args.command == "help":
        if args.command:
            if args.command in ["sft", "dpo", "validate", "evaluate", "chat", "merge"]:
                subparsers.choices[args.command].print_help()
            else:
                parser.print_help()
        else:
            parser.print_help()
        sys.exit(0)

    # 设置日志
    setup_logging()

    if args.command in ["sft", "dpo"]:
        trainer = TrainerFactory.create_trainer(method=args.command, args=args)
        success = trainer.train()
        if not success:
            logging.error(f"{args.command.upper()} training failed")
            sys.exit(1)

    elif args.command == "merge":
        from src.merger.model_merger import merge_models
        merge_models(args)

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

    elif args.command == "evaluate":
        from src.evaluators.evaluator import Evaluator
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