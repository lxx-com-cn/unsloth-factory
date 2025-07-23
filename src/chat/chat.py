#!/usr/bin/env python3
# src/chat/chat.py
import os
import re
import json
import logging
import warnings
import torch
import gc
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("chat.log")]
)
logger = logging.getLogger(__name__)

class ChatSystem:
    """逐字符流式对话系统"""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self.history = []

    def detect_model_type(self, model_path: str) -> str:
        try:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                model_name = config.get("_name_or_path", "").lower()
                if "qwen3" in model_name:
                    return "qwen3"
                elif "qwen2" in model_name:
                    return "qwen2"
                elif "deepseek" in model_name and "qwen" in model_name:
                    return "deepseek_r1_qwen"
        except:
            pass
        return "qwen2"

    def load_model(self):
        logger.info("正在加载模型: {}".format(self.args.model))
        self.model_type = self.detect_model_type(self.args.model)
        logger.info("检测到模型类型: {}".format(self.model_type))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model,
            trust_remote_code=True,
            padding_side="left"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            load_in_4bit=True,
            trust_remote_code=True
        )

        if hasattr(self.args, 'adapter') and self.args.adapter and os.path.exists(self.args.adapter):
            from peft import PeftModel  # ✅ 修复：原来是 peef
            self.model = PeftModel.from_pretrained(self.model, self.args.adapter, is_trainable=False)
            self.model = self.model.merge_and_unload()
            logger.info("适配器已合并")

        self.model.eval()
        logger.info("模型加载完成")

    def build_messages(self, user_input: str) -> List[Dict[str, str]]:
        messages = []
        system_prompt = getattr(self.args, 'system', '') or "你是一个专业的医疗助手，请用中文回答"
        messages.append({"role": "system", "content": system_prompt})

        if not getattr(self.args, 'no_context', False) and self.history:
            for role, content in self.history:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_input})
        return messages

    def stream_chat(self):
        """逐字符流式对话"""
        try:
            self.load_model()

            print("=" * 80)
            print("智能助手已启动 - 模型: {}".format(self.model_type))
            print("输入 'exit' 或 'quit' 退出")
            print("输入 'clear' 清空历史")
            print("=" * 80)

            if getattr(self.args, 'think_chain', False):
                print("已启用思维链分析模式")
                print("=" * 80)

            self.history = []

            while True:
                try:
                    user_input = input("\n<用户>: ").strip()

                    if user_input.lower() in ['exit', 'quit']:
                        break
                    elif user_input.lower() == 'clear':
                        self.history.clear()
                        print("已清空对话历史")
                        continue
                    elif not user_input:
                        continue

                    # 构建消息
                    messages = self.build_messages(user_input)
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    # 逐字符流式生成
                    self._stream_generate(prompt)

                except KeyboardInterrupt:
                    print("\n用户中断，退出...")
                    break
                except Exception as e:
                    logger.error("聊天错误: {}".format(e))
                    print("发生错误: {}".format(e))
                    continue

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _stream_generate(self, prompt: str):
        """逐字符流式生成 - 核心实现"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=getattr(self.args, 'max_seq_length', 8192)
        ).to(self.model.device)

        max_new_tokens = min(
            getattr(self.args, 'max_new_tokens', 2048),
            8192 - inputs['input_ids'].shape[1]
        )

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=30
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer
        }

        # 启动生成线程
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # 状态管理
        buffer = ""
        in_think = False
        think_complete = False

        if getattr(self.args, 'think_chain', False):
            print("\n<思维链分析>", end="", flush=True)

        for token in streamer:
            if not token:
                continue

            buffer += token

            # 实时检测思维链
            if getattr(self.args, 'think_chain', False):
                if not in_think and "<think>" in buffer:
                    in_think = True
                    buffer = buffer.split("<think>")[-1]
                    continue

                if in_think and "</think>" in buffer:
                    think_end = buffer.find("</think>")
                    think_content = buffer[:think_end]
                    if think_content.strip():
                        print("\n{}".format(think_content.strip()), flush=True)
                    buffer = buffer[think_end + 8:]
                    print("\n<正式回答>", end="", flush=True)
                    in_think = False
                    think_complete = True
                    continue

                if in_think:
                    # 逐字符输出思维链
                    print(token, end="", flush=True)
                elif think_complete or "<think>" not in buffer:
                    # 逐字符输出正式回答
                    print(token, end="", flush=True)
            else:
                # 直接逐字符输出
                print(token, end="", flush=True)

        # 收集完整响应用于历史
        full_response = buffer
        if not getattr(self.args, 'no_context', False):
            self.history.append(("user", "用户输入"))
            self.history.append(("assistant", full_response))
            if len(self.history) > 10:
                self.history = self.history[-10:]

    def start(self):
        """启动入口"""
        self.stream_chat()