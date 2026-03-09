#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChatSystem - 核心推理引擎
提供模型加载、推理生成、响应格式化等核心功能
供服务端各组件使用，客户端通过API调用
"""

import os
import asyncio
import queue
import sys
import re
import logging
import gc
import threading
from typing import Optional, Dict, Any, List, Generator, Tuple, AsyncGenerator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.utils.helpers import log_memory_usage
from src.utils.formatter import ResponseFormatter

logger = logging.getLogger(__name__)


class ChatSystem:
    """
    聊天系统核心类 - 服务端推理引擎

    负责:
    - 模型加载与管理（支持4-bit量化、CPU卸载）
    - 对话生成（流式/非流式）
    - 思维链解析与格式化
    - 多会话历史管理
    """

    def __init__(self, args):
        """
        初始化ChatSystem

        Args:
            args: 配置参数对象，包含:
                - model: 基础模型路径（必需）
                - adapter: LoRA适配器路径（可选）
                - max_seq_length: 最大序列长度（默认8192）
                - max_new_tokens: 最大生成token数（默认4096）
                - temperature: 采样温度（默认0.7）
                - top_p: nucleus sampling参数（默认0.9）
                - top_k: top-k sampling参数（默认50）
                - repetition_penalty: 重复惩罚（默认1.05）
                - think_chain: 是否启用思维链（默认True）
                - domain: 领域（medical/finance/legal/education/psychology，默认medical）
                - system: 自定义系统提示词（可选）
        """
        self.args = args

        # 模型组件
        self.model = None
        self.tokenizer = None
        self.model_type = "unknown"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 生成参数
        self.max_seq_length = getattr(args, 'max_seq_length', 8192)
        self.max_new_tokens = getattr(args, 'max_new_tokens', 4096)
        self.temperature = getattr(args, 'temperature', 0.7)
        self.top_p = getattr(args, 'top_p', 0.9)
        self.top_k = getattr(args, 'top_k', 50)
        self.repetition_penalty = getattr(args, 'repetition_penalty', 1.05)

        # 功能配置
        self.think_chain = getattr(args, 'think_chain', True)
        self.domain = getattr(args, 'domain', 'medical')

        # 会话管理: {session_id: [(role, content), ...]}
        self.sessions: Dict[str, List[Tuple[str, str]]] = {}

        # 响应格式化器
        self.formatter = ResponseFormatter()

        # 系统提示词
        self.system_prompt = self._build_system_prompt()

        logger.info(f"ChatSystem初始化: domain={self.domain}, "
                    f"think_chain={self.think_chain}, device={self.device}")

    def _build_system_prompt(self) -> str:
        """构建领域特定的系统提示词"""
        # 用户自定义提示词优先
        custom_system = getattr(self.args, 'system', None)
        if custom_system and isinstance(custom_system, str) and custom_system.strip():
            return custom_system.strip()

        # 领域默认提示词
        domain_prompts = {
            "medical": (
                "你是一个专业医疗助手。\n"
                "【重要】你的所有回答都必须遵守以下格式：\n"
                "1. 首先，在 <think> 标签内写出你的详细思考过程。\n"
                "2. 然后，在 </think> 标签后写出正式回答。\n"
                "3. 严禁输出自然语言思考而不使用 <think> 标签。\n"
                "示例：\n"
                "<think>\n"
                "用户询问高血压的定义，这是一种常见的慢性病...\n"
                "</think>\n"
                "高血压是指..."
            ),
            "finance": "你是一个专业的金融顾问助手。请提供准确、客观的金融分析和建议。",
            "legal": "你是一个专业的法律顾问助手。请提供基于法律法规的准确建议，但请注意这不构成正式法律意见。",
            "education": "你是一个专业的教育助手。请提供清晰、有帮助的教育指导和解答。",
            "psychology": "你是一个专业的心理咨询助手。请提供温暖、专业的心理支持，但请注意这不替代专业心理治疗。",
        }

        return domain_prompts.get(self.domain, domain_prompts["medical"])

    def load_model(self) -> bool:
        """
        加载模型和分词器

        Returns:
            bool: 是否成功加载
        """
        logger.info("=" * 80)
        logger.info("ChatSystem开始加载模型...")
        logger.info(f"基础模型: {self.args.model}")
        logger.info(f"适配器: {getattr(self.args, 'adapter', None)}")
        logger.info("=" * 80)

        # 清理显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        try:
            # 加载分词器
            logger.info("加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model,
                trust_remote_code=True,
                padding_side="right"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 检测模型类型
            self.model_type = self._detect_model_type(self.args.model)
            logger.info(f"检测到模型类型: {self.model_type}")

            # 配置量化
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            # 计算显存分配（T4 16GB优化）
            max_memory = None
            model_path_lower = self.args.model.lower()
            adapter_path = getattr(self.args, 'adapter', '') or ''
            if "14b" in model_path_lower or "14b" in adapter_path.lower():
                if torch.cuda.is_available():
                    total_mem = torch.cuda.get_device_properties(0).total_memory
                    # 保留2GB缓冲
                    gpu_mem = int((total_mem - 2 * 1024 ** 3) / (1024 ** 3))
                    max_memory = {0: f"{gpu_mem}GiB", "cpu": "32GiB"}
                    logger.info(f"14B模型启用CPU卸载: GPU={gpu_mem}GiB, CPU=32GiB")

            # 加载基础模型
            logger.info("加载基础模型...")
            load_kwargs = {
                "pretrained_model_name_or_path": self.args.model,
                "quantization_config": bnb_config,
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            }

            if max_memory:
                load_kwargs["device_map"] = "auto"
                load_kwargs["max_memory"] = max_memory
            else:
                load_kwargs["device_map"] = "auto"

            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

            # 加载适配器（如果提供）
            adapter_path = getattr(self.args, 'adapter', None)
            if adapter_path and os.path.exists(adapter_path):
                logger.info(f"加载LoRA适配器: {adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                self.model = self.model.merge_and_unload()
                logger.info("适配器已合并")

            # 设置评估模式
            self.model.eval()

            # 报告显存使用
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                logger.info(f"模型加载完成: 显存占用={allocated:.2f}GB, 预留={reserved:.2f}GB")

            logger.info("=" * 80)
            logger.info("ChatSystem模型加载成功")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None
            return False

    def _detect_model_type(self, model_path: str) -> str:
        """检测模型类型"""
        path_lower = model_path.lower()
        if "qwen3" in path_lower:
            if "14b" in path_lower:
                return "qwen3_14b"
            return "qwen3"
        elif "qwen" in path_lower:
            return "qwen"
        elif "deepseek" in path_lower:
            return "deepseek"
        return "unknown"

    def _build_prompt(self, user_input: str, session_id: Optional[str] = None) -> str:
        """构建完整的对话提示词"""
        messages = []

        # 系统提示
        messages.append({"role": "system", "content": self.system_prompt})

        # 历史消息（如果有会话ID）
        if session_id and session_id in self.sessions:
            history = self.sessions[session_id][-5:]  # 最近5轮
            for role, content in history:
                messages.append({"role": role, "content": content})

        # 当前用户输入
        messages.append({"role": "user", "content": user_input})

        # 应用聊天模板
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def generate_response(self, user_input: str, session_id: Optional[str] = None) -> Dict[str, str]:
        """
        生成完整响应（非流式）

        Args:
            user_input: 用户输入文本
            session_id: 会话ID（可选，用于保持上下文）

        Returns:
            dict: 包含 think_content, answer_content, full_response
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载，请先调用load_model()")

        prompt = self._build_prompt(user_input, session_id)

        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length
        ).to(self.device)

        # 生成参数
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
        }

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)

        # 解码输出
        full_response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # 解析思维链和答案
        think_content, answer_content = self._parse_response(full_response)

        # 更新会话历史
        if session_id:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            self.sessions[session_id].append(("user", user_input))
            self.sessions[session_id].append(("assistant", answer_content))

        return {
            "think_content": think_content,
            "answer_content": answer_content,
            "full_response": full_response,
        }

    async def stream_generate_response(self, user_input: str, session_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        流式生成响应
        """
        logger.info(f"stream_generate_response被调用: user_input={user_input[:50]}...")
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载，请先调用load_model()")

        prompt = self._build_prompt(user_input, session_id)
        logger.info(f"构建的prompt长度: {len(prompt)}")

        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length
        ).to(self.device)

        # 创建流式生成器 - 使用队列获取token
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # 生成参数
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
            "streamer": streamer,
        }

        # 在后台线程生成
        generation_thread = threading.Thread(
            target=self._generate_in_thread,
            kwargs=generation_kwargs
        )
        generation_thread.start()

        # 流式输出 - 关键修复：异步迭代 + 立即yield
        collected_tokens = []
        in_think_phase = True
        think_buffer = []
        answer_buffer = []
        think_sent = False

        try:
            # 使用streamer的text_queue进行异步迭代
            while generation_thread.is_alive() or not streamer.end:
                try:
                    # 非阻塞获取token，超时则让出控制权
                    text = streamer.text_queue.get(timeout=0.05)

                    # 关键修复：空字符串也yield，确保连接保持
                    if text is None:
                        break

                    collected_tokens.append(text)

                    # 检测思维链标签
                    partial_response = "".join(collected_tokens)

                    # 思维链阶段检测
                    if in_think_phase:
                        # 检查是否包含完整的think标签
                        if "<think>" in partial_response and not think_sent:
                            # 提取think内容
                            think_match = re.search(r'<think>(.*?)</think>', partial_response, re.DOTALL)
                            if think_match:
                                think_content = think_match.group(1).strip()
                                # 发送think内容，逐字符
                                for char in think_content:
                                    yield f"THINK:{char}"
                                    await asyncio.sleep(0)  # 关键：立即让出控制权
                                think_sent = True
                                in_think_phase = False
                            else:
                                # 累积中，发送非标签内容
                                clean_text = text.replace("<think>", "").replace("</think>", "")
                                if clean_text:
                                    yield f"THINK:{clean_text}"
                                    await asyncio.sleep(0)
                        elif "</think>" in text:
                            # 结束标签后的内容
                            in_think_phase = False
                            clean_text = text.replace("</think>", "").strip()
                            if clean_text:
                                yield clean_text
                                await asyncio.sleep(0)
                        else:
                            # 仍在think阶段
                            clean_text = text.replace("<think>", "").replace("</think>", "")
                            if clean_text:
                                yield f"THINK:{clean_text}"
                                await asyncio.sleep(0)
                    else:
                        # 答案阶段 - 直接输出
                        yield text
                        await asyncio.sleep(0)  # 关键：每个token都yield控制权

                except queue.Empty:
                    # 队列为空，让出控制权等待
                    await asyncio.sleep(0.01)
                    continue

            # 处理剩余内容（如果没有检测到标签）
            if not think_sent and collected_tokens:
                full_text = "".join(collected_tokens)
                # 尝试提取think内容
                think_match = re.search(r'<think>(.*?)</think>', full_text, re.DOTALL)
                if think_match:
                    think_content = think_match.group(1).strip()
                    for char in think_content:
                        yield f"THINK:{char}"
                        await asyncio.sleep(0)
                    # 发送剩余内容作为答案
                    answer_text = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL).strip()
                    for char in answer_text:
                        yield char
                        await asyncio.sleep(0)
                else:
                    # 没有think标签，全部作为答案
                    for char in full_text:
                        yield char
                        await asyncio.sleep(0)

            # 更新会话历史
            full_response = "".join(collected_tokens)
            self._update_session(user_input, full_response, session_id)

        except Exception as e:
            logger.error(f"流式生成出错: {e}")
            raise
        finally:
            # 确保线程结束
            generation_thread.join(timeout=2.0)

    def _generate_in_thread(self, **kwargs):
        """在独立线程中执行生成"""
        try:
            with torch.no_grad():
                self.model.generate(**kwargs)
        except Exception as e:
            logger.error(f"生成线程出错: {e}")

    def _update_session(self, user_input: str, full_response: str, session_id: Optional[str]):
        """更新会话历史"""
        if not session_id:
            return

        # 解析思维链和答案
        think_content, answer_content = self._parse_response(full_response)

        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append(("user", user_input))
        self.sessions[session_id].append(("assistant", answer_content))

        # 限制历史长度
        if len(self.sessions[session_id]) > 20:
            self.sessions[session_id] = self.sessions[session_id][-20:]

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """
        解析响应，分离思维链和答案

        Args:
            response: 完整响应文本

        Returns:
            tuple: (think_content, answer_content)
        """
        # 尝试匹配 <think>...</think> 格式
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, response, re.DOTALL)

        if think_match:
            think_content = think_match.group(1).strip()
            # 移除think标签后的内容作为答案
            answer_content = re.sub(think_pattern, '', response, flags=re.DOTALL).strip()
            return think_content, answer_content

        # 如果没有think标签，整个响应作为答案
        return "", response.strip()

    def clear_session(self, session_id: str):
        """清空指定会话的历史"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"会话已清空: {session_id}")

    def get_session_history(self, session_id: str) -> List[Tuple[str, str]]:
        """获取会话历史"""
        return self.sessions.get(session_id, [])

    def format_response(self, answer_content: str, think_content: str = "") -> Dict[str, Any]:
        """
        格式化响应输出

        Args:
            answer_content: 答案内容
            think_content: 思维链内容（可选）

        Returns:
            dict: 格式化后的响应
        """
        # 使用formatter进行领域特定格式化
        if self.domain == "medical":
            formatted = self.formatter.format_medical_response(answer_content, self.think_chain)
        elif self.domain == "finance":
            formatted = self.formatter.format_finance_response(answer_content, self.think_chain)
        elif self.domain == "legal":
            formatted = self.formatter.format_legal_response(answer_content, self.think_chain)
        elif self.domain == "education":
            formatted = self.formatter.format_education_response(answer_content, self.think_chain)
        elif self.domain == "psychology":
            formatted = self.formatter.format_psychology_response(answer_content, self.think_chain)
        else:
            formatted = self.formatter.format_general_response(answer_content, self.think_chain)

        return {
            "formatted": formatted,
            "raw_answer": answer_content,
            "think_content": think_content,
            "domain": self.domain,
        }

    def unload(self):
        """卸载模型，释放资源"""
        logger.info("卸载ChatSystem模型...")

        self.model = None
        self.tokenizer = None
        self.sessions.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("ChatSystem模型已卸载")