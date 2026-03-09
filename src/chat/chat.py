#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI控制台客户端（轻薄版）
仅提供命令行交互界面，所有推理通过REST API调用
"""

import os
import sys
import json
import time
import uuid
import logging
import argparse
from typing import Optional, Dict, Any
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class ChatCLI:
    """
    命令行聊天客户端
    通过REST API与推理服务通信，本地不加载模型
    """

    def __init__(self, api_url: str = "http://localhost:12001", domain: str = "medical"):
        self.api_url = api_url.rstrip('/')
        self.domain = domain
        self.session_id: Optional[str] = None
        self.user_id = f"cli_{uuid.uuid4().hex[:8]}"
        self.history: list = []

        self._check_service()

    def _check_service(self):
        """检查推理服务是否可用"""
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=5)
            data = resp.json()

            if data.get("model_loaded"):
                logger.info(f"推理服务已连接: {self.api_url}")
                logger.info(f"模型状态: 已加载")
                logger.info(f"GPU内存: {data.get('gpu_memory', {})}")
            else:
                logger.warning("模型未加载，服务可能正在启动...")

        except requests.exceptions.ConnectionError:
            logger.error(f"无法连接到推理服务: {self.api_url}")
            logger.error("请确保服务已启动: ./scripts/start_server.sh")
            logger.error(f"如果服务在其他机器或端口，请使用 --api-url 指定，例如:")
            logger.error(f"  python -m src.chat.chat --api-url http://172.16.0.95:12001")
            sys.exit(1)
        except Exception as e:
            logger.error(f"服务检查失败: {e}")
            sys.exit(1)

    def _send_request(self, message: str, stream: bool = True, think_chain: bool = True) -> Dict[str, Any]:
        """
        发送聊天请求到API服务 - 修复版：颜色区分思维链和正文
        """
        messages = []
        for role, content in self.history[-10:]:
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        payload = {
            "messages": messages,
            "model": "qwen3-14b-medical",
            "stream": stream,
            "think_chain": think_chain,
            "domain": self.domain,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "max_tokens": 4096,
        }

        try:
            if stream:
                resp = requests.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=payload,
                    stream=True,
                    timeout=300,
                )
                resp.raise_for_status()

                think_content = []
                answer_content = []

                # ANSI颜色代码定义 - 关键修复：区分颜色
                # 思维链：深灰色（暗淡）
                COLOR_THINK = "\033[38;5;244m"  # 灰色
                # 正文：亮白色加粗
                COLOR_ANSWER = "\033[97m\033[1m"  # 亮白加粗
                # 重置
                COLOR_RESET = "\033[0m"
                # 标签颜色
                COLOR_LABEL = "\033[36m"  # 青色

                # 打印助手标签
                print(f"\n{COLOR_LABEL}助手{COLOR_RESET}: ", end='', flush=True)

                # 状态标记
                in_think = False
                think_started = False
                answer_started = False

                for line in resp.iter_lines():
                    if not line:
                        continue

                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]

                        if data_str == '[DONE]':
                            break

                        try:
                            data = json.loads(data_str)
                            delta = data.get('choices', [{}])[0].get('delta', {})

                            # 处理思维链token
                            if 'think_token' in delta:
                                if not think_started:
                                    # 开始思维链，打印标签并设置颜色
                                    print(f"\n{COLOR_THINK}[思考过程] ", end='', flush=True)
                                    think_started = True
                                    in_think = True
                                think_content.append(delta['think_token'])
                                print(delta['think_token'], end='', flush=True)

                            # 处理正文token
                            elif 'content' in delta:
                                if in_think:
                                    # 结束思维链，重置颜色并换行
                                    print(f"{COLOR_RESET}")
                                    in_think = False
                                if not answer_started:
                                    # 开始正文，打印标签并设置亮白色
                                    print(f"\n{COLOR_ANSWER}[回答] ", end='', flush=True)
                                    answer_started = True
                                answer_content.append(delta['content'])
                                print(delta['content'], end='', flush=True)

                        except json.JSONDecodeError:
                            continue

                # 确保重置颜色并换行
                print(f"{COLOR_RESET}\n")

                return {
                    "think_content": "".join(think_content),
                    "answer_content": "".join(answer_content),
                    "finish_reason": "stop",
                }

            else:
                # 非流式处理
                resp = requests.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=payload,
                    timeout=300,
                )
                resp.raise_for_status()

                data = resp.json()
                choice = data.get('choices', [{}])[0]
                message = choice.get('message', {})

                # 颜色区分非流式输出
                think = data.get('think_content', '')
                answer = message.get('content', '')

                COLOR_THINK = "\033[38;5;244m"
                COLOR_ANSWER = "\033[97m\033[1m"
                COLOR_RESET = "\033[0m"
                COLOR_LABEL = "\033[36m"

                print(f"\n{COLOR_LABEL}助手{COLOR_RESET}: ")
                if think:
                    print(f"{COLOR_THINK}[思考过程] {think}{COLOR_RESET}\n")
                print(f"{COLOR_ANSWER}[回答] {answer}{COLOR_RESET}")

                return {
                    "think_content": think,
                    "answer_content": answer,
                    "finish_reason": choice.get('finish_reason', 'stop'),
                }

        except requests.exceptions.Timeout:
            logger.error("请求超时，服务可能繁忙")
            return {"error": "timeout", "answer_content": "【请求超时】"}
        except Exception as e:
            logger.error(f"请求失败: {e}")
            return {"error": str(e), "answer_content": f"【请求失败: {e}】"}

    def chat_loop(self):
        """主聊天循环"""
        print("\n" + "=" * 80)
        print("  智能医疗助手 - 控制台客户端")
        print("=" * 80)
        print(f"  API地址: {self.api_url}")
        print(f"  用户ID: {self.user_id}")
        print(f"  领域: {self.domain}")
        print("=" * 80)
        print("  命令:")
        print("    exit / quit  - 退出")
        print("    clear        - 清空历史")
        print("    new          - 新建会话")
        print("    stats        - 查看服务状态")
        print("    think on/off - 开关思维链")
        print("=" * 80 + "\n")

        think_chain = True

        while True:
            try:
                # 使用颜色区分用户输入
                COLOR_USER = "\033[94m"  # 蓝色
                COLOR_RESET = "\033[0m"
                user_input = input(f"\n{COLOR_USER}用户{COLOR_RESET}: ").strip()

                if not user_input:
                    continue

                cmd = user_input.lower()
                if cmd in ['exit', 'quit', 'q']:
                    print("再见！")
                    break
                elif cmd == 'clear':
                    self.history = []
                    print("历史已清空")
                    continue
                elif cmd == 'new':
                    self.session_id = f"session_{uuid.uuid4().hex[:8]}"
                    self.history = []
                    print(f"新会话已创建: {self.session_id[:16]}...")
                    continue
                elif cmd == 'stats':
                    self._show_stats()
                    continue
                elif cmd == 'think on':
                    think_chain = True
                    print("思维链: 开启")
                    continue
                elif cmd == 'think off':
                    think_chain = False
                    print("思维链: 关闭")
                    continue

                start_time = time.time()
                result = self._send_request(user_input, stream=True, think_chain=think_chain)
                elapsed = time.time() - start_time

                if "error" not in result:
                    self.history.append(("user", user_input))
                    self.history.append(("assistant", result["answer_content"]))

                # 使用暗淡颜色显示统计信息
                COLOR_DIM = "\033[90m"
                COLOR_RESET = "\033[0m"
                print(f"{COLOR_DIM}[{elapsed:.2f}s | think: {len(result.get('think_content', ''))} chars]{COLOR_RESET}")

            except KeyboardInterrupt:
                print("\n\n用户中断，退出...")
                break
            except Exception as e:
                logger.error(f"错误: {e}")
                continue

    def _show_stats(self):
        """显示服务状态"""
        try:
            resp = requests.get(f"{self.api_url}/stats", timeout=5)
            data = resp.json()

            print("\n" + "-" * 40)
            print("服务状态:")
            print("-" * 40)

            model_status = data.get('model', {})
            print(f"模型加载: {'是' if model_status.get('model_loaded') else '否'}")
            print(f"GPU显存: {model_status.get('gpu_memory', {})}")

            queue_status = data.get('queue', {})
            if queue_status:
                print(f"队列长度: {queue_status.get('queue_size', 'N/A')}")
                print(f"活跃请求: {queue_status.get('processing_count', 'N/A')}")

            print("-" * 40 + "\n")

        except Exception as e:
            print(f"获取状态失败: {e}")


def main():
    """CLI入口"""
    parser = argparse.ArgumentParser(description="Medical LLM CLI客户端")
    parser.add_argument("--api-url", type=str, default="http://localhost:12001",
                        help="推理服务API地址 (默认: http://localhost:12001)")
    parser.add_argument("--domain", type=str, default="medical",
                        choices=["medical", "finance", "legal", "education", "psychology"],
                        help="领域")

    args = parser.parse_args()

    cli = ChatCLI(api_url=args.api_url, domain=args.domain)
    cli.chat_loop()


if __name__ == "__main__":
    main()