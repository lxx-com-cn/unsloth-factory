# src/chat/web_ui.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web UI客户端（流式输出最终版）
提供浏览器界面，通过REST API调用推理服务
关键修复：使用标准库asyncio和threading实现可靠的流式代理
"""

import os
import json
import logging
import asyncio
import threading
import queue as std_queue
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# 使用标准库http.client进行流式请求
import http.client
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 获取项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

INDEX_HTML = os.path.join(TEMPLATES_DIR, "index.html")

# HTML模板（与之前相同，省略以节省空间，实际使用时应包含完整内容）
INDEX_HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能医疗助手</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .chat-header h1 { font-size: 24px; margin-bottom: 5px; }
        .chat-header p { font-size: 14px; opacity: 0.9; }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 15px;
            max-width: 80%;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user { margin-left: auto; }
        .message.assistant { margin-right: auto; }
        .message-content {
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.6;
            word-wrap: break-word;
        }
        .message.user .message-content {
            background: #667eea;
            color: white;
            border-bottom-right-radius: 4px;
        }
        .message.assistant .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .think-content {
            background: #f5f5f5 !important;
            color: #666 !important;
            font-size: 13px;
            border-left: 3px solid #667eea;
            margin-bottom: 8px;
            font-style: italic;
        }
        .answer-content {
            background: white !important;
            color: #000 !important;
            font-weight: 500;
        }
        .chat-input {
            display: flex;
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        .chat-input input {
            flex: 1;
            padding: 12px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        .chat-input input:focus { border-color: #667eea; }
        .chat-input button {
            margin-left: 10px;
            padding: 12px 24px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .chat-input button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .chat-input button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .typing-indicator {
            display: none;
            padding: 12px 16px;
            background: white;
            border-radius: 18px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        .typing-indicator.active { display: block; }
        .dots { display: flex; gap: 4px; }
        .dot {
            width: 8px;
            height: 8px;
            background: #999;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }
        .dot:nth-child(1) { animation-delay: -0.32s; }
        .dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        .settings {
            padding: 10px 20px;
            background: #f8f9fa;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 15px;
            align-items: center;
            font-size: 14px;
        }
        .settings label { display: flex; align-items: center; gap: 5px; cursor: pointer; }
        .settings input[type="checkbox"] { cursor: pointer; }
        .error-message {
            background: #fee !important;
            color: #c33 !important;
            border-left-color: #c33 !important;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>智能医疗助手</h1>
            <p>基于 Qwen3-14B 医学大模型 | 支持思维链推理</p>
        </div>

        <div class="chat-messages" id="messages">
            <div class="message assistant">
                <div class="message-content">
                    您好！我是智能医疗助手，可以为您解答医学健康问题。<br>
                    我会先进行专业分析思考，然后给出详细建议。
                </div>
            </div>
        </div>

        <div class="typing-indicator" id="typing">
            <div class="dots">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>

        <div class="settings">
            <label>
                <input type="checkbox" id="thinkChain" checked>
                <span>显示思维链</span>
            </label>
            <label>
                <input type="checkbox" id="streamOutput" checked>
                <span>流式输出</span>
            </label>
            <span id="status" style="margin-left: auto; color: #666;">就绪</span>
        </div>

        <div class="chat-input">
            <input type="text" id="userInput" placeholder="请输入您的医学问题..." autocomplete="off">
            <button id="sendBtn" onclick="sendMessage()">发送</button>
        </div>
    </div>

    <script>
        const messagesDiv = document.getElementById('messages');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const typing = document.getElementById('typing');
        const status = document.getElementById('status');
        const thinkChainCheck = document.getElementById('thinkChain');
        const streamCheck = document.getElementById('streamOutput');

        let sessionId = generateId();
        let isProcessing = false;

        function generateId() {
            return 'session_' + Math.random().toString(36).substr(2, 9);
        }

        function addMessage(role, content, isThink = false, isAnswer = false) {
            const div = document.createElement('div');
            div.className = 'message ' + role;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            if (isThink) contentDiv.classList.add('think-content');
            if (isAnswer) contentDiv.classList.add('answer-content');
            contentDiv.innerHTML = content;

            div.appendChild(contentDiv);
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return contentDiv;
        }

        function updateStatus(text) {
            status.textContent = text;
        }

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text || isProcessing) return;

            addMessage('user', text);
            userInput.value = '';
            isProcessing = true;
            sendBtn.disabled = true;
            typing.classList.add('active');
            updateStatus('思考中...');

            const showThink = thinkChainCheck.checked;
            const useStream = streamCheck.checked;

            try {
                if (useStream) {
                    await streamRequest(text, showThink);
                } else {
                    await normalRequest(text, showThink);
                }
            } catch (err) {
                console.error('请求失败:', err);
                addMessage('assistant', '[连接失败: ' + err.message + ']', true).classList.add('error-message');
            } finally {
                isProcessing = false;
                sendBtn.disabled = false;
                typing.classList.remove('active');
                updateStatus('就绪');
                userInput.focus();
            }
        }

        async function streamRequest(text, showThink) {
            console.log('开始流式请求:', text);
            
            const response = await fetch('/v1/chat/completions', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    messages: [{role: 'user', content: text}],
                    stream: true,
                    think_chain: true,
                    session_id: sessionId,
                    max_tokens: 4096
                })
            });
            
            if (!response.ok) {
                throw new Error('HTTP ' + response.status);
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            let thinkDiv = null;
            let answerDiv = null;
            let thinkContent = '';
            let answerContent = '';
            let hasThinkStarted = false;
            let hasAnswerStarted = false;
            let buffer = '';
            
            while (true) {
                const {done, value} = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, {stream: true});
                buffer += chunk;
                
                // 使用正则分割行，兼容 \r\n 和 \n
                const lines = buffer.split(/\r?\n/);
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (!line.trim()) continue;
                    if (!line.startsWith('data: ')) continue;
                    
                    const dataStr = line.slice(6).trim();
                    if (dataStr === '[DONE]') continue;
                    
                    try {
                        const json = JSON.parse(dataStr);
                        const delta = json.choices?.[0]?.delta || {};
                        
                        if (delta.think_token !== undefined) {
                            thinkContent += delta.think_token;
                            if (showThink) {
                                if (!hasThinkStarted) {
                                    thinkDiv = addMessage('assistant', '', true, false);
                                    hasThinkStarted = true;
                                }
                                thinkDiv.innerHTML = '[思考] ' + thinkContent;
                                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                            }
                        } else if (delta.content !== undefined) {
                            answerContent += delta.content;
                            if (!hasAnswerStarted) {
                                answerDiv = addMessage('assistant', '', false, true);
                                hasAnswerStarted = true;
                            }
                            answerDiv.innerHTML = answerContent.replace(/\n/g, '<br>');
                            messagesDiv.scrollTop = messagesDiv.scrollHeight;
                        }
                    } catch (e) {
                        console.error('解析JSON失败:', e, '数据:', dataStr);
                    }
                }
            }
            
            // 处理剩余buffer
            if (buffer.trim() && buffer.startsWith('data: ')) {
                const dataStr = buffer.slice(6).trim();
                if (dataStr !== '[DONE]') {
                    try {
                        const json = JSON.parse(dataStr);
                        const delta = json.choices?.[0]?.delta || {};
                        if (delta.content !== undefined) {
                            answerContent += delta.content;
                            if (!hasAnswerStarted) {
                                answerDiv = addMessage('assistant', '', false, true);
                                hasAnswerStarted = true;
                            }
                            answerDiv.innerHTML = answerContent.replace(/\n/g, '<br>');
                        }
                    } catch (e) {
                        console.error('解析剩余buffer失败:', e);
                    }
                }
            }
            
            console.log('流式请求完成. 思考长度:', thinkContent.length, '回答长度:', answerContent.length);
            
            if (!hasThinkStarted && !hasAnswerStarted) {
                addMessage('assistant', '[未收到任何响应]', false, true).classList.add('error-message');
            }
        }

        async function normalRequest(text, showThink) {
            console.log('开始非流式请求:', text);
            
            const response = await fetch('/v1/chat/completions', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    messages: [{role: 'user', content: text}],
                    stream: false,
                    think_chain: true,
                    session_id: sessionId,
                    max_tokens: 4096
                })
            });

            if (!response.ok) throw new Error('HTTP ' + response.status);

            const data = await response.json();
            console.log('收到非流式响应:', data);
            
            const content = data.choices?.[0]?.message?.content || '[无响应]';
            const think = data.think_content || '';

            if (showThink && think) {
                addMessage('assistant', '[思考] ' + think, true, false);
            }
            addMessage('assistant', content, false, true);
        }

        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        userInput.focus();

        setInterval(async () => {
            try {
                const resp = await fetch('/health');
                const data = await resp.json();
                if (!data.model_loaded) updateStatus('模型加载中...');
            } catch (e) {
                updateStatus('服务断开');
            }
        }, 30000);
    </script>
</body>
</html>"""

# 强制更新模板文件
with open(INDEX_HTML, 'w', encoding='utf-8') as f:
    f.write(INDEX_HTML_CONTENT)
logger.info(f"已更新模板文件: {INDEX_HTML}")

# 配置
API_BASE_URL = os.getenv("API_URL", "http://localhost:12001")

# 创建 FastAPI 应用
app = FastAPI(
    title="Medical LLM Web UI",
    description="Browser Interface for Qwen3-14B Medical Assistant",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """主页"""
    return templates.TemplateResponse("index.html", {"request": request})


def _parse_api_url(url: str):
    """解析API URL为组件"""
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    path_prefix = parsed.path or ''
    is_https = parsed.scheme == 'https'
    return host, port, path_prefix, is_https


async def _stream_proxy_generator(target_url: str, body: bytes, headers: dict):
    """
    流式代理生成器 - 使用标准库http.client在后台线程中执行
    借鉴rest_sse_chat.py的生成器模式
    """
    import time
    
    host, port, path_prefix, is_https = _parse_api_url(target_url)
    
    # 构建完整路径
    parsed_target = urllib.parse.urlparse(target_url)
    path = parsed_target.path
    if parsed_target.query:
        path += '?' + parsed_target.query
    
    # 数据队列，用于线程间通信
    data_queue = std_queue.Queue()
    stop_event = threading.Event()
    
    def make_streaming_request():
        """在后台线程中执行流式HTTP请求"""
        try:
            # 创建连接
            if is_https:
                conn = http.client.HTTPSConnection(host, port, timeout=300)
            else:
                conn = http.client.HTTPConnection(host, port, timeout=300)
            
            # 准备头
            request_headers = {
                'Content-Type': headers.get('content-type', 'application/json'),
                'Accept': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
            
            # 发送请求
            conn.request('POST', path, body=body, headers=request_headers)
            response = conn.getresponse()
            
            # 读取响应头
            content_type = response.getheader('Content-Type', '')
            data_queue.put(('headers', {
                'status': response.status,
                'content_type': content_type
            }))
            
            # 流式读取内容
            chunk_size = 1024
            while not stop_event.is_set():
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                data_queue.put(('data', chunk.decode('utf-8', errors='replace')))
            
            conn.close()
            data_queue.put(('end', None))
            
        except Exception as e:
            logger.error(f"后台请求线程错误: {e}")
            data_queue.put(('error', str(e)))
    
    # 启动后台线程
    thread = threading.Thread(target=make_streaming_request)
    thread.daemon = True
    thread.start()
    
    # 等待响应头
    headers_received = False
    start_time = time.time()
    timeout = 30  # 等待头的超时时间
    
    while not headers_received and (time.time() - start_time) < timeout:
        try:
            msg_type, msg_data = data_queue.get(timeout=0.1)
            if msg_type == 'headers':
                headers_received = True
                if msg_data['status'] != 200:
                    # 错误响应
                    yield json.dumps({
                        "error": f"Backend returned {msg_data['status']}",
                        "status": msg_data['status']
                    }).encode('utf-8')
                    stop_event.set()
                    return
            elif msg_type == 'error':
                yield json.dumps({"error": msg_data}).encode('utf-8')
                stop_event.set()
                return
            elif msg_type == 'data':
                # 已经有数据了，直接yield
                yield msg_data.encode('utf-8')
                headers_received = True
        except std_queue.Empty:
            await asyncio.sleep(0.01)
    
    if not headers_received:
        yield json.dumps({"error": "Timeout waiting for backend response"}).encode('utf-8')
        stop_event.set()
        return
    
    # 继续读取数据
    while True:
        try:
            msg_type, msg_data = data_queue.get(timeout=0.1)
            if msg_type == 'data':
                yield msg_data.encode('utf-8')
            elif msg_type == 'end':
                break
            elif msg_type == 'error':
                logger.error(f"流式读取错误: {msg_data}")
                break
        except std_queue.Empty:
            # 检查线程是否还在运行
            if not thread.is_alive() and data_queue.empty():
                break
            await asyncio.sleep(0.01)
    
    # 确保线程结束
    stop_event.set()
    if thread.is_alive():
        thread.join(timeout=2)


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_api(request: Request, path: str):
    """
    API 代理：将前端请求转发到推理服务
    关键修复：使用标准库http.client实现可靠的流式代理
    """
    target_url = f"{API_BASE_URL}/v1/{path}"
    
    method = request.method
    body = await request.body()
    headers = dict(request.headers)
    
    # 清理 hop-by-hop 头
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("accept-encoding", None)
    headers.pop("transfer-encoding", None)
    headers.pop("connection", None)
    
    # 确保接受流式响应
    headers["accept"] = "text/event-stream"
    headers["cache-control"] = "no-cache"
    
    try:
        # 检测是否为流式请求
        is_stream_request = False
        try:
            body_json = json.loads(body.decode('utf-8')) if body else {}
            is_stream_request = body_json.get("stream", False)
        except:
            pass
        
        logger.info(f"代理请求: {method} {target_url}, stream={is_stream_request}")
        
        if is_stream_request and method == "POST":
            # 流式请求：使用自定义生成器
            async def stream_generator():
                async for chunk in _stream_proxy_generator(target_url, body, headers):
                    yield chunk
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "content-type": "text/event-stream; charset=utf-8",
                    "cache-control": "no-cache",
                    "connection": "keep-alive",
                    "x-accel-buffering": "no",
                }
            )
        else:
            # 非流式请求：使用同步http.client
            host, port, _, is_https = _parse_api_url(target_url)
            parsed_target = urllib.parse.urlparse(target_url)
            path_only = parsed_target.path
            if parsed_target.query:
                path_only += '?' + parsed_target.query
            
            # 在线程中执行同步请求
            result_queue = std_queue.Queue()
            
            def make_sync_request():
                try:
                    if is_https:
                        conn = http.client.HTTPSConnection(host, port, timeout=60)
                    else:
                        conn = http.client.HTTPConnection(host, port, timeout=60)
                    
                    request_headers = {
                        'Content-Type': headers.get('content-type', 'application/json'),
                    }
                    
                    conn.request(method, path_only, body=body, headers=request_headers)
                    response = conn.getresponse()
                    
                    data = response.read().decode('utf-8', errors='replace')
                    conn.close()
                    
                    result_queue.put(('success', response.status, data))
                except Exception as e:
                    result_queue.put(('error', str(e)))
            
            thread = threading.Thread(target=make_sync_request)
            thread.start()
            thread.join(timeout=60)
            
            if thread.is_alive():
                return JSONResponse(
                    status_code=504,
                    content={"error": "Backend request timeout"}
                )
            
            status, msg_data = result_queue.get()
            if status == 'error':
                return JSONResponse(
                    status_code=502,
                    content={"error": msg_data}
                )
            
            _, status_code, data = msg_data
            try:
                json_data = json.loads(data)
                return JSONResponse(content=json_data, status_code=status_code)
            except:
                return JSONResponse(
                    content={"raw_response": data},
                    status_code=status_code
                )
                
    except Exception as e:
        logger.error(f"代理错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        host, port, _, is_https = _parse_api_url(API_BASE_URL)
        
        def check_health():
            try:
                if is_https:
                    conn = http.client.HTTPSConnection(host, port, timeout=10)
                else:
                    conn = http.client.HTTPConnection(host, port, timeout=10)
                
                conn.request('GET', '/health')
                response = conn.getresponse()
                data = response.read().decode('utf-8')
                conn.close()
                return json.loads(data)
            except Exception as e:
                return {"error": str(e)}
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, check_health)
        return result
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "degraded",
            "ui": "running",
            "api": "disconnected",
            "api_url": API_BASE_URL,
            "error": str(e)
        }


def start_web_ui(host: str = "0.0.0.0", port: int = 8080, api_url: str = None):
    """启动 Web UI 服务"""
    global API_BASE_URL
    if api_url:
        API_BASE_URL = api_url

    logger.info("=" * 60)
    logger.info("Web UI Service Starting")
    logger.info("=" * 60)
    logger.info(f"  Access URL: http://{host}:{port}")
    logger.info(f"  Backend API: {API_BASE_URL}")
    logger.info("=" * 60)

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-url", default="http://localhost:12001")

    args = parser.parse_args()
    start_web_ui(args.host, args.port, args.api_url)