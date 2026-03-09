#!/bin/bash
# -*- coding: utf-8 -*-
# scripts/start_server.sh - LLM推理服务启动脚本（T4生产环境推荐）
#
# 特性：
# - 启用请求队列管理，防止T4 16GB显存OOM
# - 最大2并发，超出的请求排队等待
# - 自动清理KV缓存，请求间显存回收
# - 支持健康检查和Kubernetes探针
#
# 用法：
#   ./scripts/start_server.sh                    # 使用默认配置（端口12001）
#   ./scripts/start_server.sh --port 8000        # 自定义端口
#   ./scripts/start_server.sh --adapter /path    # 指定适配器路径

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Qwen3-14B Medical LLM 推理服务启动器  ${NC}"
echo -e "${GREEN}========================================${NC}"

# 加载环境变量（如果存在 .env 文件）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${GREEN}加载环境变量: $PROJECT_ROOT/.env${NC}"
    source "$PROJECT_ROOT/.env"
fi

# 默认配置（优先使用环境变量，否则使用硬编码默认值）
MODEL_PATH="${MODEL_PATH:-/home/yaoxp/models/Qwen3-14B/}"
ADAPTER_PATH="${ADAPTER_PATH:-/home/yaoxp/work/sft/uf8/output/experiments/sft_medical_002_0305_194002/final_adapter/}"
SERVER_HOST="${API_HOST:-0.0.0.0}"
SERVER_PORT="${API_PORT:-12001}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
DOMAIN="${DOMAIN:-medical}"

# 解析命令行参数（覆盖环境变量）
while [[ $# -gt 0 ]]; do
    case $1 in
        --port|-p)
            SERVER_PORT="$2"
            shift 2
            ;;
        --model|-m)
            MODEL_PATH="$2"
            shift 2
            ;;
        --adapter|-a)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --host)
            SERVER_HOST="$2"
            shift 2
            ;;
        --concurrent|-c)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --max-seq-length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --domain|-d)
            DOMAIN="$2"
            shift 2
            ;;
        --single)
            MAX_CONCURRENT=1
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --port PORT              服务端口 (默认: 12001, 环境变量: API_PORT)"
            echo "  --host HOST              绑定地址 (默认: 0.0.0.0, 环境变量: API_HOST)"
            echo "  --model PATH             模型路径 (环境变量: MODEL_PATH)"
            echo "  --adapter PATH           适配器路径 (环境变量: ADAPTER_PATH)"
            echo "  --concurrent N           最大并发数 (默认: 2, 环境变量: MAX_CONCURRENT)"
            echo "  --max-seq-length N       最大序列长度 (默认: 8192, 环境变量: MAX_SEQ_LENGTH)"
            echo "  --domain DOMAIN          领域 (默认: medical, 环境变量: DOMAIN)"
            echo "  --single                 单并发模式"
            echo "  --help                   显示此帮助"
            echo ""
            echo "环境变量配置（创建 .env 文件）:"
            echo "  MODEL_PATH=/path/to/model"
            echo "  ADAPTER_PATH=/path/to/adapter"
            echo "  API_URL=http://172.16.0.93:12001"
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            exit 1
            ;;
    esac
done

# 验证路径
if [[ ! -d "$MODEL_PATH" ]]; then
    echo -e "${RED}错误: 模型路径不存在: $MODEL_PATH${NC}"
    echo -e "${YELLOW}请设置 MODEL_PATH 环境变量或使用 --model 参数${NC}"
    exit 1
fi

if [[ ! -d "$ADAPTER_PATH" ]]; then
    echo -e "${YELLOW}警告: 适配器路径不存在: $ADAPTER_PATH${NC}"
    echo -e "${YELLOW}将使用基础模型继续启动...${NC}"
    ADAPTER_PATH=""
fi

# 检查GPU
echo -e "${GREEN}[1/5] 检查GPU环境...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}错误: nvidia-smi 未找到，请检查CUDA驱动${NC}"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
echo -e "  GPU: $GPU_INFO"

# 检查显存（T4应该有16GB）
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "0")
if [[ "$GPU_MEM" -lt 15000 ]]; then
    echo -e "${YELLOW}警告: 检测到显存 ${GPU_MEM}MB，低于T4标准(16GB)${NC}"
    echo -e "${YELLOW}建议降低 MAX_CONCURRENT 到 1${NC}"
fi

# 检查Python环境
echo -e "${GREEN}[2/5] 检查Python环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: python3 未找到${NC}"
    exit 1
fi

# 检查依赖
echo -e "${GREEN}[3/5] 检查依赖包...${NC}"
REQUIRED_PACKAGES=("fastapi" "uvicorn" "torch" "transformers" "accelerate" "bitsandbytes")
MISSING_PACKAGES=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
    echo -e "${YELLOW}警告: 缺少依赖包: ${MISSING_PACKAGES[*]}${NC}"
    echo -e "${YELLOW}尝试安装...${NC}"
    pip install "${MISSING_PACKAGES[@]}" || {
        echo -e "${RED}错误: 依赖安装失败${NC}"
        exit 1
    }
fi

# 设置Python路径
echo -e "${GREEN}[4/5] 配置环境...${NC}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
echo -e "  项目根目录: $PROJECT_ROOT"
echo -e "  PYTHONPATH: $PYTHONPATH"

# 配置验证
echo -e "${GREEN}[5/5] 配置摘要:${NC}"
echo -e "  模型路径:    $MODEL_PATH"
echo -e "  适配器路径:  $ADAPTER_PATH"
echo -e "  服务地址:    $SERVER_HOST:$SERVER_PORT"
echo -e "  最大并发:    $MAX_CONCURRENT (队列模式)"
echo -e "  序列长度:    $MAX_SEQ_LENGTH"
echo -e "  领域:        $DOMAIN"
echo -e "  队列管理:    启用"

# 启动服务
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  正在启动LLM推理服务...                ${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  API文档:    http://${SERVER_HOST}:${SERVER_PORT}/docs"
echo -e "  健康检查:   http://${SERVER_HOST}:${SERVER_PORT}/health"
echo -e "  模型列表:   http://${SERVER_HOST}:${SERVER_PORT}/v1/models"
echo -e "  聊天接口:   http://${SERVER_HOST}:${SERVER_PORT}/v1/chat/completions"
echo ""

cd "$PROJECT_ROOT"

# 通过命令行参数传递给Python模块
exec python3 -m src.server.inference_server \
    --model "$MODEL_PATH" \
    --adapter "$ADAPTER_PATH" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --max-concurrent "$MAX_CONCURRENT" \
    --domain "$DOMAIN"