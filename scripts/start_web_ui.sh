#!/bin/bash
# -*- coding: utf-8 -*-
# scripts/start_web_ui.sh - Web界面启动脚本

set -e

# 获取项目路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 加载环境变量（如果存在 .env 文件）
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "加载环境变量: $PROJECT_ROOT/.env"
    source "$PROJECT_ROOT/.env"
fi

# 配置（优先使用环境变量，否则使用默认值）
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-8080}"
API_URL="${API_URL:-http://localhost:12001}"

echo "========================================"
echo "  Web UI 启动器"
echo "========================================"

# 设置Python路径
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "配置信息:"
echo "  Web地址:    $WEB_HOST:$WEB_PORT"
echo "  API后端:    $API_URL"

echo ""
echo "启动Web UI..."
echo "  访问地址: http://$WEB_HOST:$WEB_PORT"
echo ""

cd "$PROJECT_ROOT"

# 导出环境变量供Python读取
export API_URL="$API_URL"

# 启动Web UI
exec python3 -m src.chat.web_ui \
    --host "$WEB_HOST" \
    --port "$WEB_PORT" \
    --api-url "$API_URL"