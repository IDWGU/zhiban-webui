#!/bin/bash
# 知伴开发环境重启脚本
# 清理所有相关进程后重新启动 vite dev server

set -e
cd "$(dirname "$0")/.."

echo "==> 清理旧进程..."

# 清理 sidecar 端口
if lsof -ti :18921 >/dev/null 2>&1; then
  lsof -ti :18921 | xargs kill -9 2>/dev/null
  echo "    ✅ 已清理端口 18921"
else
  echo "    ℹ️  端口 18921 无残留"
fi

# 清理 vite/electron dev 进程
pkill -f "vite" 2>/dev/null && echo "    ✅ 已停止 vite" || echo "    ℹ️  无 vite 进程"

sleep 1

echo "==> 启动开发服务器 (纯 web 模式)..."
npm run dev:web
