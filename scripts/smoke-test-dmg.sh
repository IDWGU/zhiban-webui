#!/bin/bash
# 知伴 DMG 冒烟测试 — 模拟用户安装流程全链路验证
# 用法: bash scripts/smoke-test-dmg.sh [dmg路径或app路径]
#
# 测试流程:
#   1. 挂载DMG / 或直接使用.app
#   2. 检查 app bundle 结构完整性
#   3. 验证 Python 可运行性 (最关键)
#   4. 检查零 Homebrew rpath 残留
#   5. 启动 sidecar → 验证HTTP端点
#   6. WebSocket 查询 → 验证完整流程
#   7. 清理

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS=0; FAIL=0

pass() { echo -e "  ${GREEN}✅ $1${NC}"; PASS=$((PASS+1)); }
fail() { echo -e "  ${RED}❌ $1${NC}"; FAIL=$((FAIL+1)); }
warn() { echo -e "  ${YELLOW}⚠️  $1${NC}"; }

cleanup() {
  if [ -n "${MOUNT_POINT:-}" ] && [ -d "$MOUNT_POINT" ]; then
    hdiutil detach "$MOUNT_POINT" 2>/dev/null || true
  fi
  if [ -n "${SIDECAR_PID:-}" ]; then
    kill $SIDECAR_PID 2>/dev/null || true
    sleep 1
  fi
  lsof -ti :18921 | xargs kill 2>/dev/null || true
}
trap cleanup EXIT

# ── 参数解析 ──
TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  # 默认: 查找最新构建的 .app
  TARGET=$(find release -name "知伴.app" -type d -maxdepth 3 2>/dev/null | head -1)
  if [ -z "$TARGET" ]; then
    echo "用法: bash scripts/smoke-test-dmg.sh [dmg路径 | app路径]"
    echo "      或放在项目根目录自动检测 release/ 下的产物"
    exit 1
  fi
fi

echo "╔══════════════════════════════════════════╗"
echo "║  知伴 DMG 冒烟测试                       ║"
echo "╚══════════════════════════════════════════╝"
echo "目标: $TARGET"

# ── 阶段1: 获取 .app ──
APP=""
MOUNT_POINT=""
if [[ "$TARGET" == *.dmg ]]; then
  echo ""
  echo "┌── 阶段1: 挂载 DMG ────────────────────────┐"
  MOUNT_POINT=$(mktemp -d /tmp/zhiban-smoke.XXXXXX)
  hdiutil attach "$TARGET" -mountpoint "$MOUNT_POINT" -nobrowse 2>/dev/null || {
    fail "DMG 挂载失败"; exit 1
  }
  APP=$(find "$MOUNT_POINT" -name "知伴.app" -type d -maxdepth 2 | head -1)
  pass "DMG 挂载成功 ($MOUNT_POINT)"
elif [[ "$TARGET" == *.app ]] || [[ "$TARGET" == *知伴.app* ]]; then
  APP="$TARGET"
fi

if [ -z "$APP" ] || [ ! -d "$APP" ]; then
  fail "找不到知伴.app"; exit 1
fi

# ── 阶段2: Bundle 结构检查 ──
echo ""
echo "┌── 阶段2: Bundle 结构检查 ──────────────────┐"

SIDECAR_DIR="$APP/Contents/Resources/sidecar-dist"
PY_BIN="$SIDECAR_DIR/python/bin/python3.14"

[ -f "$APP/Contents/MacOS/知伴" ] && pass "Electron 主二进制" || fail "Electron 主二进制缺失"
[ -f "$APP/Contents/Info.plist" ] && pass "Info.plist" || fail "Info.plist 缺失"
[ -f "$APP/Contents/Resources/app.asar" ] && pass "app.asar" || fail "app.asar 缺失"
[ -d "$SIDECAR_DIR" ] && pass "sidecar-dist 目录" || fail "sidecar-dist 目录缺失"
[ -f "$PY_BIN" ] && pass "portable Python 二进制" || fail "portable Python 二进制缺失"
[ -f "$SIDECAR_DIR/start-sidecar.sh" ] && pass "start-sidecar.sh" || fail "start-sidecar.sh 缺失"
[ -d "$SIDECAR_DIR/sidecar-src/sidecar" ] && pass "sidecar 源码" || fail "sidecar 源码缺失"
[ -d "$SIDECAR_DIR/models/llm" ] && pass "LLM 模型目录" || fail "LLM 模型目录缺失"
[ "$(ls "$SIDECAR_DIR/models/llm/"*.gguf 2>/dev/null | wc -l)" -gt 0 ] && pass "GGUF 模型文件" || warn "GGUF 模型文件缺失 (离线模式不可用)"

# ── 阶段3: Python 可运行性 (DMG打包最关键的测试) ──
echo ""
echo "┌── 阶段3: Python 可运行性 (最关键) ─────────┐"

"$PY_BIN" --version > /dev/null 2>&1 && pass "Python --version" || fail "Python 无法运行 → DMG 有问题"

# 检查是否有 Homebrew 硬编码路径残留
HOMEBREW_DEPS=$(otool -L "$PY_BIN" 2>/dev/null | grep "/opt/homebrew" | wc -l | tr -d ' ')
if [ "$HOMEBREW_DEPS" -eq 0 ]; then
  pass "零 Homebrew rpath 残留"
else
  fail "$HOMEBREW_DEPS 个 Homebrew 路径残留 → 用户机器上必然崩溃"
fi

# 验证 @rpath
otool -L "$PY_BIN" 2>/dev/null | grep -q "@rpath" && pass "@rpath 已配置 (可重定位)" || fail "缺少 @rpath → 不可重定位"

# ── 阶段4: 关键模块导入 ──
echo ""
echo "┌── 阶段4: Python 模块导入 ──────────────────┐"

"$PY_BIN" -c "
ok = True
for m in ['fastapi','uvicorn','websockets','httpx','chromadb','sentence_transformers','psutil']:
    try:
        __import__(m)
    except:
        print(f'MODULE_FAIL:{m}')
        ok = False
if ok:
    print('ALL_PASS')
" > /tmp/zhiban-module-test.txt 2>&1

if grep -q "ALL_PASS" /tmp/zhiban-module-test.txt; then
  pass "核心模块导入 (fastapi/uvicorn/chromadb/...)"
else
  grep "MODULE_FAIL" /tmp/zhiban-module-test.txt | while read line; do
    fail "模块缺失: $line"
  done
fi

# ── 阶段5: Sidecar 启动测试 ──
echo ""
echo "┌── 阶段5: Sidecar 启动测试 ─────────────────┐"

# 确保没有残留进程
lsof -ti :18921 | xargs kill 2>/dev/null || true
sleep 1

echo "  启动 sidecar..."
SIDECAR_LOG="/tmp/zhiban-smoke-sidecar.log"
cd "$SIDECAR_DIR" && bash start-sidecar.sh > "$SIDECAR_LOG" 2>&1 &
SIDECAR_PID=$!

# 等待就绪 (最多 60s)
READY=false
for i in $(seq 1 30); do
  if curl -s http://127.0.0.1:18921/ready 2>/dev/null | grep -q "true"; then
    READY=true
    echo "  就绪耗时: ${i}s"
    break
  fi
  if ! kill -0 $SIDECAR_PID 2>/dev/null; then
    echo "  进程在 ${i}s 退出"
    break
  fi
  sleep 2
done

if $READY; then
  pass "Sidecar 启动成功"
else
  fail "Sidecar 启动失败"
  echo "  日志尾部:"
  tail -20 "$SIDECAR_LOG" | while read line; do echo "    $line"; done
fi

# 健康检查
RESP=$(curl -s http://127.0.0.1:18921/health 2>/dev/null)
if echo "$RESP" | grep -q '"status":"ok"'; then
  pass "健康检查 (/health)"
else
  fail "健康检查失败: $RESP"
fi

# ── 阶段6: WebSocket 查询 ──
echo ""
echo "┌── 阶段6: WebSocket 查询 ───────────────────┐"

WS_RESULT=$(python3 -c "
import asyncio, json, websockets

async def test():
    try:
        async with websockets.connect('ws://127.0.0.1:18921/ws', ping_timeout=5) as ws:
            # 消耗初始消息
            try:
                while True:
                    await asyncio.wait_for(ws.recv(), timeout=0.3)
            except asyncio.TimeoutError:
                pass

            await ws.send(json.dumps({
                'type': 'user_query', 'queryText': '你好',
                'context': {'conversationId': '', 'activeDoc': '', 'activeParagraph': ''},
                'openPapers': [], 'apiKey': '', 'model': '', 'baseUrl': '', 'thinking': False
            }))

            timeout = asyncio.get_event_loop().time() + 30
            result = {'workflow': False, 'llm_done': False, 'error': None}
            while asyncio.get_event_loop().time() < timeout:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=8)
                    data = json.loads(raw)
                    t = data.get('type','')
                    if t == 'llm_done':
                        result['llm_done'] = True
                        break
                    elif t == 'workflow_status':
                        result['workflow'] = True
                    elif t == 'status':
                        code = data.get('code','')
                        if code in ('llm_error', 'busy'):
                            result['error'] = f'{code}: {data.get(\"message\",\"\")[:50]}'
                            break
                except asyncio.TimeoutError:
                    break
            return result
    except Exception as e:
        return {'error': str(e)}

print(json.dumps(asyncio.run(test())))
" 2>/dev/null)

if echo "$WS_RESULT" | grep -q '"workflow": true'; then
  pass "WebSocket workflow_status 推送"
elif echo "$WS_RESULT" | grep -q '"llm_done": true'; then
  pass "WebSocket 查询完成 (llm_done)"
else
  warn "WebSocket 查询未完成 (LLM 未加载或离线)"
  echo "  WS结果: $WS_RESULT"
fi

# ── 阶段7: 清理文件检查 ──
echo ""
echo "┌── 阶段7: 打包清洁度 ───────────────────────┐"

TEST_FILES=$(find "$SIDECAR_DIR/sidecar-src" -name '*_test.py' -o -name 'verify.py' 2>/dev/null | wc -l | tr -d ' ')
[ "$TEST_FILES" -eq 0 ] && pass "无测试文件残留" || warn "$TEST_FILES 个测试文件残留"

PYCACHE=$(find "$SIDECAR_DIR" -type d -name '__pycache__' 2>/dev/null | wc -l | tr -d ' ')
[ "$PYCACHE" -eq 0 ] && pass "无 __pycache__" || warn "$PYCACHE 个 __pycache__"

DS_STORE=$(find "$SIDECAR_DIR" -name '.DS_Store' 2>/dev/null | wc -l | tr -d ' ')
[ "$DS_STORE" -eq 0 ] && pass "无 .DS_Store" || warn "$DS_STORE 个 .DS_Store"

ENV_FILES=$(find "$SIDECAR_DIR" -name '.env' -not -name '.env.example' 2>/dev/null | wc -l | tr -d ' ')
[ "$ENV_FILES" -eq 0 ] && pass "无 .env 泄露" || fail "$ENV_FILES 个 .env 泄露"

# ── 总结 ──
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  测试结果                                ║"
echo "╠══════════════════════════════════════════╣"
echo -e "║  ${GREEN}通过: $PASS${NC}  ${RED}失败: $FAIL${NC}"
echo "╚══════════════════════════════════════════╝"

if [ "$FAIL" -gt 0 ]; then
  echo ""
  echo "🔧 修复建议:"
  echo "  1. Python 无法运行 → 检查 sidecar-dist/python/ 是否为 portable 版本"
  echo "  2. Homebrew rpath 残留 → 运行: bash scripts/fix-bundle-rpath.sh"
  echo "  3. 模块缺失 → 运行: sidecar-dist/python/bin/python3.14 -m pip install <module>"
  echo "  4. Sidecar 启动失败 → 查看: $SIDECAR_LOG"
  exit 1
fi

$READY || exit 1
exit 0
