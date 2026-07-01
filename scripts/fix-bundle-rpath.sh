#!/bin/bash
# 修复 sidecar-dist 中 Python framework 的 rpath — BUILD TIME 执行
# 目的: 消除所有 /opt/homebrew 硬编码路径，使 Python 在任意 Mac 上可运行
# 用法: bash scripts/fix-bundle-rpath.sh [sidecar-dist路径]
set -euo pipefail

SD="${1:-sidecar-dist}"
FW="$SD/python/Python.framework/Versions/3.14"
BIN="$FW/bin/python3"
DEPS_DIR="$SD/python/deps"

echo "=== 知伴 Bundle Rpath 修复 ==="
echo "目标: $SD"

if [ ! -d "$FW" ]; then
  echo "❌ Python framework 不存在: $FW"
  exit 1
fi

# 步骤1: 修复所有 Mach-O 文件的 Homebrew 依赖
echo ""
echo "[1/4] 扫描并修复 Homebrew 依赖..."

fix_macho() {
  local target="$1"

  # 收集需要修改的依赖映射
  local changes=""
  while IFS= read -r line; do
    local lib_path="${line%%(*}"
    lib_path="${lib_path## }"
    lib_path="${lib_path%% }"
    [[ -z "$lib_path" ]] && continue

    # 跳过系统库
    [[ "$lib_path" == /usr/lib/* || "$lib_path" == /System/Library/* ]] && continue

    local base
    base="$(basename "$lib_path")"

    # 类型A: Python.framework 内部引用 → @executable_path/../
    if [[ "$lib_path" == *Python.framework* ]]; then
      local found
      found="$(find "$FW" -name "$base" \( -type f -o -type l \) 2>/dev/null | head -1)"
      if [[ -n "$found" ]]; then
        local rel_from_fw="${found#$FW/}"
        changes="${changes}${lib_path}|@executable_path/../${rel_from_fw}
"
      fi
      continue
    fi

    # 类型B: /opt/homebrew 其他依赖 (openssl, sqlite3, etc.) → deps/
    if [[ "$lib_path" == /opt/homebrew/* ]]; then
      mkdir -p "$DEPS_DIR"
      if [[ ! -f "$DEPS_DIR/$base" ]] && [[ -f "$lib_path" ]]; then
        cp "$lib_path" "$DEPS_DIR/$base" 2>/dev/null || true
        codesign --remove-signature "$DEPS_DIR/$base" 2>/dev/null || true
      fi
      if [[ -f "$DEPS_DIR/$base" ]]; then
        changes="${changes}${lib_path}|@executable_path/../deps/${base}
"
      fi
    fi
  done < <(otool -L "$target" 2>/dev/null | tail -n +2)

  # 应用修改
  if [[ -n "$changes" ]]; then
    while IFS='|' read -r old_path new_path; do
      [[ -z "$old_path" ]] && continue
      install_name_tool -change "$old_path" "$new_path" "$target" 2>/dev/null || true
    done <<< "$changes"
    # Ad-hoc 签名
    codesign --remove-signature "$target" 2>/dev/null || true
    codesign --sign - --force "$target" 2>/dev/null || true
    return 0
  fi
  return 1
}

# 收集所有 Mach-O 文件 (使用 while read < <(find) 避免 subshell 变量丢失)
MACHO_COUNT=0
FIXED_COUNT=0
while IFS= read -r f; do
  file "$f" 2>/dev/null | grep -q "Mach-O" || continue
  MACHO_COUNT=$((MACHO_COUNT + 1))

  # 检查是否有外部依赖
  has_ext=$(otool -L "$f" 2>/dev/null | grep -c "/opt/homebrew" || true)
  if [[ "$has_ext" -gt 0 ]]; then
    echo "  修复: ${f#$FW/} ($has_ext 个外部依赖)"
    fix_macho "$f" && FIXED_COUNT=$((FIXED_COUNT + 1)) || true
  fi
done < <(find "$FW" -type f 2>/dev/null)

echo "  扫描: $MACHO_COUNT 个 Mach-O, $FIXED_COUNT 个需修复"

# 步骤2: 对 deps 目录中的 dylib 也修复它们之间的依赖
echo ""
echo "[2/4] 修复 deps/ 内部依赖..."
if ls "$DEPS_DIR"/*.dylib &>/dev/null 2>&1; then
  for dep in "$DEPS_DIR"/*.dylib; do
    [[ -f "$dep" ]] || continue
    # 修复 deps dylib 之间的相互引用
    while IFS= read -r line; do
      lib_path="${line%%(*}"
      lib_path="${lib_path## }"
      lib_path="${lib_path%% }"
      [[ -z "$lib_path" ]] && continue
      base="$(basename "$lib_path")"
      if [[ -f "$DEPS_DIR/$base" ]]; then
        install_name_tool -change "$lib_path" "@loader_path/$base" "$dep" 2>/dev/null || true
      fi
    done < <(otool -L "$dep" 2>/dev/null | grep "/opt/homebrew" || true)
    codesign --remove-signature "$dep" 2>/dev/null || true
    codesign --sign - --force "$dep" 2>/dev/null || true
  done
fi

# 步骤3: 重新 ad-hoc 签名整个 framework
echo ""
echo "[3/4] 重新 ad-hoc 签名所有 Mach-O..."
while IFS= read -r f; do
  file "$f" 2>/dev/null | grep -q "Mach-O" || continue
  codesign --remove-signature "$f" 2>/dev/null || true
  codesign --sign - --force "$f" 2>/dev/null || true
done < <(find "$FW" -type f 2>/dev/null)

# 步骤4: 创建 marker 文件，start-sidecar.sh 检测到后会跳过运行时修复
echo ""
echo "[4/4] 创建预修复 marker..."
touch "$SD/python/.rpath_fixed"
touch "$SD/python/.signed"
echo "  ✅ Marker 文件已创建 (启动时跳过 rpath 修复 + 签名)"

echo ""
echo "=== Rpath 修复完成 ==="

# 验证
echo ""
echo "验证: 检查是否还有 Homebrew 残留..."
REMAINING=""
while IFS= read -r f; do
  file "$f" 2>/dev/null | grep -q "Mach-O" || continue
  if otool -L "$f" 2>/dev/null | grep -q "/opt/homebrew"; then
    REMAINING="${REMAINING}  ⚠️  $f
"
  fi
done < <(find "$FW" -type f 2>/dev/null)
if [ -z "$REMAINING" ]; then
  echo "  ✅ 无 Homebrew 路径残留"
else
  echo "$REMAINING"
  echo "  ⚠️  仍有残留，请手动检查"
fi

# 验证 Python 是否可运行
echo ""
echo "测试: Python 是否可运行..."
if "$BIN" --version 2>&1; then
  echo "  ✅ Python 可独立运行"
else
  echo "  ❌ Python 无法运行 — 请检查 rpath 修复"
fi
