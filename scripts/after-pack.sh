#!/bin/bash
# afterPack hook for electron-builder
# PyInstaller --onedir: remove all signatures to prevent Team ID mismatch.
set -e

APP_PATH="$1"
SIDECAR_DIR="$APP_PATH/Contents/Resources/sidecar-dist/sidecar"

if [ ! -d "$SIDECAR_DIR" ]; then
  echo "  [after-pack] sidecar dir not found, skipping"
  exit 0
fi

echo "  [after-pack] removing signatures from sidecar dir..."
find "$SIDECAR_DIR" -type f \( -name "*.dylib" -o -name "Python" -o -name "sidecar" \) | while read f; do
  codesign --remove-signature "$f" 2>/dev/null
done
echo "  [after-pack] ✅ done"
