#!/usr/bin/env bash
set -e

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=== Đóng gói ứng dụng trên Ubuntu (PyInstaller) ==="
pip install pyinstaller
pyinstaller AIDataTool.spec
chmod +x dist/AIDataTool

echo "=== Build hoàn tất! File thực thi nằm tại: dist/AIDataTool ==="
echo "Chạy ứng dụng: ./dist/AIDataTool"
