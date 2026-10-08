#!/usr/bin/env bash
set -e

echo "=== [1/3] Cài đặt các thư viện hệ thống cho Ubuntu ==="
# python3-tk: bắt buộc cho giao diện CustomTkinter / Tkinter
# libgl1 & libglib2.0-0: bắt buộc cho OpenCV (cv2)
sudo apt update
sudo apt install -y python3-tk libgl1 libglib2.0-0 python3-pip python3-venv

echo "=== [2/3] Thiết lập môi trường Python ảo (.venv) ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "=== [3/3] Cài đặt các thư viện Python từ requirements.txt ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Cài đặt hoàn tất! Bạn có thể khởi chạy bằng: ./run.sh hoặc python3 main.py ==="
