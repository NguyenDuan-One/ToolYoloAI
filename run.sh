#!/usr/bin/env bash
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi
# Force X11 backend to prevent XCB / Wayland assertion conflicts
export GDK_BACKEND=x11
python3 main.py "$@"
