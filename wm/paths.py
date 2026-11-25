"""运行根目录解析：开发模式为项目根，PyInstaller 打包后为可执行文件所在目录。

打包版（tools/package_app.py 产物）把 models/、config/、logs/ 放在可执行文件
旁边，代码内一律通过 app_root() 取根，不再各自拼 __file__，保证同一套代码
在开发目录和 frozen 目录下行为一致。
"""
from __future__ import annotations

import sys
from pathlib import Path


def app_root() -> Path:
    """应用根目录：PyInstaller frozen 时为可执行文件所在目录，否则为项目根。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]
