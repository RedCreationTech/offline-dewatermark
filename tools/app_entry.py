"""PyInstaller 打包入口跳板。

wm/app.py 内部使用相对导入（from . import ...），被 PyInstaller 当散装脚本
直接执行时会报 ImportError: attempted relative import with no known parent
package。打包入口必须是不在包内的绝对导入脚本，故设此跳板；
tools/package_app.py 以本文件为 PyInstaller 入口。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 开发目录直跑兜底

from wm.app import main

if __name__ == "__main__":
    main()
