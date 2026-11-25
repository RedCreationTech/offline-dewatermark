"""PyInstaller 打包脚本：把 wm.app 桌面前端打成独立可执行目录。

用法（在项目根目录）：
    .venv/bin/python tools/package_app.py              # one-dir 控制台版（默认）
    .venv/bin/python tools/package_app.py --windowed   # 不弹控制台（Windows 交付用）
    .venv/bin/python tools/package_app.py --clean      # 先清 build/ dist/ 再打包

产物：dist/去水印/
    去水印(.exe) + _internal/   PyInstaller one-dir 运行体
    models/                     LaMa 修复模型（从项目 models/ 复制，可替换）
    config/logging.json         日志级别配置
    使用说明.txt

把整个 dist/去水印/ 目录拷给目标机器即可运行；代码内通过 wm.paths.app_root()
定位可执行文件旁边的 models/ 与 config/。
Windows 的 .exe 必须在 Windows 机器上运行本脚本生成（PyInstaller 不跨平台编译），
脚本与参数在两个平台一致。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "去水印"
VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

# 运行不需要、且容易把体积/依赖搞乱的模块，显式排除
EXCLUDES = [
    "torch", "torchvision", "transformers",
    "pandas", "matplotlib", "tkinter", "pytest",
    "PyQt5", "PyQt6", "PySide2",
]

README_TEXT = """离线智能去水印 — 可执行包

1. 启动：Windows 双击「去水印.exe」；macOS 在本目录终端执行 ./去水印。
2. models/inpainting_lama_2025jan.onnx 是修复模型，缺失或损坏时程序会明确报错，
   请保持 models/ 目录与可执行文件在同一级，不要删除。
3. 日志写在 logs/wm.log；日志级别在 config/logging.json 的 level 字段
   （DEBUG/INFO/WARNING/ERROR），界面里也可调整。
4. 界面中的「提示词定位」依赖单独的 WeVisOne 大模型环境，本包不含该模型，
   请选择默认的「自动识别」。
5. 完全离线运行，不需要网络。
"""


def build(windowed: bool, clean: bool, distpath: Path, workpath: Path) -> Path:
    if clean:
        for p in (distpath, workpath, ROOT / "build"):
            if p.exists():
                shutil.rmtree(p)
                print(f"[package] 已清理 {p}")

    sep = ";" if sys.platform == "win32" else ":"
    args = [
        str(ROOT / "wm" / "app.py"),
        "--name", APP_NAME,
        "--onedir",
        "--windowed" if windowed else "--console",
        "--noconfirm",
        "--clean",
        "--paths", str(ROOT),
        "--distpath", str(distpath),
        "--workpath", str(workpath),
        "--specpath", str(workpath),
        # wm 包内资源：字体（synth 用）与界面 SVG
        "--add-data", f"wm/fonts{sep}wm/fonts",
        "--add-data", f"wm/assets{sep}wm/assets",
        # rapidocr 的 OCR 模型与配置随 wheel 安装，需显式收集
        "--collect-data", "rapidocr_onnxruntime",
    ]
    for mod in EXCLUDES:
        args += ["--exclude-module", mod]

    print("[package] PyInstaller", " ".join(args))
    from PyInstaller.__main__ import run as pyi_run
    pyi_run(args)
    return distpath / APP_NAME


def stage_runtime_files(bundle: Path, *, with_models: bool = True) -> None:
    """把模型、配置、推理服务脚本和使用说明放到可执行文件旁边（app_root 约定）。"""
    if with_models:
        models_dst = bundle / "models"
        models_dst.mkdir(parents=True, exist_ok=True)
        lama = ROOT / "models" / "inpainting_lama_2025jan.onnx"
        if not lama.is_file():
            raise SystemExit(f"[package] 缺少模型 {lama}，无法打出可运行的包")
        shutil.copy2(lama, models_dst / lama.name)

    config_dst = bundle / "config"
    config_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "config" / "logging.json", config_dst / "logging.json")

    # 提示词定位的推理服务脚本：打包版由底座包里的便携 Python 执行它，
    # 属于代码（随主程序包更新），不属底座。
    tools_dst = bundle / "tools"
    tools_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "tools" / "wevisone_server.py", tools_dst / "wevisone_server.py")

    readme = README_TEXT
    if not with_models:
        readme += ("\n6. 本包为主程序更新包，不含修复模型；models/ 与 wevisone_runtime/\n"
                   "   由「去水印底座包」提供，解压到同一目录后使用。\n")
    (bundle / "使用说明.txt").write_text(readme, encoding="utf-8")
    print(f"[package] 运行文件已就位：{bundle}")


def main() -> None:
    # Windows 控制台默认 cp1252/GBK，打印中文路径会 UnicodeEncodeError；
    # 强制 UTF-8 并容忍替换，保证打包日志在任何机器上都不因打印而崩。
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="package_app", description="PyInstaller 打包 wm.app")
    ap.add_argument("--windowed", action="store_true", help="不显示控制台窗口（Windows 交付用）")
    ap.add_argument("--clean", action="store_true", help="清理 build/ dist/ 后重新打包")
    ap.add_argument("--distpath", default=str(ROOT / "dist"), help="产物根目录（默认 dist/）")
    ap.add_argument("--workpath", default=str(ROOT / "build" / "pyinstaller"), help="中间文件目录")
    ap.add_argument("--no-models", action="store_true",
                    help="不打入 LaMa 模型（主程序更新包模式：模型由底座包提供）")
    ns = ap.parse_args()

    bundle = build(ns.windowed, ns.clean, Path(ns.distpath), Path(ns.workpath))
    stage_runtime_files(bundle, with_models=not ns.no_models)
    print(f"[package] 完成：{bundle}")
    print("[package] 验证：", f'"{bundle / APP_NAME}" --smoke <图片路径>')


if __name__ == "__main__":
    main()
