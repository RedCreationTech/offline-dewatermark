"""PyInstaller 打包脚本：把 wm.app 桌面前端打成独立可执行目录。

用法（在项目根目录）：
    .venv/bin/python tools/package_app.py              # one-dir 控制台版（默认）
    .venv/bin/python tools/package_app.py --windowed   # 不弹控制台（Windows 交付用）
    .venv/bin/python tools/package_app.py --clean      # 先清 build/ dist/ 再打包

产物：dist/去水印/
    去水印(.exe) + _internal/   PyInstaller one-dir 运行体
    models/                     LaMa 修复模型（从项目 models/ 复制，可替换）
    config/logging.json         日志级别配置
    tools/wevisone_server.py    提示词定位推理服务脚本（由底座环境的 Python 执行）
    使用说明.txt
    合并底座.ps1 / .bat         把「去水印底座包」并入本目录的一键脚本

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
4. 界面中的「提示词定位」依赖单独的 WeVisOne 大模型环境（「去水印底座包」）：
   底座包与本包解压到同一位置会自动合并；若分开了解压，双击「合并底座.bat」
   即可把底座的 models/WeVisOne-VL-2B 与 wevisone_runtime/ 并入本目录。
   没有底座包时请用默认的「自动识别」。
5. 完全离线运行，不需要网络。
"""

# 随主程序包分发的「合并底座」脚本：底座包（模型 + 便携环境）打得少、主程序包
# 更新频繁，收件人两个包分开解压时用它一键并入 exe 旁。逻辑维护在这里
# （stage_runtime_files 每次打包重新生成），不单独发脚本文件。
MERGE_PS1 = r"""# 合并底座.ps1 —— 把「去水印底座包」的模型与运行环境并入本程序目录
#
# 什么时候用：本目录（去水印.exe 旁）还没有 wevisone_runtime\ 或
# models\WeVisOne-VL-2B\，需要把底座包的内容并进来才能用「提示词定位」。
# 两个 zip 已解压到同一位置时会自动合并，本脚本会提示"无需合并"，不会重复搬运。
#
# 用法：
#   双击「合并底座.bat」（自动找本目录下的「去水印」子目录，找不到会请你把底座目录拖进来）
#   或命令行：powershell -NoProfile -ExecutionPolicy Bypass -File 合并底座.ps1 [-BaseDir D:\路径\去水印] [-Copy]
#
# 参数：
#   -BaseDir  底座包解压出的「去水印」目录（给它的上一级目录也行）
#   -Copy     用复制代替移动（默认移动；同盘移动是瞬时的，跨盘搬 7GB 较慢）
param(
    [string]$BaseDir = "",
    [switch]$Copy
)
$ErrorActionPreference = 'Stop'
if (-not $PSScriptRoot) { $PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }
$target = $PSScriptRoot

function Test-Merged([string]$root) {
    return (Test-Path (Join-Path $root 'wevisone_runtime\python\python.exe')) -and
           (Test-Path (Join-Path $root 'models\WeVisOne-VL-2B\model.safetensors'))
}

function Resolve-Base([string]$dir) {
    # 接受底座目录本身，或其上一级（里面还有一层「去水印」）
    if (Test-Path (Join-Path $dir 'wevisone_runtime\python\python.exe')) { return $dir }
    if (Test-Path (Join-Path $dir '去水印\wevisone_runtime\python\python.exe')) { return (Join-Path $dir '去水印') }
    return $null
}

if (Test-Merged $target) {
    Write-Host '本目录已包含提示词环境（wevisone_runtime 与 models\WeVisOne-VL-2B），无需合并。'
    exit 0
}

$base = $null
foreach ($c in @($BaseDir, (Join-Path $target '去水印'))) {
    if ($c) { $base = Resolve-Base $c; if ($base) { break } }
}
if (-not $base) {
    $in = Read-Host '未找到底座目录。请把底座包解压出的「去水印」文件夹拖到本窗口，然后按回车'
    $base = Resolve-Base ($in.Trim().Trim('"'))
}
if (-not $base) {
    Write-Host '错误：底座目录无效——里面应有 wevisone_runtime\ 与 models\WeVisOne-VL-2B\。'
    Write-Host '请确认底座包已解压，或用 -BaseDir 指定其位置后重跑。'
    exit 1
}
Write-Host "底座目录：$base"
Write-Host "并入目标：$target"

$items = @('wevisone_runtime', 'models\WeVisOne-VL-2B', 'models\inpainting_lama_2025jan.onnx', '底座说明.txt')
foreach ($it in $items) {
    $src = Join-Path $base $it
    $dst = Join-Path $target $it
    if (-not (Test-Path $src)) { continue }
    if (Test-Path $dst) {
        $same = (Test-Path $src -PathType Leaf) -and ((Get-Item $src).Length -eq (Get-Item $dst).Length)
        if ($same) { Write-Host "跳过（已存在且大小相同）：$it" } else { Write-Host "跳过（目标已存在）：$it" }
        continue
    }
    $parent = Split-Path $dst -Parent
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
    Write-Host "并入：$it …"
    if ($Copy) { Copy-Item $src $dst -Recurse -Force } else { Move-Item $src $dst -Force }
}

if (Test-Merged $target) {
    Write-Host ''
    Write-Host '合并完成！双击 去水印.exe，找水印方式选「提示词定位」即可使用。'
    Write-Host '（首次使用需加载模型约 20-60 秒，状态栏会显示进度）'
    exit 0
}
Write-Host '错误：合并后仍缺 wevisone_runtime\python\python.exe 或 models\WeVisOne-VL-2B\model.safetensors，'
Write-Host '请检查底座包是否完整（可重新解压底座 zip 后重跑本脚本）。'
exit 1
"""

# cmd 按 ANSI 读 .bat，中文内容会乱码；保持纯 ASCII，中文文件名靠 %~dpn0
# （bat 自身路径去扩展名）在运行时拼出，与 ps1 文件名始终保持一致。
MERGE_BAT = """@echo off
rem Merge-base bootstrap: run the sibling .ps1 with execution policy bypass.
rem Keep this file pure ASCII; the Chinese .ps1 name is resolved at runtime
rem via %~dpn0 (this script's own full path without extension).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dpn0.ps1" %*
pause
"""


def build(windowed: bool, clean: bool, distpath: Path, workpath: Path) -> Path:
    if clean:
        for p in (distpath, workpath, ROOT / "build"):
            if p.exists():
                shutil.rmtree(p)
                print(f"[package] 已清理 {p}")

    sep = ";" if sys.platform == "win32" else ":"
    args = [
        # 入口必须用包外跳板（绝对导入）：直接打 wm/app.py 会因相对导入
        # 在启动即崩（ImportError: attempted relative import）
        str(ROOT / "tools" / "app_entry.py"),
        "--name", APP_NAME,
        "--onedir",
        "--windowed" if windowed else "--console",
        "--noconfirm",
        "--clean",
        "--paths", str(ROOT),
        "--distpath", str(distpath),
        "--workpath", str(workpath),
        "--specpath", str(workpath),
        # wm 包内资源：字体（synth 用）与界面 SVG；源路径必须绝对——相对路径
        # 会被 PyInstaller 按 spec 所在目录解析（Windows CI 实测报找不到）。
        "--add-data", f"{ROOT / 'wm' / 'fonts'}{sep}wm/fonts",
        "--add-data", f"{ROOT / 'wm' / 'assets'}{sep}wm/assets",
        # rapidocr 的 OCR 模型与配置随 wheel 安装，需显式收集
        "--collect-data", "rapidocr_onnxruntime",
        # rapidocr 的 det/cls/rec 子模块由 importlib 动态加载
        # （rapid_ocr_api.init_module），静态分析收集不到；缺失时 RapidOCR()
        # 初始化抛 ImportError，OCR 静默失效（2026-11-26 打包批次实测：
        # 提示词定位置信度塌缩、60/70 透明度档被证据闸门拦截）。
        # 其依赖 pyclipper/shapely/six 一并显式声明。
        "--collect-submodules", "rapidocr_onnxruntime",
        "--hidden-import", "pyclipper",
        "--hidden-import", "shapely",
        "--hidden-import", "six",
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

    # 合并底座脚本：ps1 用 UTF-8 带 BOM（PowerShell 5.1 对无 BOM 文件按 ANSI
    # 误读中文）；bat 纯 ASCII（cmd 按 ANSI 解析批处理），编码声明即防线。
    (bundle / "合并底座.ps1").write_text(MERGE_PS1, encoding="utf-8-sig")
    (bundle / "合并底座.bat").write_text(MERGE_BAT, encoding="ascii")

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
