"""去水印底座包构建脚本：在 GPU 机器上一次性打出「不变的大件」。

底座包内容（约 7~8GB，打好后长期使用，代码更新不需要重打）：
    去水印/                                  ← 与主程序包同名文件夹，解压自动合并
    ├─ models/inpainting_lama_2025jan.onnx   LaMa 修复模型
    ├─ models/WeVisOne-VL-2B/                提示词定位大模型权重
    ├─ wevisone_runtime/python/              便携 Python + GPU 版 PyTorch + 依赖
    └─ 底座说明.txt                           内容清单与用法（本脚本生成）

用法（在跑过本项目含提示词模型的 Windows 机器上，项目根目录）：
    python tools/build_runtime_base.py              # 全流程：组装 + 自检 + 压缩
    python tools/build_runtime_base.py --dry-run    # 只打印计划，不动手
    python tools/build_runtime_base.py --skip-zip   # 只组装目录，不压缩

前提（缺了会明确报错）：
1. 本机已有开发环境：out/third_party/WeVisOne-VL/.venv 可用（脚本从它读取
   依赖清单和 Python 版本，并复制它的基础解释器做便携运行时，保证版本一致；
   freeze 里的本地 wheel 直接安装原 wheel 文件，editable/VCS 安装反查
   direct_url.json 后从本地源码目录装非 editable 版，均不依赖原下载渠道）；
2. models/WeVisOne-VL-2B/ 权重已就位；models/inpainting_lama_2025jan.onnx 在；
3. 有网络（pip 下载其余依赖，约 4GB，仅本次需要；torch 家族若为普通 pin
   才走 --torch-index 的 GPU 源）。

配套：主程序包由 GitHub Actions 每次自动打（不含模型与环境）；收件人把
底座包和主程序包解压到同一目录即可运行。详见 docs/底座包打包说明.md。
"""
from __future__ import annotations

import argparse
import configparser
import datetime
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SRC_VENV = ROOT / "out" / "third_party" / "WeVisOne-VL" / ".venv"
LAMA = ROOT / "models" / "inpainting_lama_2025jan.onnx"
WEIGHTS = ROOT / "models" / "WeVisOne-VL-2B"
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
TORCH_FAMILY = ("torch", "torchvision", "torchaudio", "triton")
SKIP_PKGS = ("pip", "setuptools", "wheel")


def fail(msg: str) -> None:
    raise SystemExit(f"[底座] 错误：{msg}")


def run(cmd: list[str], **kw) -> None:
    print("[底座] 执行:", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def venv_python(venv: Path) -> Path:
    for rel in ("Scripts/python.exe", "bin/python"):
        p = venv / rel
        if p.is_file():
            return p
    fail(f"找不到源环境解释器：{venv}（先在这台机器上把 WeVisOne 开发环境装好）")


def _runtime_ignore(dirpath: str, names: list[str]) -> list[str]:
    """copytree 忽略规则：__pycache__/*.pyc 全排除；Lib/test 与 Lib/site-packages
    只在一级 Lib 下排除（shutil.ignore_patterns 只匹配基名，"Lib/test" 这种带
    路径的模式永远匹配不上，会静默失效）。"""
    skip = {"test", "site-packages"} if Path(dirpath).name == "Lib" else set()
    return [n for n in names if n in skip or n == "__pycache__" or n.endswith(".pyc")]


def base_interpreter(venv: Path) -> Path:
    """源 venv 的基础解释器目录（pyvenv.cfg 的 home），复制它做便携运行时。"""
    cfg = configparser.ConfigParser()
    cfg.read_string("[venv]\n" + (venv / "pyvenv.cfg").read_text(encoding="utf-8"))
    home = Path(cfg["venv"]["home"].strip())
    if "WindowsApps" in str(home):
        fail("源环境基于微软商店版 Python，复制出来无法运行；"
             "请先从 python.org 安装正式版 Python 并重建 WeVisOne venv")
    if not (home / "python.exe").is_file() and not (home / "python").is_file():
        fail(f"pyvenv.cfg 指向的基础解释器不存在：{home}")
    return home


def freeze(venv: Path) -> list[str]:
    out = subprocess.run([str(venv_python(venv)), "-m", "pip", "freeze"],
                         check=True, capture_output=True, text=True).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _file_url_path(ref: str) -> Path:
    """`file:///D:/...#sha256=...` 或裸路径 → 本地路径（去 fragment、URL 解码、补 Windows 盘符）。"""
    p = unquote(ref.split("#", 1)[0])
    if p.startswith("file://"):
        p = p[len("file://"):]
    if re.match(r"^/[A-Za-z]:[\\/]", p):  # file:///D:/... 形式去掉盘符前的斜杠
        p = p[1:]
    return Path(p)


def _editable_source_dir(name: str) -> Path | None:
    """从源 venv 的 dist-info/direct_url.json 反查 editable 安装的实际源码目录。"""
    site_packages = SRC_VENV / "Lib" / "site-packages"
    for dist in site_packages.glob(f"{name}-*.dist-info"):
        direct = dist / "direct_url.json"
        if direct.is_file():
            url = json.loads(direct.read_text(encoding="utf-8")).get("url", "")
            if url.startswith("file://"):
                path = _file_url_path(url)
                if path.is_dir():
                    return path
    return None


def split_requirements(lines: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    """拆成四组：torch 家族 pin（走 GPU 源）、torch 家族本地 wheel（直装）、
    本地路径包（editable/VCS 反查实际源码目录，如 wevisone_vl）、普通包。"""
    torch_pins, torch_wheels, local_pkgs, pypi = [], [], [], []
    for ln in lines:
        name = ln.split("==")[0].split("@")[0].strip().lower().replace("-", "_")
        if name in SKIP_PKGS:
            continue
        if ln.startswith("-e ") or " @ file://" in ln:
            if " @ file://" in ln:
                path = _file_url_path(ln.split(" @ ", 1)[1])
            else:
                ref = ln[3:].strip()
                if ref.startswith(("git+", "http://", "https://")):
                    # pip freeze 把本地 editable 安装显示成 VCS 来源（克隆时的 remote），
                    # 代理地址未必可达；反查源环境 direct_url.json 拿实际源码目录。
                    egg = re.search(r"[#&]egg=([\w.-]+)", ref)
                    name = egg.group(1).lower().replace("-", "_") if egg else name
                    path = _editable_source_dir(name) if egg else None
                else:
                    path = _file_url_path(ref)
            if path is None or not path.exists():
                fail(f"本地依赖来源无法解析：{ln}\n"
                     "       editable/VCS 安装需能在源环境 dist-info 里反查到本地源码目录")
            (torch_wheels if name in TORCH_FAMILY else local_pkgs).append(str(path))
        elif name in TORCH_FAMILY:
            torch_pins.append(ln)
        else:
            pypi.append(ln)
    return torch_pins, torch_wheels, local_pkgs, pypi


def main() -> None:
    # Windows 控制台默认 cp1252/GBK，打印中文会 UnicodeEncodeError；强制 UTF-8。
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="build_runtime_base", description="组装去水印底座包")
    ap.add_argument("--out", default=str(ROOT / "dist_base"), help="输出目录（默认 dist_base/）")
    ap.add_argument("--torch-index", default=TORCH_INDEX,
                    help="torch 家族普通 pin 的 GPU 下载源（freeze 为本地 wheel 时直装，不用此源）")
    ap.add_argument("--skip-zip", action="store_true", help="只组装目录不压缩")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划不执行")
    ns = ap.parse_args()

    for p, hint in ((LAMA, "把修复模型放进 models/"),
                    (WEIGHTS, "把 WeVisOne-VL-2B 权重目录放进 models/"),
                    (SRC_VENV, "先装好 WeVisOne 开发环境")):
        if not p.exists():
            fail(f"缺少 {p}——{hint}")
    home = base_interpreter(SRC_VENV)
    lines = freeze(SRC_VENV)
    torch_pins, torch_wheels, local_pkgs, pypi = split_requirements(lines)

    bundle = Path(ns.out) / "去水印"
    runtime_py = bundle / "wevisone_runtime" / "python"
    print(f"[底座] 计划：运行时复制自 {home}；torch 源 {ns.torch_index}；")
    print(f"[底座] 依赖 {len(pypi)} 个普通包、{len(torch_pins)} 个 torch pin、"
          f"{len(torch_wheels)} 个 torch 本地 wheel、{len(local_pkgs)} 个本地包："
          f"{[Path(p).name for p in torch_wheels + local_pkgs]}")
    if ns.dry_run:
        print("[底座] dry-run，未做任何修改")
        return

    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "models").mkdir(parents=True)

    print("[底座] 1/5 复制模型（约 5GB，耐心等）…")
    shutil.copy2(LAMA, bundle / "models" / LAMA.name)
    shutil.copytree(WEIGHTS, bundle / "models" / WEIGHTS.name)

    print(f"[底座] 2/5 复制便携 Python（来自 {home}）…")
    shutil.copytree(home, runtime_py, ignore=_runtime_ignore)
    py = runtime_py / ("python.exe" if sys.platform == "win32" else "python")
    # 复制时排除了 Lib/site-packages，基础解释器自带的 pip 不在其中，先用 ensurepip 补回
    run([py, "-m", "ensurepip", "--upgrade"])
    run([py, "-m", "pip", "install", "--no-warn-script-location",
         "--upgrade", "pip", "setuptools", "wheel"])

    print("[底座] 3/5 安装 GPU 版 torch 与其余依赖（约 4GB 下载）…")
    if torch_pins or torch_wheels:
        # --no-deps：依赖闭包全在 freeze 清单里，随普通包批次按钉版安装；
        # 也避免 pip 去 torch 专用源找 numpy/pillow 等公共依赖而失败
        run([py, "-m", "pip", "install", "--no-warn-script-location", "--no-deps",
             "--index-url", ns.torch_index] + torch_pins + torch_wheels)
    if pypi:
        run([py, "-m", "pip", "install", "--no-warn-script-location"] + pypi)
    for pkg in local_pkgs:
        run([py, "-m", "pip", "install", "--no-warn-script-location", pkg])

    print("[底座] 4/5 自检：导入 torch/transformers/wevisone_vl 并查显卡…")
    run([py, "-c", "import torch, transformers, wevisone_vl; "
                   "print('torch', torch.__version__, "
                   "'| CUDA 可用:', torch.cuda.is_available())"])

    (bundle / "底座说明.txt").write_text(
        "去水印底座包\n"
        f"生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M}（{Path.home().name}@{sys.platform}）\n\n"
        "内容：LaMa 修复模型、WeVisOne-VL-2B 提示词模型、便携 Python 运行时\n"
        "（含 GPU 版 PyTorch，依赖版本与打包机开发环境逐钉一致）。\n\n"
        "用法：本包与「主程序包」解压到同一位置，两个 zip 里都是「去水印」文件夹，\n"
        "会自动合并成一个目录，双击其中的 去水印.exe 即可。\n"
        "本底座长期不变；程序更新只需用新主程序包覆盖，不要删 models/ 与 wevisone_runtime/。\n",
        encoding="utf-8")

    print("[底座] 5/5 压缩…")
    if ns.skip_zip:
        print(f"[底座] 完成（未压缩）：{bundle}")
        return
    stamp = datetime.datetime.now().strftime("%Y%m%d")
    archive = shutil.make_archive(str(Path(ns.out) / f"去水印底座_{stamp}"), "zip",
                                  root_dir=ns.out, base_dir="去水印")
    size_gb = Path(archive).stat().st_size / 1e9
    print(f"[底座] 完成：{archive}（{size_gb:.1f}GB）")
    if size_gb > 1.9:
        print("[底座] 提示：超过 2GB，若要传 GitHub Release 请先用 7-Zip 分卷：\n"
              f'       7z a -v1900m 去水印底座_{stamp}.7z "{archive}"')


if __name__ == "__main__":
    main()
