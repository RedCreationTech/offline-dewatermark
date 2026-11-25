"""去水印底座包构建脚本：在 GPU 机器上一次性打出「不变的大件」。

底座包内容（约 7~8GB，打好后长期使用，代码更新不需要重打）：
    去水印底座/
    ├─ models/inpainting_lama_2025jan.onnx   LaMa 修复模型
    ├─ models/WeVisOne-VL-2B/                提示词定位大模型权重
    ├─ wevisone_runtime/python/              便携 Python + GPU 版 PyTorch + 依赖
    └─ 底座说明.txt                           内容清单与用法（本脚本生成）

用法（在配有 RTX 显卡的 Windows 打包机上，项目根目录）：
    python tools/build_runtime_base.py              # 全流程：组装 + 自检 + 压缩
    python tools/build_runtime_base.py --dry-run    # 只打印计划，不动手
    python tools/build_runtime_base.py --skip-zip   # 只组装目录，不压缩

前提（缺了会明确报错）：
1. 本机已有开发环境：out/third_party/WeVisOne-VL/.venv 可用（脚本从它读取
   依赖清单和 Python 版本，并复制它的基础解释器做便携运行时，保证版本一致）；
2. models/WeVisOne-VL-2B/ 权重已就位；models/inpainting_lama_2025jan.onnx 在；
3. 有网络（pip 下载 torch GPU 版及其余依赖，约 4GB，仅本次需要）。

配套：主程序包由 GitHub Actions 每次自动打（不含模型与环境）；收件人把
底座包和主程序包解压到同一目录即可运行。详见 docs/底座包打包说明.md。
"""
from __future__ import annotations

import argparse
import configparser
import datetime
import shutil
import subprocess
import sys
from pathlib import Path

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


def split_requirements(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    """拆成三组：torch 家族（走 GPU 源）、本地路径包（如 wevisone_vl）、普通包。"""
    torch_pins, local_pkgs, pypi = [], [], []
    for ln in lines:
        name = ln.split("==")[0].split("@")[0].strip().lower().replace("-", "_")
        if name in SKIP_PKGS:
            continue
        if ln.startswith("-e ") or " @ file://" in ln:
            path = Path(ln.split("=")[-1].replace("file://", "").strip("/")
                        if " @ file://" in ln else ln[3:].strip())
            if not path.is_absolute():
                path = Path("/") / path  # file:///C:/... 形式
            (local_pkgs if path.exists() else pypi).append(str(path) if path.exists() else ln)
        elif name in TORCH_FAMILY:
            torch_pins.append(ln)
        else:
            pypi.append(ln)
    return torch_pins, local_pkgs, pypi


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_runtime_base", description="组装去水印底座包")
    ap.add_argument("--out", default=str(ROOT / "dist_base"), help="输出目录（默认 dist_base/）")
    ap.add_argument("--torch-index", default=TORCH_INDEX, help="torch GPU 版下载源")
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
    torch_pins, local_pkgs, pypi = split_requirements(lines)

    bundle = Path(ns.out) / "去水印底座"
    runtime_py = bundle / "wevisone_runtime" / "python"
    print(f"[底座] 计划：运行时复制自 {home}；torch 源 {ns.torch_index}；")
    print(f"[底座] 依赖 {len(pypi)} 个普通包、{len(torch_pins)} 个 torch 家族、"
          f"{len(local_pkgs)} 个本地包：{[Path(p).name for p in local_pkgs]}")
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
    shutil.copytree(home, runtime_py, ignore=shutil.ignore_patterns(
        "*.pyc", "__pycache__", "Lib/test", "Lib/site-packages"))
    py = runtime_py / ("python.exe" if sys.platform == "win32" else "python")
    run([py, "-m", "pip", "install", "--no-warn-script-location",
         "--upgrade", "pip", "setuptools", "wheel"])

    print("[底座] 3/5 安装 GPU 版 torch 与其余依赖（约 4GB 下载）…")
    if torch_pins:
        run([py, "-m", "pip", "install", "--no-warn-script-location",
             "--index-url", ns.torch_index] + torch_pins)
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
        "用法：把「主程序包」解压到本目录（去水印.exe 与本说明同级），双击即用。\n"
        "本底座长期不变；程序更新只需覆盖主程序包，不要删 models/ 与 wevisone_runtime/。\n",
        encoding="utf-8")

    print("[底座] 5/5 压缩…")
    if ns.skip_zip:
        print(f"[底座] 完成（未压缩）：{bundle}")
        return
    stamp = datetime.datetime.now().strftime("%Y%m%d")
    archive = shutil.make_archive(str(Path(ns.out) / f"去水印底座_{stamp}"), "zip",
                                  root_dir=ns.out, base_dir="去水印底座")
    size_gb = Path(archive).stat().st_size / 1e9
    print(f"[底座] 完成：{archive}（{size_gb:.1f}GB）")
    if size_gb > 1.9:
        print("[底座] 提示：超过 2GB，若要传 GitHub Release 请先用 7-Zip 分卷：\n"
              f'       7z a -v1900m 去水印底座_{stamp}.7z "{archive}"')


if __name__ == "__main__":
    main()
