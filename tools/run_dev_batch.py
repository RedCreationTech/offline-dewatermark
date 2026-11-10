"""以桌面 GUI 默认参数对指定图片目录跑一轮真实批次（自动识别模式）。

复刻 GUI「开始处理」的完整作业构造：documents.load_many 组建单一批次，
BatchWorker 走 PDF 结构检查 → 同尺寸组跨页先验 → 单图笔画识别 → 修复，
文字识别与笔画补全始终开启。推理只读取输入图片，不读取任何参考图、
alpha 真值或合成参数；评估类数据只能在推理完成后另行加载。

每张处理完成即落盘（原图／结果图／水印图／掩码），最后写 run.json 清单；
即使清单环节出错也不丢已完成的页面产物。

用法：
  python tools/run_dev_batch.py --input-dir out/某批次/输入 --out out/某批次/run1
  python tools/run_dev_batch.py --input-dir DIR --out OUT [--name 批次名] [--recursive]
  python tools/run_dev_batch.py --self-test   # 小图端到端自检（秒级）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from wm import documents, lama
from wm.app import make_app
from wm.batches import Batch, BatchWorker
from wm.contract import PageResult, Thresholds

# GUI 默认滑杆位置（见 wm/app.py KNOB_DEFAULTS / Thresholds 默认值）。
GUI_DEFAULTS = Thresholds(
    mask_alpha_min=0.08,
    lama_context_margin=256,
    lama_tile_overlap=64,
    lama_mask_dilate=0,
    watermark_recall_priority=True,  # 技术能力始终开启，与桌面一致
)
GUI_CLOSE_PX = 11  # KNOB_DEFAULTS["close_px"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    """只收集待测输入图片；默认只取顶层，避免把参考/真值子目录带进批次。"""
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file()]
    files = [p for p in files if p.suffix.lower() in documents.IMAGE_SUFFIXES]
    return sorted(files, key=lambda p: p.name)


def code_snapshot() -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "wm").glob("*.py"))}


def record_page(folder: Path, page, result: PageResult | None, error: str | None) -> dict:
    """单页产物落盘并返回清单记录；处理失败也保留原图与错误信息。"""
    folder.mkdir(parents=True, exist_ok=True)
    rec: dict = {"index": page.page_index, "doc_id": page.doc_id,
                 "source_path": str(Path(page.source_path or ""))}
    if page.source_path:
        shutil.copy2(page.source_path, folder / "input.png")
    if error:
        rec["error"] = error
    if result is not None:
        documents.save_png(folder / "output.png", result.output.image)
        mask_binary = result.mask.binary()
        documents.save_png(folder / "mask.png", mask_binary)
        if result.extracted_watermark is not None:
            documents.save_png(folder / "watermark.png", result.extracted_watermark)
        src, out_img = page.image, result.output.image
        changed = np.any(src != out_img, axis=2)
        outside = changed & (mask_binary == 0)
        rec.update(
            method=result.mask.method,
            repair_mode=result.mask.repair_mode,
            mask_pixels=int((mask_binary > 0).sum()),
            mask_coverage=round(float((mask_binary > 0).mean()), 6),
            changed_pixels=int(changed.sum()),
            changed_outside_mask_pixels=int(outside.sum()),
            selection_layer=result.mask.selection is not None,
            verdict=result.verdict.to_dict(),
            processing={k: v for k, v in result.processing.items() if k != "thresholds"},
            thresholds=result.processing.get("thresholds"),
        )
    return rec


def run_batch(input_dir: Path, out_dir: Path, name: str | None, recursive: bool,
              mode: str = "auto", prompt: str | None = None) -> dict:
    images = collect_images(input_dir, recursive)
    if not images:
        raise SystemExit(f"目录中没有可处理图片：{input_dir}")

    # 与桌面 GUI 相同：一批手工挑选的同水印图片 → 共享跨页先验。
    pages = documents.load_many(images, dpi=200)
    batch = Batch(name or input_dir.name, pages)
    batch.detection_mode = mode  # 找水印方式：auto＝自动识别；prompt＝提示词定位
    if mode == "prompt":
        if not prompt:
            raise SystemExit("提示词模式需要 --prompt")
        batch.prompt_query = prompt
        batch.prompt_treatment = "strokes"  # 按笔画（GUI 默认）

    make_app()  # 与 GUI 同一 Qt 环境；worker 同步执行，不开事件循环

    pages_dir = out_dir / "pages"
    records: list[dict] = []
    errors: list[str] = []

    def on_event(event: dict) -> None:
        if event.get("kind") != "page":
            return
        index = event["index"]
        page = batch.pages[index]
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in page.doc_id)[:100] or f"page{index:02d}"
        try:
            records.append(record_page(pages_dir / f"{index + 1:02d}_{safe}", page,
                                       event.get("result"), event.get("error")))
        except Exception as exc:  # 单页落盘失败不拖垮整批
            errors.append(f"{page.doc_id}: {exc}")

    frozen = code_snapshot()
    started = time.perf_counter()
    if mode == "prompt":
        # 本机 8GB 显存下常驻服务跨页累积分配会 OOM（实测前 12 页连续失败、
        # 每页独立服务则稳定成功）：提示词模式逐页启停模型服务，每页独立进程。
        from wm.wevisone_client import Service
        if not Service.available():
            raise SystemExit("提示词定位不可用：未找到 WeVisOne 环境（out/third_party/WeVisOne-VL）")
        for index in range(len(pages)):
            service = Service()
            job = (batch, [index], GUI_DEFAULTS, GUI_CLOSE_PX, True, batch.style)
            worker = BatchWorker([job], None, record_batch_timing=True, service=service)
            worker.update.connect(on_event)
            worker.run()
            service.stop()
    else:
        indices = list(range(len(pages)))
        job = (batch, indices, GUI_DEFAULTS, GUI_CLOSE_PX, True, batch.style)
        worker = BatchWorker([job], None, record_batch_timing=True)
        worker.update.connect(on_event)
        worker.run()
    elapsed = time.perf_counter() - started
    if frozen != code_snapshot():
        raise SystemExit("运行期间 wm/ 源码发生变化，本轮结果不可追溯，已中止保存。")
    if errors:
        raise SystemExit("页面落盘失败：\n" + "\n".join(errors))

    records.sort(key=lambda r: r["index"])
    manifest = {
        "schema_version": 1,
        "runner": Path(__file__).name,
        "batch_name": batch.name,
        "detection_mode": batch.detection_mode,
        "prompt_query": batch.prompt_query,
        "gui_defaults": {
            "thresholds": {k: getattr(GUI_DEFAULTS, k) for k in (
                "mask_alpha_min", "lama_context_margin", "lama_tile_overlap",
                "lama_mask_dilate", "watermark_recall_priority")},
            "close_px": GUI_CLOSE_PX,
            "use_ocr": True,
            "style": None,
            "automatic_selection": False,
        },
        "input_dir": str(input_dir),
        "inputs": [{"name": p.name, "sha256": sha256(p)} for p in images],
        "dpi": 200,
        "total_pages": len(pages),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "outcomes": worker.outcomes,
        "elapsed_seconds": round(elapsed, 3),
        "code_snapshot": frozen,
        "lama_model_sha256": sha256(Path(lama._resolve_model_path())),
        "records": records,
    }
    (out_dir / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def print_summary(manifest: dict) -> None:
    failed = sum(1 for r in manifest["records"] if "error" in r)
    print(f"批次：{manifest['batch_name']}；页数：{manifest['total_pages']}；"
          f"失败：{failed}；耗时：{manifest['elapsed_seconds']:.1f}s")
    for r in manifest["records"]:
        status = "失败" if "error" in r else (
            "通过" if not r.get("verdict", {}).get("failures") else "未通过")
        print(f"  [{r['index'] + 1:02d}] {r['doc_id']}: {status} "
              f"method={r.get('method', '—')} mask_px={r.get('mask_pixels', 0):,} "
              f"changed_outside={r.get('changed_outside_mask_pixels', 0):,}")


def self_test() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="wm_dev_batch_selftest_"))
    try:
        input_dir = tmp / "inputs"
        out_dir = tmp / "run"
        input_dir.mkdir()
        for i, color in enumerate(((252, 252, 250), (250, 250, 252))):
            Image.fromarray(np.full((96, 128, 3), color, np.uint8)).save(input_dir / f"t{i}.png")
        manifest = run_batch(input_dir, out_dir, "self-test", recursive=False)
        assert len(manifest["records"]) == 2, "应有两条记录"
        for r in manifest["records"]:
            assert "verdict" in r or "error" in r, f"记录缺 verdict/error: {r['doc_id']}"
            assert (out_dir / "pages").is_dir()
        print_summary(manifest)
        print("SELF-TEST OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="以 GUI 默认参数跑一轮真实批次（自动识别模式）")
    parser.add_argument("--input-dir", type=Path, help="待测输入图片目录")
    parser.add_argument("--out", type=Path, help="本轮产物输出目录（新建，不覆盖旧轮）")
    parser.add_argument("--name", default=None, help="批次名（默认取输入目录名）")
    parser.add_argument("--recursive", action="store_true", help="递归收集子目录图片（默认只取顶层）")
    parser.add_argument("--mode", choices=("auto", "prompt"), default="auto",
                        help="找水印方式：auto＝自动识别（默认）；prompt＝提示词定位（需 --prompt）")
    parser.add_argument("--prompt", default=None, help="提示词模式的水印描述（如：斜向平铺的内部资料文字水印）")
    parser.add_argument("--self-test", action="store_true", help="小图端到端自检后退出")
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(self_test())
    if not args.input_dir or not args.out:
        raise SystemExit("需要 --input-dir 与 --out（或 --self-test）")

    input_dir = args.input_dir.resolve()
    out_dir = args.out.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{input_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise SystemExit(f"输出目录已存在且非空（每轮实验用新目录）：{out_dir}")
    # 同一秒启动的重复实例会拿到同一输出目录并竞争 GPU（本机已实测双双起服务
    # 导致 OOM）：用原子创建锁保证单实例，第二个实例立即退出。
    lock_path = out_dir / ".running.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        raise SystemExit(f"输出目录已有活动任务锁（{out_dir}），本实例退出以防止双实例竞争。")

    try:
        manifest = run_batch(input_dir, out_dir, args.name, args.recursive,
                             mode=args.mode, prompt=args.prompt)
    finally:
        lock_path.unlink(missing_ok=True)
    print_summary(manifest)
    print(f"产物：{out_dir}")
    if any("error" in r for r in manifest["records"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
