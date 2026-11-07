"""提示词找水印功能的真实模型端到端自测。

用固定回归样本 core_009（斜向平铺文字）分别跑「自动识别」与「提示词定位」
两条真实 BatchWorker 路径，对照真值 alpha 计算选区指标，产物写入独立目录：
    out/提示词功能自测_20261107/

真实链路覆盖：wevisone_client 起停服务 → wevisone_server（WeVisOne venv）
→ 模型框 → prompt_detect 框内证据提取 → 闸门 → 修复 → 判定。

用法（GPU 机器）：
    set WM_WEVISONE_DEC_BATCH=4
    .venv/Scripts/python.exe tools/run_prompt_feature_selftest.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "out" / "提示词功能自测_20261107"
SAMPLE = ROOT / "data/watermark_regression_min_v1/inputs/core/core_009.png"
GT_ALPHA = ROOT / "data/watermark_regression_min_v1/references/core_009/alpha.png"


def encode_png(image: np.ndarray) -> str:
    import base64, io
    buffer = io.BytesIO()
    Image.fromarray(image[..., :3]).save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def metrics(selected: np.ndarray, alpha: np.ndarray) -> dict:
    gt = alpha > 0
    strong = alpha >= 128
    tp = int((selected & gt).sum())
    return {
        "selected_px": int(selected.sum()),
        "recall_alpha_gt0": round(tp / gt.sum(), 4),
        "recall_alpha_ge128": round(int((selected & strong).sum()) / strong.sum(), 4),
        "precision": round(tp / max(1, int(selected.sum())), 4),
        "missed_px": int((gt & ~selected).sum()),
        "extra_px": int((selected & ~gt).sum()),
    }


def run_mode(mode: str, query: str, service) -> dict:
    from wm.batches import Batch, BatchWorker
    from wm.contract import PageImage, Thresholds

    image = np.array(Image.open(SAMPLE).convert("RGB"))
    batch = Batch(f"selftest-{mode}", [PageImage(image, 96, "core_009")])
    batch.detection_mode = mode
    batch.prompt_query = query
    batch.params = {"mask_alpha_min": 8, "lama_context_margin": 256,
                    "lama_tile_overlap": 64, "lama_mask_dilate": 0, "close_px": 11}
    thr = Thresholds(mask_alpha_min=0.08, watermark_recall_priority=True)
    worker = BatchWorker([(batch, [0], thr, 11, True, None)],
                         service=service if mode == "prompt" else None)
    events = []
    worker.update.connect(lambda payload: events.append(payload.get("message"))
                          if payload.get("kind") == "stage" else None)
    # 同步 run() 时信号没有 GUI 接收方，需自行把结果落回批次（等价 app.queue_update）
    def collect(payload):
        if payload.get("kind") != "page":
            return
        if payload["result"] is not None:
            batch.results[payload["index"]] = payload["result"]
            batch.errors.pop(payload["index"], None)
        else:
            batch.results.pop(payload["index"], None)
            batch.errors[payload["index"]] = payload["error"]
    worker.update.connect(collect)
    started = time.perf_counter()
    worker.run()  # 同步跑，避免 Qt 事件循环依赖
    elapsed = round(time.perf_counter() - started, 1)
    result = batch.results.get(0)
    if result is None:
        return {"mode": mode, "error": batch.errors.get(0, "无结果"), "elapsed_s": elapsed}
    selected = result.mask.binary(0.08) > 0
    alpha = np.array(Image.open(GT_ALPHA).convert("L"))
    info = {
        "mode": mode, "query": query, "elapsed_s": elapsed,
        "outcome": worker.outcomes[0] if worker.outcomes else None,
        "detection": result.processing.get("prompt_detection"),
        "stage_messages": [m for m in events if m],
        "engine": str(result.verdict.engine),
        "repairs": [str(r) for r in result.verdict.repairs],
        **metrics(selected, alpha),
    }
    # 产物：选区叠加图（红）＋模型框（橙）＋输出
    overlay = image.copy()
    tint = np.array([255, 40, 40], np.uint8)
    sel = selected.astype(np.float32)[..., None]
    overlay = np.clip(overlay.astype(np.float32) * (1 - sel * 0.45) + tint * sel * 0.45,
                      0, 255).astype(np.uint8)
    if mode == "prompt" and result.mask.instances:
        import cv2
        for inst in result.mask.instances:
            x0, y0, x1, y1 = inst.bbox
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 159, 67), 2)
    Image.fromarray(overlay).save(OUT / f"{mode}_selection.png")
    Image.fromarray(result.output.image[..., :3]).save(OUT / f"{mode}_output.png")
    return info


def run_real010(service) -> dict:
    """真实软件 PDF（斜向黑字压正文）的提示词定位自检：尺度通道收回巨大水印笔画。"""
    from wm import documents, prompt_detect, repair
    from wm.contract import Thresholds
    import cv2

    pages = documents.load_pages(ROOT / "data/watermark_regression_min_v1/inputs/real/real_010.pdf", dpi=200)
    image = pages[0].image[..., :3]
    layer, info = prompt_detect.locate_strokes(image, '斜向黑字"内部演示稿"', service, Thresholds())
    selected = layer.footprint
    output, used = repair.repair(image.copy(), layer, Thresholds())
    overlay = image.copy()
    overlay[selected] = (0.5 * overlay[selected] + 0.5 * np.array([255, 40, 40])).astype(np.uint8)
    for box in info["model_boxes"]:
        cv2.rectangle(overlay, tuple(box[:2]), tuple(box[2:]), (255, 159, 67), 3)
    Image.fromarray(overlay).save(OUT / "real010_selection.png")
    Image.fromarray(output).save(OUT / "real010_output.png")
    return {
        "mode": "prompt", "sample": "real_010.pdf p1 (1654x2339)",
        "query": '斜向黑字"内部演示稿"', "treatment": "strokes",
        "model_boxes": info["model_boxes"], "selected_px": int(selected.sum()),
        "repairs": [str(u) for u in used],
        "note": "巨大斜向黑字压正文：尺度通道（开运算去细结构+近黑实芯）收回水印笔画，"
                "含水印伸出模型框的延伸；正文行开运算后不移除，交叠处轻微损伤",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from wm.wevisone_client import Service

    service = Service()
    if not Service.available():
        print("WeVisOne 环境不可用，无法跑提示词定位真实链路。")
        return 1
    started = time.perf_counter()
    service.start()
    print(f"服务就绪：{service.last_model}（{round(time.perf_counter()-started, 1)}s）")
    results = {}
    try:
        results["auto"] = run_mode("auto", "", service)
        print("自动识别完成:", json.dumps(results["auto"], ensure_ascii=False)[:200])
        results["prompt"] = run_mode("prompt", "验证素材 3409", service)
        print("提示词定位完成:", json.dumps(results["prompt"], ensure_ascii=False)[:200])
        results["real_010"] = run_real010(service)
    finally:
        service.stop()
    (OUT / "自测结果.json").write_text(
        json.dumps({"sample": "core_009", "date": "2026-11-07", "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {OUT/'自测结果.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
