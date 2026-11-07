"""Prompt-located watermark detection: WeVisOne boxes grant search ranges only.

The model's boxes never become repair pixels. Each box is widened 64 px and the
existing pixel-evidence pipeline re-detects inside it (estimate_single_page +
recall._pixel_strokes), exactly like the manual rectangle search in assisted.py.
Strokes found inside boxes merge into one MaskLayer with source=MODEL; the
merged layer uses the conservative inpaint route because per-box measured veil
fields cannot be combined into one validated physical model.

matte is not requested, not transferred and not displayed (design Q3); boxes are
kept as WatermarkInstance entries for the canvas overlay and export evidence.
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from . import masks, recall
from .contract import MaskLayer, MaskSource, Thresholds, WatermarkInstance

METHOD = "prompt_located_strokes"
BOX_PAD = 64
MIN_BOX = 8


def _clamp_box(box, width: int, height: int) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, x1 = sorted((max(0, x0), min(width, x1)))
    y0, y1 = sorted((max(0, y0), min(height, y1)))
    if x1 - x0 < MIN_BOX or y1 - y0 < MIN_BOX:
        return None
    return x0, y0, x1, y1


def locate_strokes(image: np.ndarray, query: str, service,
                     thr: Thresholds, *, close_px: int = 11,
                     use_ocr: bool = True) -> tuple[MaskLayer, dict]:
    """Run prompt detection and return (mask_layer, info).

    service 只需实现 segment(image_rgb, query) -> [xyxy...]，测试可传假实现。
    """
    if not query or not query.strip():
        raise ValueError("提示词为空，请输入要去除的水印文字描述。")
    started = perf_counter()
    raw_boxes = service.segment(image_rgb=image[..., :3], query=query.strip())
    height, width = image.shape[:2]
    settings = replace(thr, lama_mask_dilate=0)  # 与圈选一致：预览/许可范围不外扩
    alpha = np.zeros((height, width), np.float32)
    paper: np.ndarray | None = None
    keepout = np.zeros((height, width), bool)
    instances: list[WatermarkInstance] = []
    confidences: list[float] = []
    supported_boxes = 0
    for index, raw in enumerate(raw_boxes):
        clamped = _clamp_box(raw, width, height)
        if clamped is None:
            continue
        x0, y0, x1, y1 = clamped
        left, top = max(0, x0 - BOX_PAD), max(0, y0 - BOX_PAD)
        right, bottom = min(width, x1 + BOX_PAD), min(height, y1 + BOX_PAD)
        crop = image[top:bottom, left:right]
        local = masks.estimate_single_page(crop, settings, close_px=close_px,
                                           use_ocr=use_ocr, style=None)
        domain = np.zeros(crop.shape[:2], bool)
        domain[y0 - top:y1 - top, x0 - left:x1 - left] = True
        seed = local.binary(settings.mask_alpha_min) > 0
        box_h = max(12., min(x1 - x0, y1 - y0) / 3.)
        supported = recall._pixel_strokes(crop, domain, seed & domain, box_h)
        chosen = seed & supported & domain
        if not chosen.any():
            continue
        supported_boxes += 1
        confidences.append(float(local.confidence))
        stamp = np.zeros((height, width), np.float32)
        stamp[top:bottom, left:right] = np.where(chosen, local.alpha, 0.)
        alpha = np.maximum(alpha, stamp)
        if local.paper is not None:
            fill = float(np.median(local.paper))
            if paper is None:
                paper = np.full((height, width) + local.paper.shape[2:], fill, np.float32)
            paper[top:bottom, left:right] = local.paper
        if local.text_keepout is not None:
            keepout[top:bottom, left:right] |= local.text_keepout
        instances.append(WatermarkInstance(
            instance_id=index + 1, bbox=clamped, pixel_count=int(chosen.sum()),
            mean_alpha=float(local.alpha[chosen].mean()) if chosen.any() else 0.0,
            style_signature="prompt", keep=True))
    selected = alpha >= thr.mask_alpha_min
    layer = MaskLayer(
        alpha, source=MaskSource.MODEL, confidence=min(confidences) if confidences else 0.0,
        veil_strength=np.ones(3, np.float32), veil_offset=np.zeros(3, np.float32),
        paper=paper, instances=instances, footprint=selected.copy(),
        repair_mode="inpaint", method=METHOD,
        text_keepout=keepout if keepout.any() else None)
    info = {
        "detector": METHOD,
        "query": query.strip(),
        "model_boxes": [list(b) for b in raw_boxes],
        "boxes_with_support": supported_boxes,
        "selected_pixels": int(selected.sum()),
        "service_ms": round((perf_counter() - started) * 1000),
        "service_model": getattr(service, "last_model", None),
    }
    return layer, info
