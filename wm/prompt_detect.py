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

import cv2
import numpy as np

from . import masks, recall
from .contract import MaskLayer, MaskSource, Thresholds, WatermarkInstance

METHOD = "prompt_located_strokes"
BOX_PAD = 64
MIN_BOX = 8


def _contrast_seed(crop: np.ndarray, domain: np.ndarray, height: float) -> np.ndarray:
    """检测器在纯淡笔画框内可能无种子时的兜底种子：框内局部对比响应最高的一批像素。

    仅用于让 _pixel_strokes 有起点；最终选区仍由证据生长决定。
    """
    smooth = cv2.GaussianBlur(crop.astype(np.float32), (0, 0), .8)
    gray = smooth.mean(2)
    channels = [gray]
    channels.extend(smooth[..., c] - (smooth.sum(2) - smooth[..., c]) / 2 for c in range(3))
    kernel = np.ones((min(31, max(9, int(.35 * height) | 1)),) * 2, np.uint8)
    best = None
    for channel in channels:
        for operation in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
            feature = cv2.morphologyEx(channel, operation, kernel)
            best = feature if best is None else np.maximum(best, feature)
    values = best[domain]
    if values.size == 0:
        return np.zeros(crop.shape[:2], bool)
    cut = float(np.percentile(values, 97))
    return domain & (best >= max(cut, 6.0))


def _clamp_box(box, width: int, height: int) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, x1 = sorted((max(0, x0), min(width, x1)))
    y0, y1 = sorted((max(0, y0), min(height, y1)))
    if x1 - x0 < MIN_BOX or y1 - y0 < MIN_BOX:
        return None
    return x0, y0, x1, y1


def locate_strokes(image: np.ndarray, query: str, service,
                     thr: Thresholds, *, close_px: int = 11,
                     use_ocr: bool = True,
                     treatment: str = "strokes") -> tuple[MaskLayer, dict]:
    """Run prompt detection and return (mask_layer, info).

    service 只需实现 segment(image_rgb, query) -> [xyxy...]，测试可传假实现。
    treatment="strokes"（默认）按笔画：模型框限定搜索范围，框内所有具备
    局部对比证据的连通笔画都进选区（与补全笔画同一套生长逻辑，非种子的
    淡笔画由强核连通生长加入，证据不足的不加）。
    treatment="box"（整框重绘）把模型框直接作为重绘许可选区，适用于框内
    基本是空白底的水印；框压正文时正文会被一并重绘，调用方应先提示用户。
    """
    if not query or not query.strip():
        raise ValueError("提示词为空，请输入要去除的水印文字描述。")
    started = perf_counter()
    raw_boxes = service.segment(image_rgb=image[..., :3], query=query.strip())
    height, width = image.shape[:2]
    boxes = [b for b in (_clamp_box(raw, width, height) for raw in raw_boxes) if b]
    info = {
        "detector": METHOD,
        "query": query.strip(),
        "treatment": treatment,
        "model_boxes": [list(b) for b in raw_boxes],
        "boxes_with_support": 0,
        "text_overlap_ratio": 0.0,
    }
    if not boxes:
        return _empty_layer(height, width), info
    if treatment == "box":
        return _box_layer(image, boxes, info, started)
    settings = replace(thr, lama_mask_dilate=0)  # 与圈选一致：预览/许可范围不外扩
    alpha = np.zeros((height, width), np.float32)
    paper: np.ndarray | None = None
    keepout = np.zeros((height, width), bool)
    instances: list[WatermarkInstance] = []
    confidences: list[float] = []
    supported_boxes = 0
    for index, (x0, y0, x1, y1) in enumerate(boxes):
        left, top = max(0, x0 - BOX_PAD), max(0, y0 - BOX_PAD)
        right, bottom = min(width, x1 + BOX_PAD), min(height, y1 + BOX_PAD)
        crop = image[top:bottom, left:right]
        local = masks.estimate_single_page(crop, settings, close_px=close_px,
                                           use_ocr=use_ocr, style=None)
        domain = np.zeros(crop.shape[:2], bool)
        domain[y0 - top:y1 - top, x0 - left:x1 - left] = True
        protected = local.text_keepout if local.text_keepout is not None else \
            np.zeros(crop.shape[:2], bool)
        grow_domain = domain & ~protected  # 正文保护区只约束水印选区，不生长
        seed = (local.binary(settings.mask_alpha_min) > 0) & grow_domain
        box_h = max(12., min(x1 - x0, y1 - y0) / 3.)
        if not seed.any():
            seed = _contrast_seed(crop, grow_domain, box_h)
        # 由强核出发做连通生长：淡笔画体随强核整体入选，证据不足的像素不进。
        grown = recall._pixel_strokes(crop, grow_domain, seed, box_h)
        if seed.any() and grown.any():
            # 只保留含种子的连通域：框内照片颗粒、纹理等独立强对比区域不随弱连接入选。
            count, labels = cv2.connectedComponents(grown.astype(np.uint8), 8)
            hit = np.unique(labels[seed & grown])
            hit = hit[hit > 0]
            grown = grown & np.isin(labels, hit) if hit.size else np.zeros_like(grown)
        chosen = grown & grow_domain
        if not chosen.any():
            continue
        supported_boxes += 1
        confidences.append(float(local.confidence))
        stamp = np.zeros((height, width), np.float32)
        stamp[top:bottom, left:right] = np.where(chosen, np.maximum(local.alpha, 0.09), 0.)
        alpha = np.maximum(alpha, stamp)
        if local.paper is not None:
            fill = float(np.median(local.paper))
            if paper is None:
                paper = np.full((height, width) + local.paper.shape[2:], fill, np.float32)
            paper[top:bottom, left:right] = local.paper
        if local.text_keepout is not None:
            keepout[top:bottom, left:right] |= local.text_keepout
        instances.append(WatermarkInstance(
            instance_id=index + 1, bbox=(x0, y0, x1, y1), pixel_count=int(chosen.sum()),
            mean_alpha=float(np.maximum(local.alpha, 0.09)[chosen].mean()),
            style_signature="prompt", keep=True))
    selected = alpha >= thr.mask_alpha_min
    layer = MaskLayer(
        alpha, source=MaskSource.MODEL, confidence=min(confidences) if confidences else 0.0,
        veil_strength=np.ones(3, np.float32), veil_offset=np.zeros(3, np.float32),
        paper=paper, instances=instances, footprint=selected.copy(),
        repair_mode="inpaint", method=METHOD,
        text_keepout=keepout if keepout.any() else None)
    info.update({
        "boxes_with_support": supported_boxes,
        "selected_pixels": int(selected.sum()),
        "service_ms": round((perf_counter() - started) * 1000),
        "service_model": getattr(service, "last_model", None),
    })
    return layer, info


def _empty_layer(height: int, width: int) -> MaskLayer:
    return MaskLayer(np.zeros((height, width), np.float32), source=MaskSource.MODEL,
                     confidence=0.0, veil_strength=np.ones(3, np.float32),
                     veil_offset=np.zeros(3, np.float32), method=METHOD,
                     repair_mode="inpaint", footprint=np.zeros((height, width), bool))


def _box_layer(image: np.ndarray, boxes: list[tuple[int, int, int, int]],
               info: dict, started: float) -> tuple[MaskLayer, dict]:
    """整框重绘：模型框整体作为重绘许可，框内正文会被一并重绘。"""
    from . import ocr as _ocr

    height, width = image.shape[:2]
    selection = np.zeros((height, width), bool)
    for x0, y0, x1, y1 in boxes:
        selection[y0:y1, x0:x1] = True
    zones = _ocr.text_zones(image[..., :3])
    if zones is not None:
        overlap = float((selection & zones).sum() / max(1, selection.sum()))
    else:
        overlap = 0.0
    layer = MaskLayer(selection.astype(np.float32), source=MaskSource.MODEL,
                      confidence=0.8, veil_strength=np.ones(3, np.float32),
                      veil_offset=np.zeros(3, np.float32),
                      footprint=selection.copy(), repair_mode="inpaint",
                      method="prompt_box_repaint",
                      instances=[WatermarkInstance(
                          instance_id=i + 1, bbox=box, pixel_count=int(selection[box[1]:box[3], box[0]:box[2]].sum()),
                          mean_alpha=1.0, style_signature="prompt_box", keep=True)
                          for i, box in enumerate(boxes)])
    info.update({
        "boxes_with_support": len(boxes),
        "selected_pixels": int(selection.sum()),
        "text_overlap_ratio": round(overlap, 4),
        "service_ms": round((perf_counter() - started) * 1000),
        "service_model": None,
    })
    return layer, info
