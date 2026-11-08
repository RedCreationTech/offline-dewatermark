"""Prompt-located watermark detection: WeVisOne boxes grant search ranges only.

The model's boxes never become repair pixels. Each box is widened 64 px and the
existing pixel-evidence pipeline re-detects inside it (estimate_single_page +
recall._pixel_strokes), exactly like the manual rectangle search in assisted.py.
Strokes found inside boxes merge into one MaskLayer with source=MODEL; the
merged layer uses the conservative inpaint route because per-box measured veil
fields cannot be combined into one validated physical model.

The model's matte (the prompt-qualified watermark soft mask) is used as a search
range constraint: growth and seeds are confined to the matte support, so strong
background content inside the box (printed lines, tables) cannot win the
evidence competition. matte is never the repair mask itself — every selected
pixel still needs local contrast evidence, and the matte halo without evidence
is dropped. Boxes are kept as WatermarkInstance entries for the canvas overlay
and export evidence.
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import cv2
import numpy as np

from . import masks, ocr, recall
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
    # 密度要足以覆盖笔画主体：背景对照区 = 域内未进种子的部分；
    # 种子太稀时笔画本身会污染背景参照，导致阈值爆炸、生长为空。
    cut = float(np.percentile(values, 85))
    return domain & (best >= max(cut, 8.0))


def _ink_mask(crop: np.ndarray, domain: np.ndarray, height: float) -> np.ndarray:
    """框内墨迹二值：局部对比响应经 Otsu 分出"墨迹 vs 纸面"双峰。"""
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
    scaled = np.clip(best / max(1.0, float(best.max())) * 255.0, 0, 255).astype(np.uint8)
    values = scaled[domain]
    if values.size == 0:
        return np.zeros(domain.shape, bool)
    cut, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return domain & (scaled >= max(float(cut), 24.0))


def _scale_channel(image: np.ndarray, box: tuple[int, int, int, int],
                   height: float) -> np.ndarray:
    """尺度通道：与正文小字尺度完全不同的巨大水印笔画。

    巨大且不透明的水印笔画（如整页斜向黑字）在形态学带通里响应不稳，
    但经"开运算去细结构"后，连通域尺度上与正文小字完全分开：正文行
    半宽约 2–3px 被 9px 椭圆核开运算移除，大水印笔画（半宽 ≥6px）保留。
    只保留与模型框相交的连通域（框内水印及其笔画延伸），框外大标题等
    大字不入选。这是对证据生长通道的补充。
    """
    # 实心墨迹：按纸面亮度阈值取暗区域（黑帽只响应边缘，粗笔画内部无响应，
    # 会导致开运算清空）。纸面亮度取全页高分位，墨迹阈值 = 纸面 - 50。
    gray = cv2.GaussianBlur(image.astype(np.float32), (0, 0), .8).mean(2)
    paper = float(np.percentile(gray, 90))
    if paper - 50. <= 60.:
        return np.zeros(image.shape[:2], bool)
    ink = gray <= (paper - 50.)
    if not ink.any():
        return np.zeros(image.shape[:2], bool)
    opened = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    if not opened.any():
        return np.zeros(image.shape[:2], bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened.astype(np.uint8), 8)
    x0, y0, x1, y1 = box
    keep = []
    for i in range(1, count):
        bx, by = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if not (bx < x1 and bx + bw > x0 and by < y1 and by + bh > y0):
            continue
        # 只留"近黑实芯"的大块：水印大字通常近黑；织物暗纹、灰色底纹
        # 虽是深色连通域但不是近黑，不得入选。
        mean_gray = float(cv2.mean(gray, (labels == i).astype(np.uint8))[0])
        if mean_gray <= paper - 90.:
            keep.append(i)
    if not keep:
        return np.zeros(image.shape[:2], bool)
    selected = np.isin(labels, keep)
    # 开运算后正文行已移除；再加一道宽度闸防细结构残留（半宽≥4px 才算笔画），
    # 随后外扩 5px 还原开运算收缩并盖住抗锯齿边缘。
    width = cv2.distanceTransform(selected.astype(np.uint8), cv2.DIST_L2, 5)
    selected &= width >= 4.
    if not selected.any():
        return selected
    return cv2.dilate(selected.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)


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
                     treatment: str = "strokes",
                     remove_overlap: bool = False) -> tuple[MaskLayer, dict]:
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
    raw_instances = service.segment(image_rgb=image[..., :3], query=query.strip())
    height, width = image.shape[:2]
    boxes = []
    supports: list[np.ndarray | None] = []
    for raw_box, matte in raw_instances:
        clamped = _clamp_box(raw_box, width, height)
        if clamped is None:
            continue
        boxes.append(clamped)
        support = None
        if matte is not None and matte.shape[:2] == (height, width):
            # matte＝提示词限定的水印软遮罩，只作搜索范围：外扩 4px 衔接抗锯齿边缘
            base = matte if matte.dtype == bool else (matte >= 128)
            if base.any():
                support = cv2.dilate(base.astype(np.uint8),
                                     np.ones((9, 9), np.uint8)).astype(bool)
        supports.append(support)
    info = {
        "detector": METHOD,
        "query": query.strip(),
        "treatment": treatment,
        "remove_overlap": remove_overlap,
        "model_boxes": [list(b) for b, _ in raw_instances],
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
    for index, ((x0, y0, x1, y1), support) in enumerate(zip(boxes, supports)):
        left, top = max(0, x0 - BOX_PAD), max(0, y0 - BOX_PAD)
        right, bottom = min(width, x1 + BOX_PAD), min(height, y1 + BOX_PAD)
        crop = image[top:bottom, left:right]
        local = masks.estimate_single_page(crop, settings, close_px=close_px,
                                           use_ocr=use_ocr, style=None)
        domain = np.zeros(crop.shape[:2], bool)
        domain[y0 - top:y1 - top, x0 - left:x1 - left] = True
        if use_ocr and crop.shape[1] >= 200 and crop.shape[0] >= 60:
            # estimate_single_page 不回传 text_keepout，直接取 OCR 正文区。
            # 保护区默认约束选区生长；remove_overlap（用户显式开启）时框内
            # 不再受保护区限制——按项目边界"允许牺牲正文/背景保护"的口径，
            # 水印笔画全覆盖去除，交叠文字由 LaMa 重绘。
            zones = ocr.text_zones(crop[..., :3])
            protected = zones if (zones is not None and not remove_overlap) else \
                np.zeros(crop.shape[:2], bool)
        else:
            protected = np.zeros(crop.shape[:2], bool)
        grow_domain = domain & ~protected  # 正文保护区约束选区生长
        if support is not None:
            support_crop = support[top:bottom, left:right]
            grow_domain &= support_crop  # matte 搜索范围约束：背景不参与
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
        # 尺度通道补充（全页墨迹、开运算去细结构、只留与模型框相交的大块）：
        # 证据生长对"巨大且不透明的水印笔画"响应不稳时，靠笔画尺度收回水印
        # （含水印伸出模型框的笔画延伸；正文行已被开运算移除）。
        scale_sel = _scale_channel(image, (x0, y0, x1, y1), box_h)
        chosen = (grown & grow_domain) | scale_sel[top:bottom, left:right]
        if remove_overlap:
            # 墨迹连续性扩展：remove_overlap 是用户显式的"全覆盖去除"选择，
            # 把与已选笔画相连的墨迹（典型为穿过正文行的笔画段）并入选区；
            # 无关的正文行不与已选笔画相连，不受影响。
            ink_here = (crop.astype(np.float32).mean(2) <=
                        (float(np.median(crop.astype(np.float32).mean(2))) - 50.))
            link_domain = domain & (support_crop if support is not None else True)
            for _ in range(3):
                spread = cv2.dilate(chosen.astype(np.uint8),
                                    np.ones((7, 7), np.uint8)).astype(bool)
                add = spread & ink_here & link_domain & ~chosen
                if not add.any():
                    break
                chosen |= add
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
        keepout[top:bottom, left:right] |= protected
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
