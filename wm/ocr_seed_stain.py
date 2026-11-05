"""OCR 单行小字种子 → 局部染色实测通道（ocr_seed_stain）——隔离实验候选。

种子范围：日期/时间样式的 OCR 行（正则定义，见 _TIMESTAMP_RE）——浅灰
**真实文字**（如免责声明）与浅灰水印字在色调上不可分，只有语义模式能
区分；本候选只承诺时间戳类，其他浅灰小字一律不动（已登记为边界）。

适用场景（2026-11-05 探针结论）：高纸面占比（≥65%）文档页上，OCR 能读出
日期时间戳那行小字（如 document_029–032：一轮读出、命中真值），但既有通道
不消费"单行小字 OCR 框"——repeated_photo/photo_text 只跑照片底，
document_overlay 只靠对比度、看不见 α≈0.01–0.02 的亚阈值染色。本通道
补上这一环：

- OCR 框**只作搜索范围**：框内/邻域的最终掩码必须来自实测染色证据
  （纸面参照下 0.9–14 级的中性偏暗笔画，且不得含深色墨迹——深色=真实
  正文/脚注，整框拒绝）；
- 不做闭运算/填洞/凸包：掩码沿实测笔画分布，字内空洞与字间空白保持；
- 修复是确定性代数逆变换（veil s=0, t=Ŵ），逐像素改动受染色量约束。

默认路径不触发：仅 Thresholds.experimental_ocr_line_seed=True 运行。
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from .contract import MaskLayer, MaskSource, Thresholds

METHOD = 'ocr_seed_stain'

# 日期/时间样式（与探针口径一致；不用任何固定水印文案）
_DATE_RE = re.compile(r'(19|20)\d{2}\s*[-/年.]\s*\d{1,2}\s*[-/月.]\s*\d{1,2}')
_TIME_RE = re.compile(r'\d{1,2}\s*[:：]\s*\d{2}')


def is_timestamp_like(text: str) -> bool:
    t = re.sub(r'\s+', ' ', str(text))
    return bool(_DATE_RE.search(t) or _TIME_RE.search(t))


# 种子（OCR 行）筛选：横平、置信度足够、成一行字；
# 字号/相对大小**不作**条件（实测：document_030 的时间戳字高 28px 大于
# 正文模态 23px）。真正的把关在物理证据门（染色带+深色墨迹拒绝），
# 种子只负责把候选行提出来。
_SEED_MIN_CONF = 0.6
_SEED_MAX_TILT = 12.0       # 度
_SEED_MIN_LEN_RATIO = 2.5 # 合并后行宽 ≥ 2.5×字高（一行字，不是印章/碎片）
_SEED_MERGE_GAP = 3.0      # 同行碎片合并：横向间隙 ≤ 3×字高
_PAD_RATIO = 1.5            # 种子框外扩倍数（字高）

# 染色证据带（相对纸面 P 的偏离，级）；上限为动态的 0.35P——
# 实测时间戳偏离跨度大（document_030 达 38 级），深色真实墨迹由
# 专门的墨迹门（框内 gray<0.5P 占比）拒绝，不由本带承担
_STAIN_DEV_LO = 0.9
_STAIN_DEV_HI_FRAC = 0.50
_STAIN_NEUTRAL_MAX = 60.0   # 通道间离散度上限（时间戳为浅彩色：实测离散 12–38 级）
_STAIN_BRIGHT_MIN = 0.45    # 染色笔画不得深于 0.45P（document_029 核心偏离达 114 级）
_PAPER_LOCAL_STD_MAX = 6.0  # 框域干净纸离散度（排除纹理底）
_DARK_INK_MAX_FRAC = 0.04   # 框内深色墨迹占比上限（真实内容保护）
_COVERAGE_MIN = 0.12        # 实测笔画占框面积比例下限
_COVERAGE_MAX = 0.85        # 上限：整框均匀染色不属本通道
_MIN_STROKE_PX = 150
_MIN_COMPONENT_PX = 6
_W_NOMINAL = 75.0           # 水印颜色名义值（W∈[25,120] 时残差 ≤1.2 级）
_ALPHA_CLIP = (0.004, 0.60)
_FOOTPRINT_DILATE = 3


def _geometry(box: np.ndarray) -> tuple[float, float, float]:
    from .ocr import _geometry as g
    return g(np.asarray(box, np.float32))


def _seed_lines(lines) -> list[tuple[np.ndarray, str, float]]:
    """提出候选行：横平、置信度足够；同行碎片先合并。

    字号大小不是条件（时间戳可能比正文还大，见 document_030 实测）。
    安全性由物理证据门保证：框内深色墨迹（真实内容）与染色带覆盖率
    不达标一律拒绝——正文行必然含深色墨迹而被拒。
    """
    cands = []
    for box, text, conf in lines or []:
        if not is_timestamp_like(text):
            continue  # 浅灰真实文字与浅灰水印字色调不可分，只认时间戳类
        try:
            box = np.asarray(box, np.float32)
            h, length, tilt = _geometry(box)
        except (TypeError, ValueError):
            continue
        if conf >= _SEED_MIN_CONF and tilt <= _SEED_MAX_TILT:
            cands.append({'box': box, 'text': str(text), 'conf': float(conf),
                          'h': h, 'len': length,
                          'yc': float(box[:, 1].mean()),
                          'x0': float(box[:, 0].min()), 'x1': float(box[:, 0].max())})
    # 同行碎片合并（如 charts_015 的 "2026-18 0" + "1:44"）
    merged = True
    while merged:
        merged = False
        cands.sort(key=lambda c: (c['yc'], c['x0']))
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                ya0, ya1 = a['box'][:, 1].min(), a['box'][:, 1].max()
                yb0, yb1 = b['box'][:, 1].min(), b['box'][:, 1].max()
                v_overlap = min(ya1, yb1) - max(ya0, yb0)
                if v_overlap < 0.5 * min(a['h'], b['h']):
                    continue
                gap = max(a['x0'], b['x0']) - min(a['x1'], b['x1'])
                if gap <= _SEED_MERGE_GAP * min(a['h'], b['h']):
                    xa0 = min(a['x0'], b['x0']); xa1 = max(a['x1'], b['x1'])
                    ya = np.vstack([a['box'], b['box']])
                    union = ya[(ya[:, 0] >= xa0 - 1) & (ya[:, 0] <= xa1 + 1)]
                    ys_min, ys_max = float(ya[:, 1].min()), float(ya[:, 1].max())
                    box = np.float32([[xa0, ys_min], [xa1, ys_min],
                                      [xa1, ys_max], [xa0, ys_max]])
                    h, length, tilt = _geometry(box)
                    cands[i] = {'box': box,
                                'text': a['text'] + ' ' + b['text'],
                                'conf': min(a['conf'], b['conf']),
                                'h': h, 'len': length,
                                'yc': (ys_min + ys_max) / 2,
                                'x0': xa0, 'x1': xa1}
                    cands.pop(j)
                    merged = True
                    break
            if merged:
                break
    return [(c['box'], c['text'], c['conf']) for c in cands
            if c['len'] >= _SEED_MIN_LEN_RATIO * c['h']]


def explain(img: np.ndarray, lines, thr: Thresholds = Thresholds()):
    """带拒绝原因的诊断入口（验收/排障用）；estimate 只返回结果。"""
    reasons: list[str] = []
    h_img, w_img = img.shape[:2]
    fimg = img.astype(np.float32)
    gray = fimg.mean(2)

    from .masks import deblock, paper_background_fraction, paper_level
    img2 = deblock(img)
    paper_map = paper_level(img2)
    P = float(np.median(paper_map))
    if paper_background_fraction(img2, paper_map) < 0.65:
        return None, ['纸面占比 <65%（非文档底，不属本通道）']
    seeds = _seed_lines(lines)
    if not seeds:
        return None, ['无合格种子行（小字/横平/成行/置信度门槛）']

    paperish = gray > 0.93 * P
    dev = P - gray
    neutral = (fimg.max(2) - fimg.min(2)) <= _STAIN_NEUTRAL_MAX
    # paperish 不作为染色斑像素的条件：染色本身就让像素偏离纸面
    # （38 级染色的笔画 tone≈217，不可能同时"像纸"）。页面级纸面门槛
    # （≥65%）与框域局部纸离散度门已在更外层把关底质。
    stain_band = ((dev >= _STAIN_DEV_LO) & (dev <= _STAIN_DEV_HI_FRAC * P)
                  & neutral & (gray >= _STAIN_BRIGHT_MIN * P))

    alpha = np.zeros((h_img, w_img), np.float32)
    selection = np.zeros((h_img, w_img), np.uint8)
    footprint = np.zeros((h_img, w_img), bool)
    used = 0
    for box, _text, _conf in seeds:
        xs, ys = box[:, 0], box[:, 1]
        bh = float(np.linalg.norm(box[1] - box[0]))
        bv = float(np.linalg.norm(box[2] - box[1]))
        gh = min(bh, bv)
        pad = max(8.0, _PAD_RATIO * gh)
        x0, x1 = int(max(0, xs.min() - pad)), int(min(w_img, xs.max() + pad + 1))
        y0, y1 = int(max(0, ys.min() - pad)), int(min(h_img, ys.max() + pad + 1))
        if x1 - x0 < 8 or y1 - y0 < 8:
            reasons.append('种子框过小')
            continue
        box_w = float(xs.max() - xs.min()) + 1
        box_h = float(ys.max() - ys.min()) + 1
        seed_box_area = max(box_w * box_h, 1.0)

        local_paper = paperish[y0:y1, x0:x1]
        lp = local_paper.sum()
        if lp < 0.5 * (x1 - x0) * (y1 - y0):
            reasons.append('框域纸面参照不足')
            continue
        if float(gray[y0:y1, x0:x1][local_paper].std()) > _PAPER_LOCAL_STD_MAX:
            reasons.append('框域干净纸离散度过大（纹理底）')
            continue
        sx0, sx1 = int(max(0, xs.min())), int(min(w_img, xs.max() + 1))
        sy0, sy1 = int(max(0, ys.min())), int(min(h_img, ys.max() + 1))
        dark_ink = (gray[sy0:sy1, sx0:sx1] < 0.5 * P).sum()
        if dark_ink > _DARK_INK_MAX_FRAC * (sx1 - sx0) * (sy1 - sy0):
            reasons.append('框内含深色墨迹（真实内容，拒绝）')
            continue

        stain = stain_band[y0:y1, x0:x1].copy()
        # 只保留与种子框带（外扩一字高）相交的连通笔画，防止邻域正文碎影混入
        band = np.zeros_like(stain)
        bx0, bx1 = int(max(0, xs.min() - gh - x0)), int(min(w_img, xs.max() + gh + 1 - x0))
        by0, by1 = int(max(0, ys.min() - gh - y0)), int(min(h_img, ys.max() + gh + 1 - y0))
        band[max(0, by0):by1, max(0, bx0):bx1] = True
        n, labels, stats, _ = cv2.connectedComponentsWithStats(stain.astype(np.uint8), 8)
        keep = np.zeros_like(stain)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < _MIN_COMPONENT_PX:
                continue
            comp = labels == i
            if (comp & band).any():
                keep |= comp
        stroke_px = int(keep.sum())
        cov = stroke_px / seed_box_area
        if stroke_px < _MIN_STROKE_PX or not (_COVERAGE_MIN <= cov <= _COVERAGE_MAX):
            reasons.append(f'实测笔画证据不足/越界：{stroke_px}px 覆盖率{cov:.2f}')
            continue

        raw = np.zeros((h_img, w_img), np.float32)
        raw[y0:y1, x0:x1][keep] = dev[y0:y1, x0:x1][keep]
        alpha[y0:y1, x0:x1][keep] = raw[y0:y1, x0:x1][keep]   # 暂存实测偏离
        selection[y0:y1, x0:x1][keep] = 255
        fp = cv2.dilate(keep.astype(np.uint8),
                        np.ones((_FOOTPRINT_DILATE * 2 + 1,) * 2, np.uint8)) > 0
        footprint[y0:y1, x0:x1] |= fp
        used += 1

    if used == 0:
        return None, reasons or ['所有种子被拒绝']
    # 全局一次标定（同一页多个种子共享一次测量）：
    #   δ = 笔画偏离中位；α̂ = δ/(P−W名义)；t̂ 逐通道 = (C_core−(1−α̂)P)/α̂
    # 逐像素 α(x)=α̂·dev(x)/δ 使每个染色像素（含抗锯齿边缘）精确回归纸面；
    # 不再用名义灰色——彩色水印减灰会留色相残差（用户 17:0x 截图发现的缺陷）。
    stain_px = alpha > 0
    devs = dev[stain_px]
    delta = float(np.median(devs))
    a_hat = float(np.clip(delta / max(P - _W_NOMINAL, 1.0), *_ALPHA_CLIP))
    # Ŵ 取深核（偏离 ≥70 分位）的均值颜色：中位数会被大面积抗锯齿边缘
    # （偏离远小于核心）污染而偏亮，导致欠减（029 实测教训）。
    core_mask = stain_px.copy()
    core_mask[stain_px] = devs >= np.percentile(devs, 70)
    core = fimg[core_mask]
    t_hat = (core.mean(0) - (1.0 - a_hat) * P) / max(a_hat, 1e-6)
    alpha[stain_px] = np.clip(a_hat * dev[stain_px] / max(delta, 1e-6), 0.0, 0.95)
    w_color = np.clip(t_hat, 0, 255).astype(np.float32)
    confidence = float(np.clip((alpha > 0).sum() / max(used, 1) / 800.0, 0.3, 1.0))
    layer = MaskLayer(
        alpha=alpha,
        source=MaskSource.MODEL,
        confidence=confidence,
        veil_strength=np.zeros(3, np.float32),
        veil_offset=w_color,
        paper=paper_map,
        footprint=footprint,
        overlay_opacity=alpha.copy(),
        selection=selection,
        method=METHOD,
        repair_mode=METHOD,
    )
    return layer, [f'触发（{used} 个种子）']


def estimate(img: np.ndarray, lines, thr: Thresholds = Thresholds()) -> MaskLayer | None:
    layer, _reasons = explain(img, lines, thr)
    return layer
