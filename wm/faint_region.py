"""极淡区域一致染色反变换（faint_region_uniform_tint）——隔离实验通道。

适用边界（docs/极淡水印方案与适用边界讨论_20261104.md 四级梯子）：
本通道只处理第 1 级「恒定/近白参照面」：页面有足够白纸，水印是跨越大片
区域的统一染色（强度 α 与颜色 W 全区域一致），且痕迹低于常规检测下限
（<0.04×纸面）。它对整片区域联合拟合同一个 (α, W)，再交给既有
alpha_unmix 做数学逆变换——不逐像素判定、不重画、不脑补内容。

进入硬门槛（任一不满足即返回 None，页面保持原样）：
- 白纸参照充分（纸面背景占比 ≥40%）；
- 低频残差能提出规整的疑似区域（大面积均匀偏离纸面，非零散噪点）；
- 双聚类拟合优度达标（干净纸还原残差 ≤1.5 级、水印纸 ≤1.2 级），
  拟合不落在搜索边界上；
- 区域面积受限（≥1,500 且 ≤60% 页面），防止把整页当水印。

默认路径不触发：仅 Thresholds.experimental_faint_region=True 或
样式提示锁定 faint 通道时运行（masks._estimate_single_page 级联首部）。
"""
from __future__ import annotations

import cv2
import numpy as np

from .contract import MaskLayer, MaskSource, Thresholds

METHOD = 'faint_region_uniform_tint'

# 拟合与提案参数（本通道自己的门槛，不动 Thresholds 的默认口径）
_PAPER_MIN_FRAC = 0.40      # 白纸参照充分性
_REGION_MIN_PX = 1500
_REGION_MAX_FRAC = 0.60
_STAINED_MIN_FRAC = 0.02    # 区域内"染过的纸"至少占 2%
_STAINED_IND_MIN = 1.2      # 单个纸像素"被染"的入门（级）
_STAINED_WIN = 25           # 大字形多数决窗口（px）
_STAINED_WIN_MIN = 0.5      # 窗口内被染纸占比过半才算染色证据
_STAINED_WIN_SMALL = 13    # 窄笔画（数字/小字）第二尺度窗口
_STAINED_WIN_SMALL_MIN = 0.45
_BLOB_MIN_PX = 150          # 多数决本身已排除稀疏结构，小面积即可成证据
_CORE_DILATE = 31           # 拟合用的字形核心区域膨胀
_BLOCK_DILATE = 301         # 块级区域膨胀（把整片字块连成一块；尺寸门兜底）
_ALPHA_SEARCH = (0.004, 0.09, 107)
_FIT_STAINED_MAX = 1.2      # 水印纸还原到纸面的残差（级）
_FIT_CLEAN_MAX = 1.2        # 干净纸与"预测连带量"的偏离（级）
_FIT_INK_MAX = 12.0         # 墨核还原残差上限（级）：±10 级墨核带宽自带 ≈5 级噪声底，
                            # 只拒荒谬拟合；α 的判别仍由评分里的墨项负责
_W_PLAUSIBLE_LO = 20.0      # 水印颜色合理区间（深于近黑、暗于纸面）
_W_DARK_MIN = 25.0          # 水印颜色至少比纸面深这么多（级）
_MIN_INK_PX = 200
_PAPER_LEVEL_MIN = 250.0    # 白纸纯度：纸面极点亮度
_PAPER_CLEAN_STD_MAX = 3.0  # 干净纸亮度离散度（级），排除灰底/杂底


def _paper_reference(img: np.ndarray, gray: np.ndarray) -> tuple[float, np.ndarray] | None:
    from .masks import deblock, paper_background_fraction, paper_level
    img2 = deblock(img)
    paper = paper_level(img2)
    if paper_background_fraction(img2, paper) < _PAPER_MIN_FRAC:
        return None
    return float(np.median(paper)), paper


def _evidence_mask(dev: np.ndarray, paperish: np.ndarray) -> np.ndarray | None:
    """多数决染色证据：窗口内"过半纸像素被统一染暗"。

    均值/平滑会被稀疏强结构（正文行、表格线：5–10% 覆盖 × 大偏离）骗过；
    而大面积均匀染色的标志是多数纸像素同量偏离——用盒滤波的占比图，
    稀疏结构自然落到占比低位，与染色区分离。
    """
    indicator = ((dev > _STAINED_IND_MIN) & paperish).astype(np.float32)
    ev_a = cv2.boxFilter(indicator, ddepth=-1,
                         ksize=(_STAINED_WIN, _STAINED_WIN), normalize=True) > _STAINED_WIN_MIN
    ev_b = cv2.boxFilter(indicator, ddepth=-1,
                         ksize=(_STAINED_WIN_SMALL, _STAINED_WIN_SMALL),
                         normalize=True) > _STAINED_WIN_SMALL_MIN
    evidence = ev_a | ev_b
    evidence = cv2.morphologyEx(evidence.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((9, 9), np.uint8)) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(evidence.astype(np.uint8), 8)
    keep = np.zeros_like(evidence)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= _BLOB_MIN_PX:
            keep[labels == i] = True
    return keep if keep.any() else None


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    k = np.ones((px, px), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def _fit_uniform_tint(img: np.ndarray, gray: np.ndarray, region: np.ndarray,
                      paperish: np.ndarray, P: float,
                      require_pure_paper: bool = True) -> tuple[float, np.ndarray, dict] | None:
    """双聚类分离干净纸/染纸，一维搜索 α。

    可识别性说明：染过的白纸在任何 (α, W= P+δ/α) 下都精确回到纸面，
    单靠白纸无法区分 α——真正的判别力来自区域内的墨色极点
    （B̂−t = (α真−α)(P−t)/(1−α)，墨越深杠杆越大）。区域无墨时 α 本质上
    不可识别，用水印颜色合理区间（W 必须在近黑与纸面之间）收敛到可行解，
    此时对白纸-only 区域的修复结果与真值几乎无关。
    """
    in_region = region & paperish
    if in_region.sum() < 400:
        return None
    dvec = (img - P)[in_region].mean(1)
    stained = in_region.copy()
    stained[in_region] = dvec < -1.0
    cleanp = in_region & ~stained
    if stained.sum() < 200 or cleanp.sum() < 200:
        return None
    if require_pure_paper:
        if P < _PAPER_LEVEL_MIN:
            return None
        if float(gray[cleanp].std()) > _PAPER_CLEAN_STD_MAX:
            return None
    if stained.mean() / max(region.mean(), 1e-9) < _STAINED_MIN_FRAC:
        return None
    delta = np.median(img[stained] - P, axis=0).astype(np.float32)
    d_gray = float(delta.mean())
    inkish = region & (gray < 0.6 * P)
    ink_core = None
    ink_pole = None
    if inkish.sum() >= _MIN_INK_PX:
        ink_pole = float(np.median(gray[inkish]))
        # 只用实心墨核：抗锯齿边缘位于两极之间，参与均值会把残差
        # 抬高约一个量级，掩盖真正的判别信号。
        ink_core = region & (np.abs(gray - ink_pole) <= 10.0)
        if ink_core.sum() < _MIN_INK_PX:
            ink_core = None
            ink_pole = None

    best = None

    def _scan(grid: np.ndarray):
        nonlocal best
        for a in grid:
            w_m = P + delta / a
            if w_m.min() < _W_PLAUSIBLE_LO or w_m.max() > P - _W_DARK_MIN:
                continue  # 颜色不合理：该 α 无可行水印颜色
            inv = (img - a * w_m) / (1.0 - a)
            g_inv = inv.mean(2)
            r_stained = float(np.abs(g_inv[stained] - P).mean())
            w_gray = float(w_m.mean())
            coll = (P - a * w_gray) / (1.0 - a) - P        # 未染纸的预测连带量
            r_clean = float(np.abs(g_inv[cleanp] - (P + coll)).mean())
            score = r_clean + 2.0 * r_stained
            ink_dev = 0.0
            if ink_core is not None:
                ink_target = (ink_pole - a * w_gray) / (1.0 - a)
                ink_dev = float(np.abs(g_inv[ink_core] - ink_target).mean())
                score += 3.0 * ink_dev                     # 主判别项
            if best is None or score < best[0]:
                best = (score, float(a), w_m, r_stained, r_clean, ink_dev)

    coarse = np.linspace(*_ALPHA_SEARCH)
    _scan(coarse)
    if best is not None:
        step = float(coarse[1] - coarse[0])
        fine = np.clip(best[1] + np.linspace(-3 * step, 3 * step, 13), _ALPHA_SEARCH[0], _ALPHA_SEARCH[1])
        _scan(fine)
    if best is None:
        return None
    score, a_hat, w_hat, r_stained, r_clean, ink_dev = best
    lo, hi, _ = _ALPHA_SEARCH
    if a_hat <= lo or a_hat >= hi:
        return None
    if r_stained > _FIT_STAINED_MAX or r_clean > _FIT_CLEAN_MAX:
        return None
    if ink_core is not None and ink_dev > _FIT_INK_MAX:
        return None
    return a_hat, w_hat, {'score': score, 'r_stained': r_stained, 'r_clean': r_clean,
                          'ink_dev': ink_dev,
                          'stained_frac': float(stained.mean() / max(region.mean(), 1e-9))}


def estimate(img: np.ndarray, thr: Thresholds = Thresholds()) -> MaskLayer | None:
    img = img.astype(np.float32)  # 下游做 I-P 差分，必须摆脱 uint8 回绕
    gray = img.mean(2)
    ref = _paper_reference(img, gray)
    if ref is None:
        return None
    P, paper_map = ref
    paperish = gray > 0.93 * P
    dev = P - gray                                    # 正值=比纸面暗
    evidence = _evidence_mask(dev, paperish)
    if evidence is None:
        return None

    # 第一遍：字形核心上拟合全局 (α, W)，所有硬门槛在此把关
    core = _dilate(evidence, _CORE_DILATE)
    if not (_REGION_MIN_PX <= core.sum() <= _REGION_MAX_FRAC * core.size):
        return None
    fit = _fit_uniform_tint(img, gray, core, paperish, P)
    if fit is None:
        return None
    a_hat, w_hat, stats = fit

    # 第二遍：块级放大。拟合已验证"统一染色"假设，把区域放大到整片字块：
    # 笔画间、字形边缘的痕量全部纳入。逐像素 α 保证干净纸不被触碰，
    # 白纸上连带天然饱和不可见，区域放大的风险由白纸纯度门槛兜底。
    block = _dilate(evidence, _BLOCK_DILATE)
    if not (_REGION_MIN_PX <= block.sum() <= _REGION_MAX_FRAC * block.size):
        block = core  # 块级超界则退回核心区域（核心拟合已自检，不必拒绝）
    stain_dev = max(-a_hat * (float(w_hat.mean()) - P), 0.5)   # 总染色量（正值）
    a_pix = np.clip(dev / stain_dev, 0.0, 0.95)                # 逐像素透明度标尺
    alpha = np.zeros(gray.shape, np.float32)
    on_paper = block & paperish
    alpha[on_paper] = a_pix[on_paper]
    alpha[block & ~paperish] = a_hat                           # 块内墨迹用全局 α
    layer = MaskLayer(
        alpha=alpha,
        source=MaskSource.MODEL,
        confidence=float(np.clip(stats['stained_frac'] / 0.05, 0.0, 1.0)),
        veil_strength=np.zeros(3, np.float32),
        veil_offset=w_hat.astype(np.float32),
        paper=paper_map,
        footprint=block,
        overlay_opacity=alpha.copy(),
        method=METHOD,
        repair_mode=METHOD,
    )
    return layer
