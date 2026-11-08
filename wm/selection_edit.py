"""Explicit brush decisions, kept separate from measured watermark opacity.

Pending strokes live only in memory. Processing uses the source image and the
whole edited selection; it never detects again or feeds a previous repair back
into the model. Automatic repair methods retain their measured observations.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from hashlib import sha256
from time import perf_counter

import numpy as np

try:
    from zoneinfo import ZoneInfo
    _SHANGHAI = ZoneInfo('Asia/Shanghai')
except Exception:  # Windows 无系统时区库且未装 tzdata 时，用固定 UTC+8（上海自 1991 年起无夏令时）
    from datetime import timedelta, timezone
    _SHANGHAI = timezone(timedelta(hours=8), 'Asia/Shanghai')

from . import repair, verdict
from .assisted import image_hash
from .contract import EngineKind, MaskLayer, MaskSource, PageResult, RepairName, Thresholds


def automatic_selection(result):
    automatic = result.automatic_result or result
    threshold = automatic.processing.get('thresholds', {}).get('mask_alpha_min', .08)
    return automatic.mask.binary(threshold) > 0


class SelectionEdit:
    """Last stroke wins, with per-stroke patch history rather than page copies."""

    def __init__(self, result):
        self.result = result
        self.automatic = automatic_selection(result)
        threshold = result.processing.get('thresholds', {}).get('mask_alpha_min', .08)
        current = result.mask.binary(threshold) > 0
        self.overrides = (result.selection_overrides.copy() if result.selection_overrides is not None
                          else np.full(current.shape, -1, np.int8))
        if result.selection_overrides is None:
            self.overrides[current & ~self.automatic] = 1
            self.overrides[~current & self.automatic] = 0
        self.selected = current.copy()
        self.last_change = (True, 0)  # 最近一笔的 (是否添加, 像素数)，供界面反馈
        self.undo_stack = []
        self.redo_stack = []
        self._stroke = None

    @property
    def dirty(self):
        return bool(self.undo_stack or self._stroke)

    def begin_stroke(self):
        self._stroke = []

    def paint(self, start, end, diameter, include):
        if self._stroke is None:
            self.begin_stroke()
        values = np.asarray([*start, *end, diameter], float)
        if not np.isfinite(values).all() or diameter < 1:
            raise ValueError('画笔坐标或直径无效')
        h, w = self.selected.shape
        radius = diameter / 2
        x0 = max(0, int(np.floor(min(start[0], end[0]) - radius)))
        x1 = min(w, int(np.ceil(max(start[0], end[0]) + radius)))
        y0 = max(0, int(np.floor(min(start[1], end[1]) - radius)))
        y1 = min(h, int(np.ceil(max(start[1], end[1]) + radius)))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
        xx += .5; yy += .5
        dx, dy = end[0]-start[0], end[1]-start[1]
        length = dx*dx+dy*dy
        t = np.clip(((xx-start[0])*dx+(yy-start[1])*dy)/length, 0, 1) if length else 0
        hit = (xx-start[0]-t*dx)**2 + (yy-start[1]-t*dy)**2 <= radius*radius
        region = self.overrides[y0:y1, x0:x1]
        if not np.any(hit & (region != int(include))):
            return
        before = region.copy()
        region[hit] = int(include)
        self.selected[y0:y1, x0:x1][hit] = include
        self._stroke.append((x0, y0, x1, y1, before, region.copy()))
        self.last_change = (bool(include), int((hit & (before != int(include))).sum()))
        return x0, y0, x1, y1

    def end_stroke(self):
        if self._stroke:
            self.undo_stack.append(self._stroke)
            self.redo_stack.clear()
        self._stroke = None

    def _restore(self, stroke, redo):
        for x0, y0, x1, y1, before, after in (stroke if redo else reversed(stroke)):
            patch = after if redo else before
            self.overrides[y0:y1, x0:x1] = patch
            self.selected[y0:y1, x0:x1] = np.where(
                patch == -1, self.automatic[y0:y1, x0:x1], patch == 1)

    def undo(self):
        self.end_stroke()
        if self.undo_stack:
            stroke = self.undo_stack.pop()
            self._restore(stroke, False)
            self.redo_stack.append(stroke)

    def redo(self):
        if self.redo_stack:
            stroke = self.redo_stack.pop()
            self._restore(stroke, True)
            self.undo_stack.append(stroke)

    def request(self, thresholds):
        self.end_stroke()
        return SelectionRequest(image_hash(self.result.source.image),
                                image_hash(self.result.output.image), self.selected.copy(),
                                self.overrides.copy(), replace(thresholds, lama_mask_dilate=0),
                                sha256(self.selected.tobytes()+self.overrides.tobytes()).hexdigest())


@dataclass
class SelectionRequest:
    source_hash: str
    base_output_hash: str
    selected: np.ndarray
    overrides: np.ndarray
    thresholds: Thresholds
    selection_hash: str


def _checked_repair(source, layer, thresholds, allowed):
    output, methods = repair.repair(source.copy(), deepcopy(layer), thresholds)
    if output.shape != source.shape or output.dtype != source.dtype:
        raise ValueError('修复结果尺寸或格式异常，原结果已保留。')
    if np.any(np.any(output != source, axis=2) & ~allowed):
        raise ValueError('修复改动超出确认选区，已拒绝此次结果。')
    return output, methods


def _flat_paper_fill(source: np.ndarray, mask: np.ndarray):
    """人工笔画若位于均匀纸面上：用周围环带的中位色均匀填补，返回颜色数组或 None。

    LaMa 的统计填充在纯白纸面会留下灰调噪点（尤其周围有黑字时）；周围环带
    方差足够小说明此处就是平坦纸面，直接填中位色更干净、且绝不引入纹理。
    周围复杂（压在文字/图案上）时返回 None，交由 LaMa 处理。
    """
    import cv2
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & ~mask
    if int(ring.sum()) < max(200, int(mask.sum())):
        return None
    pixels = source[..., :3][ring].astype(np.float32)
    if float(pixels.std(axis=0).max()) > 6.0:
        return None
    return np.median(pixels, axis=0).astype(np.uint8)


def apply_selection(result: PageResult, request: SelectionRequest) -> PageResult:
    source = result.source.image
    selected, overrides, settings = request.selected, request.overrides, request.thresholds
    if image_hash(source) != request.source_hash or image_hash(result.output.image) != request.base_output_hash:
        raise ValueError('原图或处理结果已变化，请重新编辑选区。')
    if (selected.shape != source.shape[:2] or selected.dtype != bool or
            overrides.shape != selected.shape or overrides.dtype != np.int8 or
            not np.isin(overrides, [-1, 0, 1]).all() or settings.lama_mask_dilate != 0):
        raise ValueError('选区数据无效，请重新编辑。')
    if sha256(selected.tobytes()+overrides.tobytes()).hexdigest() != request.selection_hash:
        raise ValueError('选区已变化，请重新确认。')
    automatic = result.automatic_result or result
    auto = automatic_selection(result)
    if not np.array_equal(selected, np.where(overrides == -1, auto, overrides == 1)):
        raise ValueError('完整选区与画笔操作不一致。')
    start = perf_counter()
    manual = selected & (overrides == 1)
    retained = selected & ~manual
    output = source.copy()
    methods = []
    route = automatic.mask.repair_mode
    # Structural rendering is not an alpha model. Editing a PDF selection is an
    # explicit raster operation; the automatic structural baseline stays intact.
    if route == 'structure':
        manual = selected.copy()
        retained[:] = False
    if retained.any():
        alpha = np.where(retained, automatic.mask.alpha, 0).astype(np.float32)
        restricted = replace(automatic.mask, alpha=alpha, footprint=retained.copy(),
                             selection=retained.copy(), instances=[])
        restored, used = _checked_repair(source, restricted, settings, retained)
        output[retained] = restored[retained]
        methods.extend(used)
    if manual.any():
        # 平坦纸面上的人工笔画直接用周围中位色均匀填补（LaMa 会留灰调噪点）；
        # 周围复杂时回退 LaMa。改动仍被限制在笔画选区内。
        paper = _flat_paper_fill(source, manual)
        if paper is not None:
            output[manual] = paper
            methods.append(RepairName.PAPER_CLEANUP)
        else:
            # This binary permission field is never sent to alpha inversion.
            layer = MaskLayer(np.zeros(selected.shape, np.float32), source=MaskSource.HUMAN,
                              confidence=1, selection=manual, footprint=manual,
                              method='explicit_brush', repair_mode='inpaint')
            restored, used = _checked_repair(source, layer, settings, manual)
            output[manual] = restored[manual]
            methods.extend(used)
    elapsed = round((perf_counter()-start)*1000)
    changed = np.any(output != source, axis=2)
    layer = replace(automatic.mask, alpha=np.where(selected, automatic.mask.alpha, 0),
                    source=MaskSource.HUMAN, confidence=1, selection=selected.copy(),
                    footprint=selected.copy(), instances=[], method='human_edited_selection',
                    repair_mode='manual_selection')
    unmeasured = replace(layer, paper=None, veil_strength=None, veil_offset=None, overlay_opacity=None)
    report = verdict.build_verdict(source, output, unmeasured, EngineKind.RASTER,
                                   list(dict.fromkeys(methods)), elapsed,
                                   {'manual_repair_ms': elapsed}, settings)
    report.page_key = result.source.key
    # Permission/confirmation cannot be scored as measured watermark opacity.
    report.conditions.pop('residual', None)
    report.metrics.pop('residual_basis', None)
    report.metrics['residual_coverage'] = float('nan')
    processing = deepcopy(result.processing)
    history = processing.setdefault('human_assistance', [])
    history.append(dict(operation='edit_complete_selection',
                        confirmed_at=datetime.now(_SHANGHAI).isoformat(timespec='seconds'),
                        source_sha256=request.source_hash, base_output_sha256=request.base_output_hash,
                        candidate_sha256=request.selection_hash, selected_pixels=int(selected.sum()),
                        added_pixels=int((selected & ~auto).sum()), removed_pixels=int((auto & ~selected).sum()),
                        manually_included_pixels=int((overrides == 1).sum()),
                        manually_excluded_pixels=int((overrides == 0).sum()),
                        changed_pixels=int(changed.sum()), outside_selected_changed_pixels=0,
                        detector_method='not_run', repair_mode='manual_selection',
                        retained_automatic_route=route, thresholds=asdict(settings),
                        evidence='explicit_human_pixel_selection', quality_verified=False))
    processing.update(mode='human_assisted', thresholds=asdict(settings),
                      selection_semantics='complete_binary_permission_not_physical_alpha')
    return PageResult(result.source, replace(result.output, image=output), layer, report,
                      processing=processing, automatic_result=automatic,
                      assistance_masks=[*result.assistance_masks, selected.copy()],
                      selection_overrides=overrides.copy())
