"""OCR 单行小字种子通道（ocr_seed_stain）隔离候选测试。

覆盖：合成页检出与修复精度、干净页/深色小字拒绝、框只作搜索范围
（字洞保持、区域外不动）、默认路径隔离、笔画补全保护。
"""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from wm import lama, ocr_seed_stain, recall
from wm.contract import Thresholds
from wm.masks import estimate_single_page

ALPHA = 0.02
W_GRAY = 75


def _page(stamp_alpha: float = ALPHA, dark_caption: bool = False):
    """白纸 + 大正文行 + 一行小字（极淡时间戳或深色小字）。

    返回 (img, gt_stroke, seed_box)。seed_box 模拟 OCR 读出的框。
    """
    img = np.full((700, 1000, 3), 255, np.uint8)
    for y in (80, 150, 620):
        cv2.putText(img, 'PURCHASE AGREEMENT SECTION FOUR', (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 40, 40), 3, cv2.LINE_AA)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    text = '2026-10-27 16:05'
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x0, y0 = 620, 660
    gt = np.zeros(img.shape[:2], bool)
    if stamp_alpha > 0:
        tone = int(np.round((1 - stamp_alpha) * 255 + stamp_alpha * W_GRAY))
        cv2.putText(img, text, (x0, y0), font, scale, (tone,) * 3, thick, cv2.LINE_AA)
        canvas = np.zeros(img.shape[:2], np.uint8)
        cv2.putText(canvas, text, (x0, y0), font, scale, 1, thick, cv2.LINE_AA)
        gt = canvas.astype(bool)
    else:
        cv2.putText(img, text, (x0, y0), font, scale, (60, 60, 60) if dark_caption else (255,) * 3,
                    thick, cv2.LINE_AA)
    pad = 4
    box = np.float32([[x0 - pad, y0 - th - base - pad], [x0 + tw + pad, y0 - th - base - pad],
                      [x0 + tw + pad, y0 + base + pad], [x0 - pad, y0 + base + pad]])
    return img, gt, box, text


def _with_body(box, text):
    """模拟真实 OCR 输出：正文大行 + 小字行（种子是相对字号判断的）。"""
    body = np.float32([[30, 40], [700, 40], [700, 90], [30, 90]])
    body2 = np.float32([[30, 110], [700, 110], [700, 160], [30, 160]])
    return [(body, 'PURCHASE AGREEMENT SECTION FOUR', 0.95),
            (body2, 'PURCHASE AGREEMENT SECTION FIVE', 0.95),
            (box, text, 0.9)]


class OcrSeedStainTests(unittest.TestCase):
    def test_detects_and_repairs_measured_stain(self):
        img, gt, box, text = _page()
        mask, reasons = ocr_seed_stain.explain(img, _with_body(box, text))
        self.assertIsNotNone(mask, reasons)
        self.assertEqual(mask.method, 'ocr_seed_stain')
        self.assertEqual(mask.repair_mode, 'ocr_seed_stain')
        sel = mask.binary(0.08) > 0
        self.assertGreater(float(sel[gt].mean()), 0.5)          # 实测笔画召回
        # 多选容差按抗锯齿边缘计（2px 内不算背景误选）
        near = cv2.dilate(gt.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.assertLess(float((sel & ~near).sum()), 0.1 * sel.sum())
        out, used = lama.repair_with_lama(img, mask, Thresholds())
        self.assertIn('alpha_unmix', used)
        before = float(np.abs(img.mean(2) - 255)[gt].mean())
        after = float(np.abs(out.mean(2) - 255)[gt].mean())
        self.assertLess(after, before * 0.4)
        outside = np.abs(out.astype(int) - img.astype(int))[~mask.footprint].max()
        self.assertEqual(int(outside), 0)                       # 区域外逐像素不动

    def test_clean_small_line_is_declined(self):
        img, _gt, box, text = _page(stamp_alpha=0.0)
        mask, reasons = ocr_seed_stain.explain(img, _with_body(box, text))
        self.assertIsNone(mask)

    def test_dark_caption_is_declined(self):
        img, _gt, box, text = _page(stamp_alpha=0.0, dark_caption=True)
        mask, reasons = ocr_seed_stain.explain(img, _with_body(box, text))
        self.assertIsNone(mask)
        self.assertTrue(any('深色墨迹' in r for r in reasons))

    def test_holes_between_strokes_kept(self):
        """选区沿笔画分布：字形内部空洞不被填上。"""
        img, gt, box, text = _page()
        mask, _ = ocr_seed_stain.explain(img, _with_body(box, text))
        sel = mask.binary(0.08) > 0
        # 膨胀选区模拟"如果填洞"会吞掉的空洞面积必须仍然存在
        dil = cv2.dilate(sel.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.assertGreater(int((dil & gt & ~sel).sum()), 0)     # 空洞真实存在且未被选
        # 选区允许含抗锯齿边缘（比理想笔画粗），但不得整框填充（>1.8×理想笔画）
        self.assertLess(float(sel.sum()), 1.8 * float(gt.sum()))

    def test_default_cascade_triggers_by_default(self):
        # 默认路径（2026-11-05 转正后）：真实 OCR 读出时间戳行时通道自动生效。
        # 合成页用较浓的戳（α=0.18），保证 RapidOCR 真能读出该行。
        img, _gt, _box, _text = _page(stamp_alpha=0.18)
        mask = estimate_single_page(img, thr=Thresholds())
        self.assertEqual(mask.method, 'ocr_seed_stain')
        # 显式关闭开关：不触发
        mask_off = estimate_single_page(
            img, thr=Thresholds(experimental_ocr_line_seed=False))
        self.assertNotEqual(mask_off.method, 'ocr_seed_stain')
        # 锁定其他样式（corner_id）：该通道不参与，不触发
        mask2 = estimate_single_page(img, style='corner_id')
        self.assertNotEqual(mask2.method, 'ocr_seed_stain')

    def test_recall_expand_leaves_measured_model_alone(self):
        img, _gt, box, text = _page()
        mask = ocr_seed_stain.estimate(img, _with_body(box, text))
        expanded = recall.expand(img, mask, Thresholds(watermark_recall_priority=True))
        self.assertIs(expanded, mask)


if __name__ == '__main__':
    unittest.main()
