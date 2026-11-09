"""极淡区域一致染色反变换（faint_region_uniform_tint）隔离通道测试。

覆盖：合成页检出与拟合精度、干净页拒绝、整页染色拒绝、
默认路径隔离（样式锁定不触发）、修复逆变换精确性、笔画补全保护。
"""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from wm import faint_region, lama, recall
from wm.contract import Thresholds
from wm.masks import estimate_single_page


def _faint_page(alpha: float = 0.02, w_color: int = 75, region_frac: float = 0.10):
    """白纸 + 正文行 + 一块平坦极淡染色（合成即叠印后的观测）。"""
    img = np.full((600, 800, 3), 255, np.uint8)
    for y in (60, 120, 540):
        cv2.putText(img, "RECORD 086 CONFIRMED", (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
    h, w = img.shape[:2]
    rw = min(int(w * region_frac ** 0.5 * 1.6), w)
    rh = min(int(h * region_frac ** 0.5), h)
    x0, y0 = max(0, (w - rw) // 2), max(0, (h - rh) // 2)
    tint = (1 - alpha) * 255 + alpha * w_color
    img[y0:y0 + rh, x0:x0 + rw] = np.round(tint).astype(np.uint8)
    gt = np.zeros((h, w), bool)
    gt[y0:y0 + rh, x0:x0 + rw] = True
    return img, gt, alpha, w_color


class FaintRegionTests(unittest.TestCase):
    def test_detects_and_fits_uniform_tint(self):
        img, gt, alpha, _ = _faint_page()
        mask = faint_region.estimate(img)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.method, 'faint_region_uniform_tint')
        self.assertEqual(mask.repair_mode, 'faint_region_uniform_tint')
        # 逐像素 α：实心染色处接近饱和上限，干净纸接近 0
        self.assertGreater(float(np.median(mask.overlay_opacity[gt])), 0.8)
        # 选区覆盖染色块并允许块级放大，但不得蔓延全页
        self.assertLess(mask.footprint.mean(), 0.6)
        self.assertGreater(float((mask.footprint & gt).sum()) / gt.sum(), 0.8)

    def test_clean_page_is_declined(self):
        img, _, _, _ = _faint_page(alpha=0.0)
        img[:] = 255
        for y in (60, 120, 540):
            cv2.putText(img, "RECORD 086 CONFIRMED", (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
        self.assertIsNone(faint_region.estimate(img))

    def test_whole_page_tint_is_declined(self):
        img, _, _, _ = _faint_page(region_frac=0.85)
        self.assertIsNone(faint_region.estimate(img))

    def test_default_cascade_does_not_trigger(self):
        img, _, _, _ = _faint_page()
        mask = estimate_single_page(img, style='corner_id')  # 锁定其他样式
        self.assertNotEqual(mask.method, 'faint_region_uniform_tint')
        mask2 = estimate_single_page(img, style='faint')     # 显式样式提示触发
        self.assertEqual(mask2.method, 'faint_region_uniform_tint')

    def test_repair_inverse_is_exact_outside(self):
        img, gt, alpha, w_color = _faint_page()
        mask = faint_region.estimate(img)
        out, used = lama.repair_with_lama(img, mask, Thresholds())
        self.assertIn('alpha_unmix', used)
        # 区域外必须与输入逐像素一致；真值=未染色的原页（色块处回归纸面）
        expected = img.copy()
        expected[gt] = 255
        outside = np.abs(out.astype(int) - expected.astype(int))[~mask.footprint].max()
        self.assertLessEqual(outside, 1)  # 区域外严格不动（含区域环带外的正文）
        inside = np.abs(out.astype(int) - 255)[gt].mean()
        self.assertLessEqual(inside, 1.5)
        # 区域内未染色内容存在已登记的连带（≈痕量亮度回调），需有界且不可见；
        # 若区域恰好只含染色像素（理想情形），连带为空即通过。
        collateral = np.abs(out.astype(int) - expected.astype(int))[mask.footprint & ~gt]
        if collateral.size:
            self.assertLessEqual(collateral.max(), 5)
            self.assertLessEqual(collateral.mean(), 1.0)

    def test_recall_expand_leaves_fitted_model_alone(self):
        img, _, _, _ = _faint_page()
        mask = faint_region.estimate(img)
        expanded = recall.expand(img, mask, Thresholds(watermark_recall_priority=True))
        self.assertIs(expanded, mask)


if __name__ == '__main__':
    unittest.main()
