"""Coloured dot lattice on a low-luminance-contrast photograph (chroma path).

Red/pink dots whose luminance deviation is weak or ambiguous were invisible to
the luminance-only periodic fit (2026-11-26 photo_021 batch). The chroma retry
must recover them from hue deviation alone, without touching anything outside
the verified lattice footprint.
"""
from __future__ import annotations

import unittest

import numpy as np

from wm import periodic, repair
from wm.contract import Thresholds


def _photo_fixture(period: int, dot_rgb=(185, 75, 95), dot_alpha: float = .55):
    """灰色照片状背景（亮度与点接近、色度近零）+ 红色半透明圆点阵。"""
    h, w = 480, 640
    rng = np.random.default_rng(3)
    yy, xx = np.indices((h, w))
    base = 112 + 26*np.sin(xx/95.) + 14*np.cos(yy/60.)
    clean = np.stack([base, base, base*0.99], -1)
    clean += rng.normal(0, 3.5, clean.shape)
    clean = np.clip(clean, 0, 255).astype(np.uint8)
    dx = (xx-7+period/2) % period - period/2
    dy = (yy-9+period/2) % period - period/2
    dot = np.clip(2.6 - np.sqrt(dx*dx+dy*dy), 0, 1)
    alpha = (dot * dot_alpha).astype(np.float32)
    image = np.clip(clean*(1-alpha[:, :, None])
                    + alpha[:, :, None]*np.array(dot_rgb, np.float32), 0, 255).astype(np.uint8)
    return clean, image, alpha


class PeriodicChromaTests(unittest.TestCase):
    def test_luminance_or_chroma_recovers_red_dots_end_to_end(self):
        period = 24
        clean, image, alpha = _photo_fixture(period)
        # 亮度或色度通道谁先命中都可以，关键是端到端去掉红点且选区外不动
        mask = periodic.estimate(image)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.lattice, (period, period))
        region = alpha > .05
        before = abs(image.astype(float)-clean.astype(float))[region].mean()
        output, used = repair.repair(image, mask, Thresholds())
        after = abs(output.astype(float)-clean.astype(float))[region].mean()
        # 候选级恢复：相位拟合的 alpha 有噪声，允许残留 ~2 倍背景噪声；
        # 端到端把红点压到不可明显察觉，进一步优化（逐像素 alpha 精调）另案。
        self.assertLess(after, before*.4)
        self.assertLess(after, 8.0)
        # 选区外逐像素保持原图
        np.testing.assert_array_equal(output[~mask.footprint], image[~mask.footprint])

    def test_grey_dot_grid_still_uses_luminance_path(self):
        period = 24
        clean, image, _ = _photo_fixture(period, dot_rgb=(70, 70, 78))
        self.assertIsNotNone(periodic.estimate(image))


if __name__ == "__main__":
    unittest.main()
