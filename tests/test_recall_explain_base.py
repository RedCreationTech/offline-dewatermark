import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
from unittest.mock import patch

import numpy as np

from wm import recall
from wm.contract import MaskLayer, Thresholds


def _block(shape, x0, x1, y0=10, y1=110):
    a = np.zeros(shape, np.float32)
    a[y0:y1, x0:x1] = 1.
    return a


class ExplainBaseRatioTests(unittest.TestCase):
    """覆盖率判别: 补全结果必须解释基础选区的大部分, 否则判失效。"""

    def test_full_cover_passes(self):
        base = _block((200, 300), 10, 110)
        located = MaskLayer(base.copy(), footprint=base, method='recall_repeated_strokes')
        self.assertTrue(recall._explains_base(base > 0, located, Thresholds()))

    def test_partial_cover_below_floor_fails(self):
        base = _block((200, 300), 10, 110)          # 100px 宽
        loc = _block((200, 300), 10, 69)            # 盖 59% +4px 容差 < 75%
        located = MaskLayer(loc, footprint=loc, method='recall_repeated_strokes')
        self.assertFalse(recall._explains_base(base > 0, located, Thresholds()))

    def test_empty_located_fails(self):
        base = _block((200, 300), 10, 110)
        located = MaskLayer(np.zeros((200, 300), np.float32),
                            footprint=np.zeros((200, 300), bool),
                            method='recall_repeated_strokes')
        self.assertFalse(recall._explains_base(base > 0, located, Thresholds()))


class ExpandFallbackTests(unittest.TestCase):
    """三种重复文字方法: 补全验证不过 → 保留 base 并标记, 不替换。"""

    def _mask(self, alpha):
        return MaskLayer(alpha.copy(), footprint=alpha > 0, confidence=.8,
                         method='photo_observed_repeats')

    def _expand(self, base_alpha, align_result, invert_result):
        image = np.zeros((200, 300, 3), np.uint8)
        base = self._mask(base_alpha)
        with patch('wm.ocr.read_lines', return_value=[]), \
             patch('wm.repeat_alignment.align_with_peaks', return_value=align_result), \
             patch('wm.repeat_alignment.complete_and_invert', return_value=invert_result):
            return recall.expand(image, base, Thresholds(watermark_recall_priority=True))

    def test_validated_completion_replaces(self):
        base_alpha = _block((200, 300), 10, 110)
        region = base_alpha > 0
        peaks = [(0, 0, 1.0), (120, 0, 1.0), (0, 120, 1.0)]
        combined = base_alpha.copy()
        inverse = base_alpha > 0
        alpha_map = np.full((200, 300), .3, np.float32)
        colour_map = np.zeros((200, 300, 3), np.float32)
        result = self._expand(base_alpha, (region, peaks), (combined, inverse, alpha_map, colour_map))
        self.assertEqual(result.method, 'recall_repeated_strokes')
        self.assertIsNone(result.skip_reason)
        np.testing.assert_array_equal(result.alpha, combined)

    def test_failed_validation_falls_back_and_marks(self):
        base_alpha = _block((200, 300), 10, 110)
        region = base_alpha > 0
        peaks = [(0, 0, 1.0), (120, 0, 1.0), (0, 120, 1.0)]
        result = self._expand(base_alpha, (region, peaks), (None, None, None, None))
        self.assertEqual(result.skip_reason, 'cannot_explain_base')
        self.assertEqual(result.method, 'photo_observed_repeats')   # base 保留
        np.testing.assert_array_equal(result.alpha, base_alpha)     # 未被替换

    def test_no_alignment_falls_back_and_marks(self):
        base_alpha = _block((200, 300), 10, 110)
        result = self._expand(base_alpha, (None, []), (None, None, None, None))
        self.assertEqual(result.skip_reason, 'cannot_explain_base')
        self.assertEqual(result.method, 'photo_observed_repeats')
        np.testing.assert_array_equal(result.alpha, base_alpha)

    def test_other_methods_untouched(self):
        image = np.zeros((200, 300, 3), np.uint8)
        alpha = _block((200, 300), 10, 110)
        mask = MaskLayer(alpha, footprint=alpha > 0, method='tonal')
        with patch('wm.ocr.read_lines', return_value=[]):
            result = recall.expand(image, mask, Thresholds(watermark_recall_priority=True))
        self.assertIsNone(result.skip_reason)
        np.testing.assert_array_equal(result.alpha, alpha)


if __name__ == '__main__':
    unittest.main()
