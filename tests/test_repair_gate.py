"""Pre-repair evidence gate: flagged pages must come back byte-identical.

The gate enforces, before repair, intents that previously only existed after
repair (verdict confidence < 0.12, repaint region > 25% of a page, empty or
negligible selections). A blocked page must equal its input exactly and carry
needs_manual_mask; a healthy mask must be repaired exactly as before.
"""
from __future__ import annotations

import unittest

import numpy as np

from wm.contract import MaskLayer, Thresholds
from wm.pipeline import _gate_block_reason, _process_one
from wm.contract import PageImage


def _page() -> PageImage:
    rng = np.random.default_rng(7)
    img = np.full((120, 160, 3), 245, np.uint8)
    img[20:40, 30:90] = 60  # dark body-text block
    img += rng.integers(0, 3, img.shape).astype(np.uint8)
    return PageImage(image=img, dpi=96, doc_id="t", page_index=0, source_kind="test")


def _mask(alpha, confidence, footprint=None) -> MaskLayer:
    return MaskLayer(alpha=alpha, confidence=confidence,
                     footprint=footprint if footprint is not None else alpha > 0)


class GateReasonTests(unittest.TestCase):
    def setUp(self):
        self.thr = Thresholds()

    def test_empty_selection_blocks(self):
        alpha = np.zeros((120, 160), np.float32)
        self.assertEqual(_gate_block_reason(_mask(alpha, 0.9), self.thr), "empty_selection")

    def test_tiny_selection_low_confidence_blocks(self):
        alpha = np.zeros((120, 160), np.float32)
        alpha[3, 3] = 1.0  # 1 px evidence, unconfident — the real_010 leak class
        self.assertEqual(_gate_block_reason(_mask(alpha, 0.083), self.thr), "negligible_selection")

    def test_tiny_selection_but_confident_passes_gate(self):
        alpha = np.zeros((120, 160), np.float32)
        alpha[3, 3] = 1.0
        self.assertIsNone(_gate_block_reason(_mask(alpha, 0.8), self.thr))

    def test_low_confidence_blocks(self):
        alpha = np.zeros((120, 160), np.float32)
        alpha[10:30, 10:60] = 0.5
        self.assertEqual(_gate_block_reason(_mask(alpha, 0.05), self.thr), "low_confidence")

    def test_broad_footprint_blocks(self):
        alpha = np.zeros((120, 160), np.float32)
        alpha[10:20, 10:60] = 0.5
        footprint = np.zeros((120, 160), bool)
        footprint[:40, :] = True  # 33% of the page
        self.assertEqual(_gate_block_reason(_mask(alpha, 1.0, footprint), self.thr),
                         "repaint_too_broad")

    def test_healthy_mask_passes(self):
        alpha = np.zeros((120, 160), np.float32)
        alpha[10:30, 10:60] = 0.5
        self.assertIsNone(_gate_block_reason(_mask(alpha, 0.9), self.thr))


class GateBehaviourTests(unittest.TestCase):
    """Blocked pages: output byte-identical to input + needs_manual_mask."""

    def _blocked_result(self, confidence, footprint=None):
        page = _page()
        alpha = np.zeros(page.image.shape[:2], np.float32)
        alpha[10:30, 10:60] = 0.5
        mask = MaskLayer(alpha=alpha, confidence=confidence,
                         veil_strength=np.full(3, .3, np.float32),
                         veil_offset=np.full(3, 40, np.float32),
                         paper=np.full(page.image.shape, 245, np.float32),
                         footprint=footprint if footprint is not None else alpha > 0)
        return page, mask, _process_one(page, mask, Thresholds())

    def test_blocked_page_returns_input_unchanged(self):
        page, mask, result = self._blocked_result(confidence=0.05)
        np.testing.assert_array_equal(result.output.image, page.image)
        self.assertEqual(result.verdict.repairs, [])

    def test_blocked_page_flagged_for_manual_review(self):
        _, _, result = self._blocked_result(confidence=0.05)
        from wm.contract import FailureCode
        self.assertIn(FailureCode.NEEDS_MANUAL_MASK, result.verdict.failures)
        self.assertTrue(result.verdict.conditions["gate_blocked"])

    def test_broad_footprint_page_returns_input_unchanged(self):
        page = _page()
        alpha = np.zeros(page.image.shape[:2], np.float32)
        alpha[10:20, 10:60] = 0.5
        footprint = np.zeros(alpha.shape, bool)
        footprint[:40, :] = True
        mask = MaskLayer(alpha=alpha, confidence=1.0,
                         veil_strength=np.full(3, .3, np.float32),
                         veil_offset=np.full(3, 40, np.float32),
                         paper=np.full(page.image.shape, 245, np.float32),
                         footprint=footprint)
        result = _process_one(page, mask, Thresholds())
        np.testing.assert_array_equal(result.output.image, page.image)

    def test_negligible_selection_page_returns_input_unchanged(self):
        page = _page()
        alpha = np.zeros(page.image.shape[:2], np.float32)
        alpha[3, 3] = 1.0
        mask = MaskLayer(alpha=alpha, confidence=0.083,
                         veil_strength=np.full(3, .3, np.float32),
                         veil_offset=np.full(3, 40, np.float32),
                         paper=np.full(page.image.shape, 245, np.float32))
        result = _process_one(page, mask, Thresholds())
        np.testing.assert_array_equal(result.output.image, page.image)

    def test_healthy_page_untouched_by_gate(self):
        # An invertible veil on paper must still be un-mixed (LaMa not needed).
        page = _page()
        alpha = np.zeros(page.image.shape[:2], np.float32)
        alpha[10:30, 10:60] = 0.3
        mask = MaskLayer(alpha=alpha, confidence=0.9,
                         veil_strength=np.full(3, .3, np.float32),
                         veil_offset=np.full(3, 40, np.float32),
                         paper=np.full(page.image.shape, 245, np.float32),
                         footprint=alpha > 0, repair_mode="texture_overlay")
        result = _process_one(page, mask, Thresholds())
        self.assertGreater(float(np.any(result.output.image != page.image, axis=2).mean()), 0)
        self.assertFalse(result.verdict.conditions["gate_blocked"])


if __name__ == "__main__":
    unittest.main()
