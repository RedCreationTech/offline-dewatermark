# -*- coding: utf-8 -*-
"""效果好基线防回退对比的单元测试。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import build_goodset_baseline_report as g


class GoodsetCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.baseline = g.BASELINE
        self.saved = self.baseline.read_bytes() if self.baseline.is_file() else None
        sample = dict(group='g1', label='l', metrics=dict(
            recall=0.99, precision=0.5, strong_recall=0.99,
            missed_pixels=10, false_positive_pixels=100, fp_frac=0.001,
            watermark_mae=0.1, background_mae=0.0, elapsed_ms=1000))
        self.baseline.parent.mkdir(parents=True, exist_ok=True)
        self.baseline.write_text(json.dumps(
            dict(samples=dict(s1=sample)), ensure_ascii=False), encoding='utf-8')

    def tearDown(self):
        if self.saved is None:
            self.baseline.unlink(missing_ok=True)
        else:
            self.baseline.write_bytes(self.saved)
        self.tmp.cleanup()

    def _records(self, **kw):
        m = dict(recall=0.99, precision=0.5, missed_pixels=10,
                 false_positive_pixels=100, fp_frac=0.001,
                 watermark_mae=0.1, background_mae=0.0, elapsed_ms=1000)
        m.update(kw)
        return dict(s1=dict(metrics=m))

    def test_same_passes(self):
        self.assertEqual(g.check(self._records()), 0)

    def test_recall_drop_fails(self):
        self.assertEqual(g.check(self._records(recall=0.98)), 1)

    def test_missed_rise_fails(self):
        self.assertEqual(g.check(self._records(missed_pixels=11)), 1)

    def test_improvement_passes(self):
        self.assertEqual(g.check(self._records(recall=1.0, missed_pixels=5)), 0)

    def test_missing_sample_fails(self):
        self.assertEqual(g.check({}), 1)

    def test_elapsed_within_tolerance_passes(self):
        self.assertEqual(g.check(self._records(elapsed_ms=1050)), 0)

    def test_elapsed_beyond_tolerance_fails(self):
        self.assertEqual(g.check(self._records(elapsed_ms=1150)), 1)


if __name__ == '__main__':
    unittest.main()
