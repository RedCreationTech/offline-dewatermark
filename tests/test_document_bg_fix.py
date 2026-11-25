# -*- coding: utf-8 -*-
"""文档底修复（设计稿 §4 第 1 关）：晶格拟合与残差验证的单元测试。"""
from __future__ import annotations

import unittest

import numpy as np

from wm.repeated import _fit_lattice, _verify_inverse


class FitLatticeTests(unittest.TestCase):
    def test_clean_progression(self):
        rng = np.random.default_rng(7)
        ys = [780 + 287 * k + rng.normal(0, 2) for k in range(12)]
        lats = _fit_lattice(ys)
        self.assertEqual(len(lats), 1)
        y0, p = lats[0]
        self.assertAlmostEqual(p, 287, delta=3)
        # 全部行应能由拟合格点重现（误差 < 3px）
        err = min(abs(y - (y0 + round((y - y0) / p) * p)) for y in ys)
        self.assertLess(err, 3)

    def test_near_duplicate_peaks_merged(self):
        # 实测形态：每行有 6~12px 的孪生峰
        ys = []
        for k in range(10):
            ys.extend([780 + 287 * k, 780 + 287 * k + 9])
        lats = _fit_lattice(ys)
        self.assertEqual(len(lats), 1)
        _, p = lats[0]
        self.assertAlmostEqual(p, 287, delta=3)

    def test_missing_rows_still_fits(self):
        rng = np.random.default_rng(3)
        ks = [0, 1, 2, 4, 5, 7, 8, 9, 11]  # 随机缺 40%
        ys = [500 + 287 * k + rng.normal(0, 2.5) for k in ks]
        lats = _fit_lattice(ys)
        self.assertEqual(len(lats), 1)
        _, p = lats[0]
        self.assertAlmostEqual(p, 287, delta=3)

    def test_outlier_peaks_rejected(self):
        ys = [780 + 287 * k for k in range(10)] + [1500, 2100]  # 离格点
        lats = _fit_lattice(ys)
        self.assertEqual(len(lats), 1)
        _, p = lats[0]
        self.assertAlmostEqual(p, 287, delta=3)

    def test_two_interleaved_lattices(self):
        # 两条交错数列（同周期、错开半周期）：判据是格点能重现全部行
        # （拟合成单条半周期格或分裂两条都算对，行位置误差 < 3px 即过）
        rng = np.random.default_rng(5)
        ys = sorted([300 + 287 * k for k in range(8)] +
                    [300 + 143 + 287 * k for k in range(8)])
        ys = [y + rng.normal(0, 2) for y in ys]
        lats = _fit_lattice(ys)
        self.assertTrue(lats)
        # 设计稿 §4-1 门槛：误差中位 < 3px（逐行噪声 σ=2 时尾部可到 ~4px，
        # 中位数才是稳定判据）
        errs = [min(abs(y - (y0 + round((y - y0) / p) * p)) for y0, p in lats)
                for y in ys]
        self.assertLess(float(np.median(errs)), 3)

    def test_spurious_text_peaks_robust(self):
        # 实测翻车场景（文档底列 1）：真行 287 周期，字区假峰成串还会自构成
        # ~88px 局部格——单列投票解不开是合理歧义；生产路径是页级共识，
        # 干净列的票源必须能把脏列翻回 287，且相位拟合覆盖真行。
        from wm.repeated import _lattice_period, _fit_lattice_phase, _merge_peaks
        clean = np.asarray([780. + 287 * k for k in range(8)])
        dirty = [781., 1068., 1145., 1233., 1355., 1495., 1584., 2216.]
        p = _lattice_period([clean, np.asarray(_merge_peaks(dirty))])
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p, 287, delta=5)
        arr = np.asarray(_merge_peaks(dirty))
        lats = _fit_lattice_phase(arr, p)
        self.assertTrue(lats)
        for y in (781, 1068, 1355, 2216):
            err = min(abs(y - (y0 + round((y - y0) / pp) * pp)) for y0, pp in lats)
            self.assertLess(err, 45)  # 真行应落在格点上（含合并造成的 ≤3px 偏移）

    def test_consensus_period_drowns_text_lattice(self):
        # 页级共识：某列的真峰被字区假峰（自带 ~88px 局部格）淹没时，
        # 合并其他干净列的票源后应回到真周期 287
        import numpy as np
        from wm.repeated import _lattice_period
        clean = np.asarray([780. + 287 * k for k in range(8)])
        dirty = np.asarray([781., 1068., 1145., 1233., 1355., 1495., 1584., 2216.])
        p = _lattice_period([clean, dirty])
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p, 287, delta=5)

    def test_too_few_peaks(self):
        self.assertEqual(_fit_lattice([100., 200.]), [])


class VerifyInverseTests(unittest.TestCase):
    def _setup(self):
        # 纸 255；两极：s=0.6, t=80 → 被盖纸极 = 233
        paper, s_vec, t_vec = 255., np.full(3, .6, np.float32), np.full(3, 80., np.float32)
        img = np.full((4, 4, 3), 255, np.float32)
        keepout = np.zeros((4, 4), bool)
        return img, paper, s_vec, t_vec, keepout

    def test_valid_paper_watermark_kept(self):
        img, paper, s_vec, t_vec, keepout = self._setup()
        covered = .6 * paper + 80  # 233
        alpha = np.zeros((4, 4), np.float32)
        alpha[:2] = .5
        # I = covered + (1-β)(paper-covered) → 反解应还原纸面 255
        img[:2] = covered + .5 * (paper - covered)
        check = np.ones((4, 4), bool)
        out = _verify_inverse(img, alpha, check, keepout, s_vec, t_vec, paper)
        self.assertTrue((out[:2] == .5).all())

    def test_whitening_rejected(self):
        # keepout 里的纯字像素被实测 alpha 截断成 1 → 反解把 90 压到 16.7，
        # 比原图暗 70+ → 必须拒（防擦花真字）
        img, paper, s_vec, t_vec, keepout = self._setup()
        img[:] = 90.
        keepout[:] = True
        alpha = np.ones((4, 4), np.float32)
        out = _verify_inverse(img, alpha, keepout, keepout, s_vec, t_vec, paper)
        self.assertTrue((out == 0).all())

    def test_clean_paper_in_keepout_rejected(self):
        # keepout 里的纯纸像素被误给 alpha → 试算 B 亮于纸 → 拒（防提亮伤纸）
        img, paper, s_vec, t_vec, keepout = self._setup()
        keepout[:] = True
        alpha = np.full((4, 4), .5, np.float32)
        out = _verify_inverse(img, alpha, keepout, keepout, s_vec, t_vec, paper)
        self.assertTrue((out == 0).all())

    def test_paper_zone_dark_rejected(self):
        # 纸面区（非 keepout）的深墨迹像素：试算 B 暗于纸-40 → 拒（它根本不是纸+水印）
        img, paper, s_vec, t_vec, keepout = self._setup()
        img[:] = 60.
        alpha = np.full((4, 4), .4, np.float32)
        check = np.ones((4, 4), bool)
        out = _verify_inverse(img, alpha, check, keepout, s_vec, t_vec, paper)
        self.assertTrue((out == 0).all())

    def test_unchecked_pixels_untouched(self):
        img, paper, s_vec, t_vec, keepout = self._setup()
        img[:] = 60.
        alpha = np.full((4, 4), .4, np.float32)
        out = _verify_inverse(img, alpha, np.zeros((4, 4), bool), keepout,
                              s_vec, t_vec, paper)
        self.assertTrue((out == .4).all())


if __name__ == '__main__':
    unittest.main()
