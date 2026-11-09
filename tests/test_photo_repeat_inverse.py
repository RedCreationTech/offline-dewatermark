import unittest
from unittest.mock import patch

import cv2
import numpy as np

from wm import photo_repeat_inverse, lama, pipeline
from wm.contract import MaskLayer, Thresholds


def sample(strength=.16):
    h, w = 600, 720
    yy, xx = np.indices((h, w))
    gray = np.uint8(65+120*xx/w+35*yy/h)
    clean = np.repeat(gray[..., None], 3, axis=2)
    cv2.putText(clean, 'KEEP ARROWS >', (120, 295), cv2.FONT_HERSHEY_SIMPLEX,
                .9, (20, 20, 20), 2, cv2.LINE_AA)
    strokes = np.zeros((h, w), np.uint8)
    for y in range(60, 600, 110):
        for x in range(10, 700, 175):
            cv2.putText(strokes, 'FX42', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, 255, 4, cv2.LINE_AA)
    opacity = strokes.astype(np.float32)/255*strength
    source = np.uint8(clean*(1-opacity[..., None])+np.array([40, 75, 140])*opacity[..., None])
    coarse = cv2.dilate(strokes, np.ones((5, 5), np.uint8)) > 0
    return clean, source, coarse, opacity


class PhotoRepeatInverseTests(unittest.TestCase):
    def test_measured_inverse_reduces_faint_repeats_without_repainting_scene(self):
        clean, source, coarse, truth = sample()
        original = source.copy()
        mask = photo_repeat_inverse.estimate(source, coarse)
        self.assertIsNotNone(mask)
        with patch('wm.lama.inpaint', side_effect=AssertionError('must not repaint')):
            output, used = lama.repair_with_lama(source, mask, Thresholds())
        marked = truth > 0
        before = np.abs(source.astype(float)-clean).mean(2)
        after = np.abs(output.astype(float)-clean).mean(2)
        self.assertLess(after[marked].mean(), before[marked].mean()*.65)
        self.assertLess(after[~marked].mean(), .3)
        np.testing.assert_array_equal(output[~mask.footprint], source[~mask.footprint])
        np.testing.assert_array_equal(original, source)

    def test_normal_scene_and_strong_watermarks_decline(self):
        clean, _, coarse, _ = sample()
        self.assertIsNone(photo_repeat_inverse.estimate(clean, coarse))
        _, heavy, coarse, _ = sample(.65)
        self.assertIsNone(photo_repeat_inverse.estimate(heavy, coarse))
        self.assertIsNone(photo_repeat_inverse.estimate(clean, np.zeros(coarse.shape, bool)))

    def test_inverse_with_invalid_model_falls_back_to_neural_inpaint(self):
        # 实测反解模型无效（缺透明度/含 NaN）时不整页报错，退回保守的 LaMa 补图。
        image = np.full((20, 30, 3), 100, np.uint8)
        region = np.zeros((20, 30), bool)
        region[5:10, 5:10] = True
        mask = MaskLayer(region.astype(np.float32), footprint=region,
                         repair_mode='photo_repeat_inverse', veil_strength=np.zeros(3),
                         veil_offset=np.array([40, 75, 140]))
        with patch('wm.lama.inpaint', return_value=image.copy()) as fake:
            output, used = lama.repair_with_lama(image, mask, Thresholds())
        self.assertEqual(used, [lama.RepairName.LAMA])
        fake.assert_called_once()
        mask.overlay_opacity = np.full(region.shape, .15, np.float32)
        output, _ = lama.repair_with_lama(image, mask, Thresholds())
        np.testing.assert_array_equal(output[~region], image[~region])
        mask.overlay_opacity[0, 0] = np.nan
        with patch('wm.lama.inpaint', return_value=image.copy()) as fake:
            lama.repair_with_lama(image, mask, Thresholds())
        fake.assert_called_once()

    def test_unmeasured_confirmed_strokes_fall_back_to_neural_inpaint(self):
        # 选区内未能实测反解的确认笔画必须补完，不能只留反解核心。
        image = np.full((64, 80, 3), 90, np.uint8)
        region = np.zeros((64, 80), bool)
        region[10:50, 10:70] = True
        opacity = np.zeros((64, 80), np.float32)
        opacity[20:40, 30:50] = .15  # 实测反解核心
        mask = MaskLayer(region.astype(np.float32), footprint=region,
                         overlay_opacity=opacity, veil_strength=np.zeros(3),
                         veil_offset=np.array([40, 75, 140]),
                         repair_mode='photo_repeat_inverse')
        with patch('wm.lama.inpaint', return_value=image.copy()) as fake:
            output, used = lama.repair_with_lama(image, mask, Thresholds())
        self.assertEqual(used, [lama.RepairName.ALPHA_UNMIX, lama.RepairName.LAMA])
        np.testing.assert_array_equal(fake.call_args.args[1], region & ~(opacity > 0))
        # 全部实测时不再额外补图，行为与原来一致。
        mask2 = MaskLayer(region.astype(np.float32), footprint=region,
                          overlay_opacity=region.astype(np.float32) * .15,
                          veil_strength=np.zeros(3),
                          veil_offset=np.array([40, 75, 140]),
                          repair_mode='photo_repeat_inverse')
        with patch('wm.lama.inpaint', return_value=image.copy()) as fake2:
            _, used2 = lama.repair_with_lama(image, mask2, Thresholds())
        fake2.assert_not_called()
        self.assertEqual(used2, [lama.RepairName.ALPHA_UNMIX])

    def test_export_uses_physical_opacity_not_binary_selection(self):
        opacity = np.full((4, 5), .15, np.float32)
        mask = MaskLayer(np.ones((4, 5), np.float32), overlay_opacity=opacity,
                         veil_strength=np.zeros(3), veil_offset=np.array([40, 75, 140]))
        exported = pipeline.extract_watermark(mask)
        np.testing.assert_array_equal(exported[..., 3], np.uint8(opacity*255))


if __name__ == '__main__':
    unittest.main()
