import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import random
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtTest import QTest

from wm import pipeline, prompt_detect, synth
from wm.app import MainWindow, make_app
from wm.batches import Batch, BatchWorker
from wm.contract import (EngineKind, MaskSource, PageImage, PageResult,
                         MaskLayer, Thresholds, VerdictReport)
from wm.wevisone_client import Service, ServiceError


class FakeService:
    """假模型服务：按预设框返回，记录调用。"""

    def __init__(self, boxes=()):
        self.boxes = list(boxes)
        self.calls = []
        self.last_model = 'fake'
        self.last_latency_ms = 1

    @property
    def running(self):
        return True

    def segment(self, image_rgb, query, **kw):
        self.calls.append((image_rgb.shape, query))
        return list(self.boxes)


def synth_page(style='tiled_text', seed=7):
    rng = random.Random(seed)
    clean = synth.render_clean_page(rng, 72, 0)
    alpha, color, strength = synth.make_watermark(clean.shape, style, 72, 0, rng)
    image = synth.apply_watermark(clean, alpha, color, strength)
    return image, alpha


class PromptDetectTests(unittest.TestCase):
    def test_boxes_are_search_ranges_and_hallucination_is_rejected(self):
        # 均匀图上的幻觉框：没有任何像素证据，不得产生选区
        image = np.full((180, 240, 3), 230, np.uint8)
        layer, info = prompt_detect.locate_strokes(image, '水印', FakeService([(4, 4, 220, 220)]),
                                                   Thresholds())
        self.assertEqual(info['boxes_with_support'], 0)
        self.assertFalse(layer.footprint.any())
        self.assertEqual(layer.source, MaskSource.MODEL)
        self.assertEqual(layer.method, prompt_detect.METHOD)
        self.assertEqual(layer.confidence, 0.0)

    def test_real_watermark_inside_box_becomes_strokes(self):
        image, alpha = synth_page()
        ys, xs = np.where(alpha > 0)
        truth = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        layer, info = prompt_detect.locate_strokes(image, '内部资料', FakeService([truth]),
                                                   Thresholds())
        self.assertEqual(info['boxes_with_support'], 1)
        gt = alpha > 0
        tp = int((layer.footprint & gt).sum())
        # 该合成页的模型框＝整页（水印铺满页且压住照片块/表格），按笔画生长会把
        # 与笔画连通的内容区少量卷入：recall 提升是第一目标，precision 允许折中。
        self.assertGreater(tp / gt.sum(), 0.55)
        self.assertGreater(tp / max(1, int(layer.footprint.sum())), 0.15)
        self.assertEqual(layer.repair_mode, 'inpaint')
        # 选区严格在框内（不外推）
        x0, y0, x1, y1 = truth
        ys2, xs2 = np.where(layer.footprint)
        self.assertGreaterEqual(xs2.min(), x0 - 1)
        self.assertLessEqual(xs2.max(), x1 + 1)

    def test_low_contrast_strokes_grow_from_strong_cores(self):
        # 浅灰手写/打印笔画（低于 8% 覆盖阈值）应随强核整体入选
        from PIL import Image, ImageDraw, ImageFont
        image = np.full((240, 320, 3), 245, np.uint8)
        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("arial.ttf", 34)
        except OSError:
            font = ImageFont.load_default()
        draw.text((60, 90), "DRAFT", fill=(205, 205, 205), font=font)
        image = np.array(pil)
        ink = (image.mean(2) < 235)
        box = (40, 60, 300, 160)
        layer, info = prompt_detect.locate_strokes(image, '灰字', FakeService([box]),
                                                   Thresholds())
        sel = layer.footprint
        recall = int((sel & ink).sum()) / max(1, int(ink.sum()))
        self.assertEqual(info['boxes_with_support'], 1)
        self.assertGreater(recall, 0.5, f"淡笔画召回过低: {recall:.2f}")
        # 证据约束：框内大片空白不得入选（文字从 x=60 起，左侧 40–55 为纯空白）
        bx0, by0, bx1, by1 = box
        blank = np.zeros(image.shape[:2], bool)
        blank[by0:by1, bx0:bx0+15] = True
        self.assertLess(int((sel & blank).sum()), 30)

    def test_box_treatment_repaints_whole_box_and_reports_overlap(self):
        image = np.full((240, 320, 3), 245, np.uint8)
        box = (40, 60, 300, 160)
        layer, info = prompt_detect.locate_strokes(image, '框', FakeService([box]),
                                                   Thresholds(), treatment='box')
        self.assertEqual(layer.method, 'prompt_box_repaint')
        self.assertEqual(int(layer.footprint.sum()), (box[2]-box[0]) * (box[3]-box[1]))
        self.assertEqual(info['treatment'], 'box')
        self.assertEqual(info['boxes_with_support'], 1)
        self.assertGreaterEqual(info['text_overlap_ratio'], 0.0)
        # 正文重叠警告信号：OCR 区全覆盖时应为 1
        from wm import ocr
        with patch.object(ocr, 'text_zones', return_value=np.ones(image.shape[:2], bool)):
            _, info2 = prompt_detect.locate_strokes(image, '框', FakeService([box]),
                                                    Thresholds(), treatment='box')
        self.assertEqual(info2['text_overlap_ratio'], 1.0)

    def test_empty_query_and_empty_boxes(self):
        image, _ = synth_page()
        with self.assertRaisesRegex(ValueError, '提示词为空'):
            prompt_detect.locate_strokes(image, '  ', FakeService([(0, 0, 50, 50)]), Thresholds())
        layer, info = prompt_detect.locate_strokes(image, '水印', FakeService([]), Thresholds())
        self.assertEqual(info['model_boxes'], [])
        self.assertFalse(layer.footprint.any())

    def test_out_of_range_box_is_clamped_or_dropped(self):
        image, alpha = synth_page()
        ys, xs = np.where(alpha > 0)
        truth = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        h, w = image.shape[:2]
        huge = [(-50, -50, w + 500, h + 500), (w - 3, 0, w, 2)]  # 第二个夹紧后退化应被丢弃
        layer, info = prompt_detect.locate_strokes(image, 'x', FakeService(huge), Thresholds())
        self.assertEqual(info['boxes_with_support'], 1)
        self.assertTrue(layer.footprint.any())

    def test_empty_mask_is_blocked_by_gate_unchanged(self):
        image = np.full((180, 240, 3), 230, np.uint8)
        layer, _ = prompt_detect.locate_strokes(image, '水印', FakeService([(4, 4, 220, 220)]),
                                                Thresholds())
        page = PageImage(image, 72, 'gate')
        result = pipeline._process_one(page, layer, Thresholds())
        np.testing.assert_array_equal(result.output.image, image)
        self.assertIn('needs_manual_mask', [str(f) for f in result.verdict.failures])
        self.assertTrue(result.verdict.conditions['gate_blocked'])


class ClientProtocolTests(unittest.TestCase):
    def test_stub_server_roundtrip_and_lifecycle(self):
        service = Service(executable=sys.executable, stub=True,
                          log_file=os.devnull)
        try:
            ready = service.start(timeout=30)
            self.assertEqual(ready['status'], 'ready')
            self.assertTrue(service.running)
            self.assertTrue(service.ping())
            image = np.zeros((48, 64, 3), np.uint8)
            boxes = service.segment(image, '测试')
            self.assertEqual(boxes, [(64, 48, 192, 144)])
            service.stop()
            self.assertFalse(service.running)
            with self.assertRaises(ServiceError):
                service.segment(image, '测试')
        finally:
            service.stop()

    def test_missing_venv_reports_clear_error(self):
        service = Service(executable=os.path.join(os.path.dirname(__file__),
                                                  'no_such_python.exe'))
        with self.assertRaisesRegex(ServiceError, '未安装'):
            service.start()


class BatchWorkerPromptTests(unittest.TestCase):
    def _run_batch(self, batch, service, prompt='内部资料'):
        batch.detection_mode = 'prompt'
        batch.prompt_query = prompt
        batch.params = {'mask_alpha_min': 8, 'lama_context_margin': 256,
                        'lama_tile_overlap': 64, 'lama_mask_dilate': 0, 'close_px': 11}
        thr = Thresholds(mask_alpha_min=0.08)
        worker = BatchWorker([(batch, [0], thr, 11, True, None)], service=service)
        # 模拟 app.queue_update 的结果落库（直接 run() 时信号没有接收方）
        def collect(payload):
            if payload.get('kind') != 'page':
                return
            if payload['result'] is not None:
                batch.results[payload['index']] = payload['result']
                batch.errors.pop(payload['index'], None)
            else:
                batch.results.pop(payload['index'], None)
                batch.errors[payload['index']] = payload['error']
        worker.update.connect(collect)
        with patch('wm.pipeline.repair.repair') as fake:
            fake.side_effect = lambda img, mask, t: (img.copy(), [])
            worker.run()
        return worker

    def test_prompt_batch_produces_model_result(self):
        image, alpha = synth_page()
        ys, xs = np.where(alpha > 0)
        truth = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        batch = Batch('p', [PageImage(image, 72, 'd')])
        worker = self._run_batch(batch, FakeService([truth]))
        self.assertEqual(len(worker.outcomes), 1)
        result = batch.results[0]
        self.assertEqual(result.mask.source, MaskSource.MODEL)
        self.assertEqual(result.mask.method, prompt_detect.METHOD)
        self.assertEqual(result.processing['prompt_detection']['boxes_with_support'], 1)
        self.assertEqual(len(result.mask.instances), 1)

    def test_prompt_batch_with_no_evidence_marks_manual(self):
        image = np.full((180, 240, 3), 230, np.uint8)
        batch = Batch('p', [PageImage(image, 72, 'd')])
        worker = self._run_batch(batch, FakeService([(4, 4, 220, 220)]))
        result = batch.results[0]
        np.testing.assert_array_equal(result.output.image, image)
        self.assertIn('needs_manual_mask', [str(f) for f in result.verdict.failures])
        self.assertEqual(worker.outcomes, ['质量未通过'])

    def test_service_failure_marks_page_failed(self):
        class Dead(FakeService):
            @property
            def running(self):
                return False
            def start(self, *a, **kw):
                raise ServiceError('WeVisOne 环境未安装')
        image, _ = synth_page()
        batch = Batch('p', [PageImage(image, 72, 'd')])
        worker = self._run_batch(batch, Dead([]))
        self.assertEqual(worker.outcomes, ['运行失败'])
        self.assertNotIn(0, batch.results)


class PromptPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = make_app()

    def setUp(self):
        self.win = MainWindow()
        self.win.show()
        image = np.zeros((60, 80, 3), np.uint8)
        self.batch = Batch('测试', [PageImage(image, 72, 'd')])
        self.win.batches = [self.batch]
        self.win.refresh_batches()
        self.win.batch_list.setCurrentRow(0)
        self.app.processEvents()

    def tearDown(self):
        self.win.wevisone.stop()
        self.win._stop_wevisone()
        self.win.close()

    def test_panel_groups_and_conditional_controls(self):
        from PySide6.QtWidgets import QGroupBox
        groups = [g.title() for g in self.win.param_box.findChildren(QGroupBox)]
        self.assertEqual(groups, ['找水印', '去水印', '修复底图'])
        self.assertFalse(self.win.service_box.isVisible())
        self.assertFalse(self.win.prompt_edit.isVisible())
        self.assertTrue(self.win.style_box.isVisible())
        self.win.mode_box.setCurrentIndex(1)
        self.app.processEvents()
        self.assertTrue(self.win.service_box.isVisible())
        self.assertTrue(self.win.prompt_edit.isVisible())
        self.assertFalse(self.win.style_box.isVisible())

    def test_mode_and_prompt_persist_per_batch(self):
        self.win.prompt_edit.setText('验证素材 3409')
        self.win.mode_box.setCurrentIndex(1)
        self.win.treatment_box.setCurrentIndex(1)  # 整框重绘
        self.app.processEvents()
        self.assertEqual(self.batch.detection_mode, 'prompt')
        self.assertEqual(self.batch.prompt_query, '验证素材 3409')
        self.assertEqual(self.batch.prompt_treatment, 'box')
        other = Batch('o', self.batch.pages)
        self.win.batches.append(other)
        self.win.refresh_batches()
        self.win.batch_list.setCurrentRow(1)
        self.app.processEvents()
        self.assertEqual(self.win.mode_box.currentData(), 'auto')
        self.assertEqual(self.win.prompt_edit.text(), '')
        self.assertEqual(self.win.treatment_box.currentData(), 'strokes')
        self.win.batch_list.setCurrentRow(0)
        self.app.processEvents()
        self.assertEqual(self.win.mode_box.currentData(), 'prompt')
        self.assertEqual(self.win.prompt_edit.text(), '验证素材 3409')
        self.assertEqual(self.win.treatment_box.currentData(), 'box')

    def test_treatment_control_visible_only_in_prompt_mode(self):
        self.assertFalse(self.win.treatment_box.isVisible())
        self.win.mode_box.setCurrentIndex(1)
        self.app.processEvents()
        self.assertTrue(self.win.treatment_box.isVisible())

    def test_start_run_aborts_without_prompt(self):
        self.win.mode_box.setCurrentIndex(1)
        self.win.prompt_edit.setText('')
        self.app.processEvents()
        self.win.start_run(scope='batch')
        self.assertIsNone(self.win.worker)
        self.assertIn('提示词', self.win.status.text())

    def test_start_run_passes_service_for_prompt_batch(self):
        self.win.prompt_edit.setText('内部资料')
        self.win.mode_box.setCurrentIndex(1)
        self.app.processEvents()
        self.assertEqual(self.batch.detection_mode, 'prompt')
        worker_holder = {}
        original = BatchWorker
        def spy(*a, **kw):
            worker_holder['service'] = kw.get('service')
            worker_holder['created'] = True
            worker = original(*a, **kw)
            worker.start = lambda: None  # 本测试只验证参数传递，不真正跑队列
            worker_holder['worker'] = worker
            return worker
        with patch('wm.wevisone_client.Service.available', return_value=True), \
             patch('wm.app.BatchWorker', side_effect=spy):
            self.win.start_run(scope='batch')
        worker_holder['worker'].deleteLater()
        if self.win.worker is not None:
            self.win.worker = None
        self.assertTrue(worker_holder.get('created'))
        self.assertIs(worker_holder['service'], self.win.wevisone)

    def test_viewer_shows_model_boxes_only_in_selection_view(self):
        image, alpha = synth_page()
        ys, xs = np.where(alpha > 0)
        truth = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        layer, _ = prompt_detect.locate_strokes(image, '内部资料', FakeService([truth]),
                                                Thresholds())
        report = VerdictReport('k', EngineKind.RASTER, [], {}, {}, [], 1)
        result = PageResult(PageImage(image, 72, 'd'),
                            PageImage(image.copy(), 72, 'd'), layer, report)
        self.batch.pages[0] = result.source
        self.batch.results[0] = result
        self.win.set_view_mode('水印选区')
        self.win.show_page(0)
        self.app.processEvents()
        self.assertEqual(self.win.canvas.instance_boxes, [truth])
        self.win.set_view_mode('原始图片')
        self.app.processEvents()
        self.assertEqual(self.win.canvas.instance_boxes, [])
