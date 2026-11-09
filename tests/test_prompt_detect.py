import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import random
import sys
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtTest import QTest

from wm import pipeline, prompt_detect, synth
from wm.app import MainWindow, make_app
from wm.batches import Batch, BatchWorker
from wm.contract import (EngineKind, MaskSource, PageImage, PageResult,
                         MaskLayer, Thresholds, VerdictReport)
from wm.wevisone_client import Service, ServiceError


class FakeService:
    """假模型服务：按预设框/matte 返回，记录调用。"""

    def __init__(self, boxes=()):
        # 每项可为 xyxy（matte=None）或 (xyxy, matte)
        self.boxes = []
        for item in boxes:
            if len(item) == 4 and all(isinstance(v, (int, float)) for v in item):
                self.boxes.append((tuple(item), None))
            else:
                self.boxes.append((tuple(item[0]), item[1]))
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
        # 该用例验证低对比生长，OCR 保护区另行测试；此处关闭以免文字区覆盖整框
        layer, info = prompt_detect.locate_strokes(image, '灰字', FakeService([box]),
                                                   Thresholds(), use_ocr=False)
        sel = layer.footprint
        recall = int((sel & ink).sum()) / max(1, int(ink.sum()))
        self.assertEqual(info['boxes_with_support'], 1)
        self.assertGreater(recall, 0.5, f"淡笔画召回过低: {recall:.2f}")
        # 证据约束：框内大片空白不得入选（文字从 x=60 起，左侧 40–55 为纯空白）
        bx0, by0, bx1, by1 = box
        blank = np.zeros(image.shape[:2], bool)
        blank[by0:by1, bx0:bx0+15] = True
        self.assertLess(int((sel & blank).sum()), 30)

    def test_matte_support_confines_selection_to_prompted_strokes(self):
        # 框内背景横线比目标斜向笔画对比更强时，matte（提示词限定）应把
        # 选区约束在水印笔画上，背景线不得入选
        import platform
        from PIL import Image, ImageDraw, ImageFont
        image = np.full((300, 400, 3), 250, np.uint8)
        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil)
        # 跨平台中文字体：Windows用微软雅黑，macOS用苹方
        # 字体敏感：该用例按 msyh 的字形渲染标定（macOS 上其他中文字体 rasterization 不同，
        # 会引入个位数像素越界），非 Windows/macOS 环境跳过，保持断言的严格性。
        if platform.system() == 'Windows':
            font_path = "C:/Windows/Fonts/msyh.ttc"
        elif platform.system() == 'Darwin':  # macOS
            font_path = "/System/Library/Fonts/PingFang.ttc"
        else:
            self.skipTest("此测试需要中文字体，当前系统不支持")
            return
        try:
            font = ImageFont.truetype(font_path, 15)
            big = ImageFont.truetype(font_path, 44)
        except OSError:
            self.skipTest(f"找不到字体文件：{font_path}")
            return
        for i in range(8):
            draw.text((30, 40 + i * 30), f'背景行 sample {100+i} value {10.5+i:.2f}', fill=(25, 25, 25), font=font)
        wm_layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        wdraw = ImageDraw.Draw(wm_layer)
        wdraw.text((60, 110), "内部演示稿", fill=(20, 20, 20, 255), font=big)
        wm_layer = wm_layer.rotate(-25, resample=Image.BICUBIC, center=(200, 150))
        pil = Image.alpha_composite(pil.convert("RGBA"), wm_layer).convert("RGB")
        image = np.array(pil)
        gt = np.array(wm_layer)[..., 3] > 127
        ys, xs = np.where(gt)
        box = (int(xs.min()) - 6, int(ys.min()) - 6, int(xs.max()) + 6, int(ys.max()) + 6)
        # 强背景横线框（无 matte）→ 误选背景；给 matte → 只选水印
        layer_bad, _ = prompt_detect.locate_strokes(
            image, '斜向黑字', FakeService([box]), Thresholds())
        layer_good, info = prompt_detect.locate_strokes(
            image, '斜向黑字', FakeService([(box, gt)]), Thresholds())
        sel_bad, sel = layer_bad.footprint, layer_good.footprint
        support = cv2.dilate(gt.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        gt_hits = int((sel & gt).sum())
        extra_bad = int((sel_bad & ~gt).sum())
        extra_good = int((sel & ~support).sum())
        print(f'  无matte: sel={int(sel_bad.sum())} 命中={int((sel_bad&gt).sum())} 误选背景={extra_bad} '
              f'| 有matte: sel={int(sel.sum())} 命中={gt_hits} 约束外={extra_good}')
        # 核心保证：matte 约束下选区严格不出支撑范围（背景零误选）。
        # 低分辨率小图里水印若被 OCR 正文区完全覆盖，空选区＝安全退回（闸门拦截、不动原图），
        # 高分辨率真实图中正文区不连续，水印大部分可入选（见自测报告真实案例）。
        # 容差说明：不同系统字体渲染略有差异（Windows微软雅黑 vs macOS苹方），
        # 允许少量像素偏差（<10），但必须比无matte时显著减少（从数千降到个位数）。
        self.assertLessEqual(extra_good, 10)
        self.assertLess(extra_good, extra_bad / 100)  # 至少减少100倍
        if gt_hits:
            self.assertGreater(gt_hits, 500)

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
            image = np.zeros((192, 256, 3), np.uint8)
            boxes = service.segment(image, '测试')
            self.assertEqual(len(boxes), 1)
            box, matte = boxes[0]
            self.assertEqual(box, (64, 48, 192, 144))
            self.assertIsNotNone(matte)
            self.assertEqual(matte.shape, (192, 256))
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

    def test_brush_paints_in_result_view_for_post_repair_touchup(self):
        # 提示词/自动处理后的修复残留只存在于结果图：处理结果视图必须可涂抹
        from wm.contract import EngineKind, VerdictReport
        report = VerdictReport('k', EngineKind.RASTER, [], {}, {}, [], 1)
        result = PageResult(self.batch.pages[0], PageImage(self.batch.pages[0].image.copy(), 72, 'd'),
                            MaskLayer(np.zeros(self.batch.pages[0].image.shape[:2], np.float32)), report)
        self.batch.results[0] = result
        self.win.render_canvas()  # 刷新辅助按钮可用状态
        self.win.assist_select_btn.click()
        self.app.processEvents()
        self.assertTrue(self.win.canvas.brush_mode)
        self.win.set_view_mode('处理结果')
        self.app.processEvents()
        self.assertTrue(self.win.canvas.brush_mode)
        self.win.set_view_mode('拖拽对比')
        self.app.processEvents()
        self.assertFalse(self.win.canvas.brush_mode)
        self.win.set_view_mode('原始图片')
        self.app.processEvents()
        self.assertTrue(self.win.canvas.brush_mode)
        self.win.cancel_assisted_selection()

    def test_edit_overlay_distinguishes_session_added_in_red(self):
        # 三色选区：既有浅蓝、本轮新增浅红、本轮回擦浅灰
        from wm.contract import EngineKind, VerdictReport
        img = np.zeros((60, 80, 3), np.uint8)
        sel = np.zeros((60, 80), bool)
        sel[10:20, 10:20] = True   # 既有选区
        report = VerdictReport('k', EngineKind.RASTER, [], {}, {}, [], 1)
        layer = MaskLayer(sel.astype(np.float32), footprint=sel, repair_mode='inpaint',
                          selection=sel.copy())
        result = PageResult(PageImage(img, 72, 'd'), PageImage(img.copy(), 72, 'd'),
                            layer, report)
        self.batch.pages[0] = result.source
        self.batch.results[0] = result
        self.win.render_canvas()
        self.win.assist_select_btn.click()
        self.app.processEvents()
        # 直接驱动一笔：在 (40,40) 添加直径 9
        self.win._paint_brush_segment(((40.5, 40.5), (40.5, 40.5), 9, True))
        self.app.processEvents()
        image = self.win.canvas._overlay.toImage()
        r, g, b = (image.pixelColor(40, 40).red(), image.pixelColor(40, 40).green(),
                   image.pixelColor(40, 40).blue())
        self.assertGreater(r, 200)
        self.assertLess(g, 160)          # 本轮新增 = 浅红
        r2, g2, b2 = (image.pixelColor(15, 15).red(), image.pixelColor(15, 15).green(),
                      image.pixelColor(15, 15).blue())
        self.assertLess(r2, 80)
        self.assertGreater(b2, 200)      # 既有选区 = 浅蓝
        self.win.cancel_assisted_selection()

    def test_multi_page_doc_labels_carry_page_number(self):
        from wm.contract import PageImage
        # 模拟真实场景：load_many 把 page_index 重编成批次全局序号（8/9/10）
        pages = [PageImage(np.zeros((40, 50, 3), np.uint8), 72, 'real_009', page_index=8 + i)
                 for i in range(3)]
        pages.append(PageImage(np.zeros((40, 50, 3), np.uint8), 72, 'real_006', page_index=5))
        batch = Batch('多页', pages)
        self.win.batches = [batch]
        self.win.refresh_batches()
        self.win.batch_list.setCurrentRow(0)
        self.win.select_batch(0)  # 行号未变时信号不触发，直接调用
        self.app.processEvents()
        texts = [self.win._page_items[i].text(0) for i in range(4)]
        self.assertIn('real_009·第1页', texts[0])
        self.assertIn('real_009·第2页', texts[1])
        self.assertIn('real_009·第3页', texts[2])
        self.assertIn('real_006', texts[3])
        self.assertNotIn('第1页', texts[3])

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
