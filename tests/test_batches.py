import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import threading
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from wm.app import MainWindow, make_app, OPEN_FILTER
from wm.batches import Batch, BatchWorker
from wm.contract import PageImage, MaskLayer, Thresholds


def page(index, shape=(20, 30, 3)):
    return PageImage(np.full(shape, 240, np.uint8), 200, 'sample', index)


def result(src, mask, thr, **_kwargs):
    return SimpleNamespace(source=src, mask=mask, verdict=SimpleNamespace(
        passed=True, review_needed=True, stage_ms={}, elapsed_ms=1))


class QueueTests(unittest.TestCase):
    def test_single_target_uses_whole_size_group_and_reuses_prior(self):
        batch = Batch('a', [page(0), page(1), page(2, (10, 10, 3))])
        mask = MaskLayer(np.zeros((20, 30), np.float32))
        jobs = [(batch, [1], Thresholds(), 11, False)]
        with patch('wm.batches.masks.estimate_crosspage', return_value=mask) as prior, \
             patch('wm.batches.pipeline._process_one', side_effect=result) as repair:
            BatchWorker(jobs).run()
            BatchWorker(jobs).run()
            self.assertEqual(prior.call_count, 1)
            self.assertEqual(len(prior.call_args.args[0]), 2)
            self.assertIs(repair.call_args.args[0], batch.pages[1])
            batch.invalidate()
            BatchWorker(jobs).run()
            self.assertEqual(prior.call_count, 2)

    def test_mixed_sizes_isolation_error_continuation_and_immediate_updates(self):
        a = Batch('a', [page(0), page(1), page(2, (10, 10, 3))])
        b = Batch('b', [page(0), page(1)])
        events = []
        def process(src, mask, thr, **_kwargs):
            # First result must have been emitted before starting the second image.
            if src is a.pages[1]:
                self.assertEqual(len([e for e in events if e['kind']=='page']), 1)
                raise ValueError('bad image')
            return result(src, mask, thr)
        worker = BatchWorker([(a, [0,1,2], Thresholds(), 11, False),
                              (b, [0,1], Thresholds(), 11, False)])
        worker.update.connect(events.append)
        with patch('wm.batches.masks.estimate_crosspage', side_effect=lambda imgs,*a,**kw: MaskLayer(np.zeros(imgs[0].shape[:2]))) as prior, \
             patch('wm.batches.masks.estimate_single_page', return_value=MaskLayer(np.zeros((10,10)))) as single, \
             patch('wm.batches.pipeline._keepout', return_value=None), \
             patch('wm.batches.pipeline._process_one', side_effect=process):
            worker.run()
        self.assertEqual(prior.call_count, 2)
        self.assertEqual(single.call_count, 5)  # every page retains its own baseline
        pages = [e for e in events if e['kind']=='page']
        self.assertEqual([e['done'] for e in pages], [1,2,3,4,5])
        self.assertEqual(pages[1]['error'], 'bad image')

    def test_resume_releases_pause(self):
        worker = BatchWorker([])
        worker.pause(True)
        reached = threading.Event()
        worker.update.connect(lambda e: reached.set(), Qt.DirectConnection)
        allowed = []
        thread = threading.Thread(target=lambda: allowed.append(worker._checkpoint()), daemon=True)
        thread.start()
        self.assertTrue(reached.wait(2))
        worker.pause(False)
        thread.join(2)
        self.assertEqual(allowed, [True])

    def test_pause_resume_and_stop_at_page_boundary(self):
        batch = Batch('a', [page(0), page(1)])
        worker = BatchWorker([(batch,[0,1],Thresholds(),11,False)])
        worker.pause(True)
        reached = threading.Event()
        worker.update.connect(lambda e: reached.set() if e['kind']=='paused' else None, Qt.DirectConnection)
        thread = threading.Thread(target=worker.run, daemon=True)
        thread.start()
        self.assertTrue(reached.wait(2))
        self.assertEqual(worker.done, 0)
        worker.stop()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        worker = BatchWorker([(batch,[0,1],Thresholds(),11,False)])
        worker.update.connect(lambda e: worker.stop() if e['kind']=='page' else None)
        with patch('wm.batches.masks.estimate_crosspage', return_value=MaskLayer(np.zeros((20,30)))), \
             patch('wm.batches.pipeline._process_one', side_effect=result):
            worker.run()
        self.assertEqual(worker.done, 1)


class UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = make_app()

    def test_parameter_isolation_invalidation_and_busy_lock(self):
        w = MainWindow()
        w.batches = [Batch('a',[page(0)]), Batch('b',[page(0)])]
        w.refresh_batches()
        w.batch_list.setCurrentRow(0)
        w._sliders['close_px'].setValue(15)
        w.batch.priors['test'] = object()
        w._sliders['lama_context_margin'].setValue(320)
        self.assertIn('test', w.batch.priors)
        w._sliders['close_px'].setValue(17)
        self.assertFalse(w.batch.priors)
        w.batch_list.setCurrentRow(1)
        self.assertEqual(w.current_close_px(), 11)
        w.batch_list.setCurrentRow(0)
        self.assertEqual(w.current_close_px(), 17)
        w.set_busy(True)
        self.assertFalse(w.param_box.isEnabled())
        self.assertFalse(w.open_btn.isEnabled())
        self.assertTrue(w.batch_list.isEnabled())
        w.set_busy(False)
        w.close()

    def test_export_current_image_keeps_other_results_dirty(self):
        from wm.contract import PageResult, VerdictReport, EngineKind
        w = MainWindow()
        batch = Batch('a/b', [page(0), page(1)])
        for i, src in enumerate(batch.pages):
            report = VerdictReport(src.key, EngineKind.RASTER, [], {}, {}, [], 1)
            batch.results[i] = PageResult(src, src, MaskLayer(np.zeros(src.image.shape[:2])), report)
            batch.dirty.add(i)
        w.batches = [batch]
        w.refresh_batches()
        w.batch_list.setCurrentRow(0)
        w._select_page(1)
        with tempfile.TemporaryDirectory() as tmp, \
             patch('wm.app.QInputDialog.getItem', return_value=('当前图片', True)), \
             patch('wm.app.QFileDialog.getExistingDirectory', return_value=tmp):
            w.export_results()
            self.assertEqual(len(list(Path(tmp).rglob('*_clean.png'))), 1)
            self.assertEqual(len(list(Path(tmp).rglob('queue.json'))), 1)
            self.assertEqual(batch.dirty, {0})
        batch.dirty.clear()
        w.close()


class OptionalPriorTests(unittest.TestCase):
    def test_bad_prior_does_not_block_single_image_processing(self):
        batch=Batch('a',[page(0),page(1)])
        single=MaskLayer(np.zeros((20,30),np.float32))
        worker=BatchWorker([(batch,[0,1],Thresholds(),11,False)])
        with patch('wm.batches.masks.estimate_crosspage',side_effect=ValueError('unusable prior')) as prior, \
             patch('wm.batches.masks.estimate_single_page',return_value=single), \
             patch('wm.batches.pipeline._process_one',side_effect=result), \
             self.assertLogs('wm.batches',level='WARNING'):
            worker.run()
        self.assertEqual(worker.done,2)


class ImportTests(unittest.TestCase):
    """目录导入（含子目录）与图片列表子目录树形展示。"""

    @classmethod
    def setUpClass(cls):
        cls.app = make_app()

    @staticmethod
    def _png(path: Path, color=(30, 60, 90)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((6, 8, 3), color, np.uint8)).save(path)

    def _tree(self, root: Path):
        # root/img1.jpg、root/a/doc.pdf、root/b/img2.png、root/b/img10.png、root/c/skip.txt
        import pymupdf as fitz
        self._png(root / 'img1.jpg')
        self._png(root / 'b' / 'img2.png')
        self._png(root / 'b' / 'img10.png')
        (root / 'a').mkdir(parents=True, exist_ok=True)
        doc = fitz.open()
        doc.new_page(width=72, height=72)
        doc.save(str(root / 'a' / 'doc.pdf'))
        doc.close()
        (root / 'c').mkdir(parents=True, exist_ok=True)
        (root / 'c' / 'skip.txt').write_text('not importable', 'utf-8')

    def test_collect_import_files_recurses_and_sorts_naturally(self):
        from wm import documents
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root)
            files = documents.collect_import_files(root)
            rel = [p.relative_to(root).as_posix() for p in files]
        self.assertEqual(rel, ['a/doc.pdf', 'b/img2.png', 'b/img10.png', 'img1.jpg'])

    def test_collect_import_files_empty_directory(self):
        from wm import documents
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(documents.collect_import_files(tmp), [])

    def test_common_root_picks_innermost_shared_directory(self):
        from wm import documents
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root)
            same = documents.common_root([root / 'b' / 'img2.png', root / 'b' / 'img10.png'])
            self.assertEqual(same, root / 'b')
            nested = documents.common_root([root / 'a' / 'doc.pdf', root / 'b' / 'img2.png'])
            self.assertEqual(nested, root)
            single = documents.common_root([root / 'img1.jpg'])
            self.assertEqual(single, root)

    def test_assign_relative_dirs_marks_subfolder_pages(self):
        from wm import documents
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'sub' / 'deep').mkdir(parents=True)
            pages = [
                PageImage(np.zeros((4, 4, 3), np.uint8), 72, 'p0', source_path=str(root / 'p0.png')),
                PageImage(np.zeros((4, 4, 3), np.uint8), 72, 'p1', source_path=str(root / 'sub' / 'p1.png')),
                PageImage(np.zeros((4, 4, 3), np.uint8), 72, 'p2', source_path=str(root / 'sub' / 'deep' / 'p2.png')),
                PageImage(np.zeros((4, 4, 3), np.uint8), 72, 'p3', source_path=str(root.parent / 'elsewhere' / 'p3.png')),
            ]
            documents.assign_relative_dirs(pages, root)
        self.assertEqual([p.source_rel_dir for p in pages], ['', 'sub', 'sub/deep', ''])

    def test_page_tree_groups_by_subfolder_and_toggles(self):
        w = MainWindow()
        try:
            def mk(rel, name, sp=None):
                p = PageImage(np.zeros((4, 4, 3), np.uint8), 72, name)
                p.source_rel_dir, p.source_path = rel, sp
                return p
            batch = Batch('t', [
                mk('', '根图'), mk('子目录', '重名', '/r/子目录/重名.png'),
                mk('子目录/深层', '重名', '/r/子目录/深层/重名.png'), mk('a/b', 'x')])
            w.batches = [batch]
            w.refresh_batches()
            w.batch_list.setCurrentRow(0)
            w.select_batch(0)
            root = w.page_list.invisibleRootItem()
            top = [root.child(i).text(0).split('\n')[0] for i in range(root.childCount())]
            self.assertEqual(top, ['01  根图', '📁 子目录 · 2 张', '📁 a · 1 张'])
            folder = root.child(1)
            self.assertFalse(bool(folder.flags() & Qt.ItemIsSelectable))
            self.assertTrue(folder.isExpanded())
            w._toggle_folder(folder)
            self.assertFalse(folder.isExpanded())
            w._toggle_folder(folder)
            self.assertTrue(folder.isExpanded())
            # 不同子目录的同名文件不算同一文档，不显示"第N页"
            self.assertNotIn('第', w._page_items[1].text(0))
            self.assertNotIn('第', w._page_items[2].text(0))
            w._select_page(2)
            self.assertEqual(w._current, 2)
            self.assertIn('重名', w.page_list.currentItem().text(0))
        finally:
            w.close()

    def test_open_file_directory_import_loads_all_subfolders(self):
        from wm import documents
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root)
            w = MainWindow()
            try:
                with patch.object(MainWindow, '_ask_import_mode', return_value='dir'), \
                     patch('wm.app.QFileDialog.getExistingDirectory', return_value=str(root)), \
                     patch('wm.app.QMessageBox.warning'):
                    w.open_file()
                self.assertEqual(len(w.batches), 1)
                batch = w.batches[0]
                self.assertIn(root.name, batch.name)
                rels = [p.source_rel_dir for p in batch.pages]
                self.assertEqual(sorted(set(rels)), ['', 'a', 'b'])
                top = [w.page_list.invisibleRootItem().child(i).text(0).split('\n')[0]
                       for i in range(w.page_list.invisibleRootItem().childCount())]
                self.assertIn('📁 b · 2 张', top)
            finally:
                w.close()

    def test_open_button_click_does_not_treat_checked_false_as_path(self):
        # 回归：clicked 信号会带 checked=False，直连 open_file 时不能把它当文件路径。
        w = MainWindow()
        try:
            with patch.object(MainWindow, '_ask_import_mode', return_value=None), \
                 patch('wm.app.QMessageBox.warning') as warn:
                w.open_btn.click()
            self.assertEqual(w.batches, [])
            self.assertEqual(warn.call_count, 0)
        finally:
            w.close()

    def test_open_file_empty_directory_warns_and_adds_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = MainWindow()
            try:
                with patch.object(MainWindow, '_ask_import_mode', return_value='dir'), \
                     patch('wm.app.QFileDialog.getExistingDirectory', return_value=tmp), \
                     patch('wm.app.QMessageBox.warning') as warn:
                    w.open_file()
                self.assertEqual(w.batches, [])
                self.assertEqual(warn.call_count, 1)
            finally:
                w.close()

    def test_open_file_files_mode_multi_directory_shows_relative_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._png(root / 'a' / 'one.png', (1, 2, 3))
            self._png(root / 'b' / 'two.png', (4, 5, 6))
            w = MainWindow()
            try:
                picked = [str(root / 'a' / 'one.png'), str(root / 'b' / 'two.png')]
                with patch.object(MainWindow, '_ask_import_mode', return_value='files'), \
                     patch('wm.app.QFileDialog.getOpenFileNames', return_value=(picked, OPEN_FILTER)), \
                     patch('wm.app.QMessageBox.warning'):
                    w.open_file()
                self.assertEqual(len(w.batches), 1)
                self.assertIn('one', w.batches[0].name)
                self.assertEqual([p.source_rel_dir for p in w.batches[0].pages], ['a', 'b'])
            finally:
                w.close()

    def test_append_files_directory_keeps_subfolder_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._png(root / 'old.png', (9, 9, 9))
            extra = root / 'extra'
            self._png(extra / 'sub' / 'new.png', (7, 7, 7))
            w = MainWindow()
            try:
                w.open_file(str(root / 'old.png'))
                self.assertEqual(w.batches[0].pages[0].source_rel_dir, '')
                with patch.object(MainWindow, '_ask_import_mode', return_value='dir'), \
                     patch('wm.app.QFileDialog.getExistingDirectory', return_value=str(extra)), \
                     patch('wm.app.QMessageBox.warning'):
                    w.append_files()
                pages = w.batches[0].pages
                self.assertEqual(len(pages), 2)
                self.assertEqual(pages[1].source_rel_dir, 'sub')
                top = [w.page_list.invisibleRootItem().child(i).text(0).split('\n')[0]
                       for i in range(w.page_list.invisibleRootItem().childCount())]
                self.assertIn('📁 sub · 1 张', top)
            finally:
                w.close()

if __name__ == '__main__':
    unittest.main()
