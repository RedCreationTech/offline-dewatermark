# -*- coding: utf-8 -*-
"""效果定档与归档的单元测试。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import archive_classified


def _png(path: Path) -> None:
    from PIL import Image
    Image.new('RGB', (8, 8), 'white').save(path)


class ArchiveClassifiedTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.run_dir = root / '轮次'
        self.run_dir.mkdir()
        src = root / 'src'
        src.mkdir()
        _png(src / 'a.png')
        for name in ('原图预览.jpg', '处理结果.png', '选区预览.jpg', '漏选多选核对.jpg'):
            _png(self.run_dir / name)
        records = [dict(key='desktop:a:1', id='a', page=1, mode='desktop', status='completed',
                        negative=False, files={name: str(self.run_dir / name)
                                               for name in ('原图预览.jpg', '处理结果.png',
                                                            '选区预览.jpg', '漏选多选核对.jpg')}),
                   dict(key='desktop:b:1', id='b', page=1, mode='desktop', status='completed',
                        negative=False, files={})]
        (self.run_dir / '运行记录.json').write_text(
            json.dumps(dict(records=records), ensure_ascii=False), encoding='utf-8')
        (self.run_dir / '样本清单.json').write_text(json.dumps(
            dict(cases=[dict(id='a', input=str(src / 'a.png')), dict(id='b', input=str(src / 'a.png'))]),
            ensure_ascii=False), encoding='utf-8')
        self.dataset = root / '数据集'

    def tearDown(self):
        self.tmp.cleanup()

    def _payload(self, classes: dict[str, str]) -> dict:
        return dict(schema_version=1, run_dir=str(self.run_dir), record_sha256='x' * 64,
                    report_title='t', preclass_thresholds={'recall': 0.98, 'precision': 0.90},
                    samples=[dict(key='desktop:a:1', id='a', page=1, group='组1', style='s',
                                  label='l', negative=False, status='completed',
                                  recall=1.0, precision=0.95, machine_candidate=True,
                                  user_class=classes.get('a')),
                             dict(key='desktop:b:1', id='b', page=1, group='组1', style='s',
                                  label='l', negative=False, status='completed',
                                  recall=0.4, precision=0.6, user_class=classes.get('b'))])

    def test_archive_copies_files_and_writes_verdict(self):
        rc = archive_classified.archive(self._payload({'a': '效果好', 'b': '待解决'}), self.dataset, False)
        self.assertEqual(rc, 0)
        good = self.dataset / '效果好' / '组1' / 'a_p1'
        self.assertTrue((good / '判定记录.json').is_file())
        self.assertTrue((good / '输入.png').is_file())
        self.assertTrue((good / '处理结果.png').is_file())
        verdict = json.loads((good / '判定记录.json').read_text(encoding='utf-8'))
        self.assertEqual(verdict['user_class'], '效果好')
        self.assertTrue(verdict['machine_candidate'])
        bad = self.dataset / '待解决' / '组1' / 'b_p1'
        self.assertTrue((bad / '判定记录.json').is_file())

    def test_unassigned_fails(self):
        with self.assertRaises(SystemExit):
            archive_classified.archive(self._payload({'a': '效果好'}), self.dataset, False)

    def test_reclass_requires_force_and_keeps_history(self):
        archive_classified.archive(self._payload({'a': '效果好', 'b': '待解决'}), self.dataset, False)
        with self.assertRaises(SystemExit):
            archive_classified.archive(self._payload({'a': '待解决', 'b': '待解决'}), self.dataset, False)
        archive_classified.archive(self._payload({'a': '待解决', 'b': '待解决'}), self.dataset, True)
        verdict = json.loads((self.dataset / '待解决' / '组1' / 'a_p1' / '判定记录.json')
                             .read_text(encoding='utf-8'))
        self.assertEqual(verdict['user_class'], '待解决')
        self.assertEqual(verdict['变更历史'][-1]['user_class'], '效果好')
        # 改档后旧分类目录不再保留同一输入
        self.assertFalse((self.dataset / '效果好' / '组1' / 'a_p1').exists())


if __name__ == '__main__':
    unittest.main()
