#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打包"不可再生"的锁定数据集用于发布（2026-11-06 口径）。

这些数据集无法从仓库内脚本重建，字节一致性就是它们的价值：
- watermark_regression_min_v1：固定哨兵回归集；字节一变，全部封存基线失去可比性。
- watermark_independent_v1：锁定的独立验证集（抓取脚本按设计永远抓新图，不可能重建同一套）。
- 应用测试输入：用户素材整理集，无任何生成脚本。

训练语料 watermark_training_v1 不在此列：它由 build_training_corpus.py +
fetch_cc0_training_bases.py 固定种子重建（见 DATA_PACKAGE_README.txt 重建指南）。

每个数据集：先过各自的 QA 门槛 → 在数据集目录内写 SHA256SUMS.txt →
打成 zip（解压到仓库根目录即落位 data/<名称>/）→ CRC 自检 →
临时目录解压并逐文件回验哈希。

用法：.venv/bin/python tools/package_locked_datasets.py
产出：out/数据发布包_20261106/<名称>.zip
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out' / '数据发布包_20261106'
SKIP_PARTS = {'__pycache__'}
SKIP_NAMES = {'.DS_Store', 'SHA256SUMS.txt'}


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def check(dataset: Path) -> None:
    name = dataset.name
    if name == 'watermark_regression_min_v1':
        qa = json.loads((dataset / 'qa/validation.json').read_text())
        assert qa.get('status') == 'passed', '固定回归集 QA 未通过，禁止打包'
    elif name == 'watermark_independent_v1':
        qa = json.loads((dataset / 'qa/validation.json').read_text())
        assert qa.get('pass_all') is True, '独立集 QA 未通过，禁止打包'
    elif name == '应用测试输入':
        assert (dataset / '输入来源与校验.json').is_file(), '应用测试输入缺少来源校验文件'
    else:
        raise ValueError(f'未登记的数据集：{name}')
    forbidden = [p for p in dataset.rglob('*') if p.is_file()
                 and ('restricted_originals/' in str(p) or 'douyin' in p.name.lower())]
    assert not forbidden, f'包含受限原件：{forbidden[:3]}'


def package(dataset: Path) -> Path:
    check(dataset)
    files = sorted(p for p in dataset.rglob('*') if p.is_file()
                   and p.name not in SKIP_NAMES and not SKIP_PARTS & set(p.parts))
    inventory = {str(p.relative_to(dataset)): digest(p) for p in files}
    (dataset / 'SHA256SUMS.txt').write_text(
        ''.join(f'{h}  {n}\n' for n, h in sorted(inventory.items())), 'utf-8')
    files.append(dataset / 'SHA256SUMS.txt')

    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / f'{dataset.name}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for p in files:
            z.write(p, arcname=str(Path('data') / dataset.name / p.relative_to(dataset)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, 'CRC 自检失败'
        with tempfile.TemporaryDirectory(prefix='wm_locked_verify_') as tmp:
            z.extractall(tmp)
            extracted = Path(tmp) / 'data' / dataset.name
            for name, h in inventory.items():
                assert digest(extracted / name) == h, f'解压回验失败：{name}'
    print(f'{dataset.name}: {len(files)} 个文件，{archive.stat().st_size / 1e6:.1f} MB -> {archive}')
    return archive


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for name in ('watermark_regression_min_v1', 'watermark_independent_v1', '应用测试输入'):
        package(ROOT / 'data' / name)


if __name__ == '__main__':
    sys.exit(main())
