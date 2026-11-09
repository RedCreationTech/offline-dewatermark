#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""常见水印样本可见度修正重生成工具（2026-11-04）。

只处理合成图片样本（watermark_training_v1 与 watermark_training_increment_v1_1），
不碰 PDF、负例、真实样本和 faint 极淡边界样本。对每个样本用
build_training_corpus.watermark_layer_checked 重新生成水印层：

- 可见度（水印区中位灰度差）>= VISIBLE_FLOOR 的样本保持逐字节不变；
- ADAPTIVE_STYLES 低可见度样本按底图重染颜色；
- center_white 浅底样本改走 center_gray（新样式，文档半透明大字）；
- 被覆盖或替换的旧文件移入 <dataset>/replaced_20261104/，写 receipt.json，
  旧清单记录保留在 receipt 中，manifest.jsonl / splits / build_summary.json 同步更新。

不重新跑任何去水印推理。用法：.venv/bin/python tools/regen_low_visibility_samples.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
STAMP = 'replaced_20261104'
CENTER_GRAY = ('C03', '灰黑半透明大字')


def load_gen(font: Path):
    spec = importlib.util.spec_from_file_location('base_builder', ROOT / 'tools/build_training_corpus.py')
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.FONT = font
    if not font.exists():
        raise FileNotFoundError(f'缺少语料字体：{font}')
    return gen


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_dataset(ds: Path, gen) -> dict:
    manifest_path = ds / 'manifest.jsonl'
    records = [json.loads(x) for x in manifest_path.read_text().splitlines() if x.strip()]
    # v1 用 bases/manifest.json 找底图；增量集底图直接在 bases/ 下。
    base_index = {}
    bm = ds / 'bases/manifest.json'
    if bm.exists():
        for b in json.loads(bm.read_text()):
            base_index[b['id']] = ds / b['path']
    else:
        for p in (ds / 'bases').glob('*.png'):
            base_index[p.stem] = p

    replaced_root = ds / STAMP
    receipt = []
    changed = 0
    untouched = 0
    for rec in records:
        if rec.get('kind') != 'image' or rec.get('is_negative'):
            continue
        style = rec['subtype']
        if style not in gen.ADAPTIVE_STYLES and style != 'center_white':
            continue
        sid = rec['id']
        base_path = base_index.get(rec['base_id'])
        if base_path is None or not base_path.exists():
            receipt.append(dict(id=sid, action='missing_base', base_id=rec['base_id']))
            continue
        base_rgb = Image.open(base_path).convert('RGB')
        final_style, overlay, params, report = gen.watermark_layer_checked(
            base_rgb.size, style, sid, rec['split'], base_rgb)
        if report is None:
            untouched += 1
            continue

        # 目录结构以清单记录为准（v1 带 split 子层，增量集没有），不从约定拼路径。
        folder = (ds / rec['paths']['watermarked']).parent
        watermarked = Image.alpha_composite(base_rgb.convert('RGBA'), overlay).convert('RGB')
        old = {role: ds / rec['paths'][role] for role in ('watermarked', 'overlay_rgba', 'alpha', 'mask')}
        old_sha = {role: digest(p) for role, p in old.items() if p.exists()}

        entry = dict(id=sid, base_id=rec['base_id'], split=rec['split'],
                     style=style, final_style=final_style, report=report, old_sha256=old_sha)

        new_paths = {}
        if final_style == style:
            # 原地更新：旧文件先移入 replaced 目录。
            archive = replaced_root / sid
            for role, p in old.items():
                if p.exists():
                    dst = archive / f'{role}.png'
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(dst))
            overlay.save(folder / 'overlay_rgba.png', compress_level=3)
            overlay.getchannel('A').save(folder / 'alpha.png')
            overlay.getchannel('A').point(lambda a: 255 if a else 0).save(folder / 'mask.png')
            watermarked.save(folder / 'watermarked.png')
            for role in ('watermarked', 'overlay_rgba', 'alpha', 'mask'):
                p = folder / f'{role}.png'
                rec['paths'][role] = str(p.relative_to(ds))
                rec['sha256'][role] = digest(p)
            rec['parameters'] = params
            rec['review_status'] = 'regenerated_20261104_visibility_fix'
            new_paths = rec['paths']
        else:
            # 样式迁移（center_white -> center_gray）：新目录与旧目录同级，旧文件归档。
            new_id = f"{rec['base_id']}__{final_style}"
            new_folder = folder.parent / new_id
            archive = replaced_root / sid
            for role, p in old.items():
                if p.exists():
                    dst = archive / f'{role}.png'
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(dst))
            if folder.exists():
                folder.rmdir()
            new_folder.mkdir(parents=True, exist_ok=True)
            overlay.save(new_folder / 'overlay_rgba.png', compress_level=3)
            overlay.getchannel('A').save(new_folder / 'alpha.png')
            overlay.getchannel('A').point(lambda a: 255 if a else 0).save(new_folder / 'mask.png')
            watermarked.save(new_folder / 'watermarked.png')
            for role in ('watermarked', 'overlay_rgba', 'alpha', 'mask'):
                p = new_folder / f'{role}.png'
                rec['paths'][role] = str(p.relative_to(ds))
                rec['sha256'][role] = digest(p)
            rec['id'] = new_id
            rec['subtype'] = final_style
            rec['label'] = CENTER_GRAY[1]
            rec['parameters'] = params
            rec['review_status'] = 'regenerated_20261104_routed_from_center_white'
            entry['new_id'] = new_id
            new_paths = rec['paths']

        entry['new_paths'] = new_paths
        entry['new_sha256'] = rec['sha256']
        receipt.append(entry)
        changed += 1

    manifest_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), 'utf-8')
    splits_dir = ds / 'splits'
    if splits_dir.exists():
        for split in ('train', 'validation', 'test'):
            (splits_dir / f'{split}.jsonl').write_text(
                ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records if r['split'] == split), 'utf-8')
    summary_path = ds / 'build_summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        styles = summary.get('image_styles', {})
        if 'center_gray' not in styles:
            styles['center_gray'] = list(CENTER_GRAY)
        summary['image_styles'] = styles
        summary['visibility_fix_20261104'] = dict(
            tool='tools/regen_low_visibility_samples.py',
            floor=gen.VISIBLE_FLOOR,
            changed=changed, untouched=untouched,
            receipt=f'{STAMP}/receipt.json',
            note='低可见度样本按底图重染或路由 center_gray；被替换旧文件在 replaced_20261104/，可回滚。')
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), 'utf-8')

    replaced_root.mkdir(parents=True, exist_ok=True)
    (replaced_root / 'receipt.json').write_text(
        json.dumps(dict(dataset=str(ds), changed=changed, untouched=untouched,
                        entries=receipt), ensure_ascii=False, indent=2), 'utf-8')
    return dict(dataset=ds.name, changed=changed, untouched=untouched, receipt_entries=len(receipt))


def main():
    results = []
    for name in ('watermark_training_v1', 'watermark_training_increment_v1_1'):
        ds = DATA / name
        if not (ds / 'manifest.jsonl').exists():
            print('跳过（无清单）', name)
            continue
        font = ds / 'assets/fonts/NotoSansCJKsc-Regular.otf'
        gen = load_gen(font)
        print('处理', name, flush=True)
        results.append(process_dataset(ds, gen))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.exit(main())
