#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""常见水印专项基线（2026-11-04）。

范围：data/应用测试输入 中 11 种常见样式的 145 张正例 + 23 张无水印负例 = 168 个输入。
口径：桌面默认单模式（文字识别与笔画补全常开），应用实际 pipeline.process_file；
真值在推理结束后才加载（alpha/干净底图），与固定最小回归同评分入口（verdict.build_verdict）。
产出：out/常见水印基线_20261104/  per-page 图 + summary.json + report.html。
这是专项观察基线，不替换官方全量基线指针。
"""
from __future__ import annotations

import collections
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys_in = ROOT / 'data' / '应用测试输入'
OUT = ROOT / 'out' / (sys.argv[1] if len(sys.argv) > 1 else '常见水印基线_20261104')

COMMON = {'timestamp', 'website', 'logo_transparent', 'logo_opaque', 'local_text',
          'tiny_corner', 'signature', 'center_text', 'center_white', 'center_gray',
          'tiled_diagonal'}
NEGATIVE = {'negative', 'normal_logo_signature_negative'}


def build_case_list():
    # 数据集清单索引：样本 id -> (数据集根, 记录)
    index = {}
    for name in ('watermark_training_v1', 'watermark_training_increment_v1_1'):
        ds = ROOT / 'data' / name
        for line in (ds / 'manifest.jsonl').read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('kind') == 'image' and not r.get('is_negative'):
                index[r['id']] = (ds, r)
    cases = []
    for p in sorted(sys_in.rglob('*.png')):
        sid = p.stem
        style = sid.split('__')[-1]
        if style in COMMON:
            if sid not in index:
                print('跳过（无真值登记）', p)
                continue
            cases.append(dict(input=p, id=sid, style=style, negative=False, truth=index[sid]))
        elif style in NEGATIVE or sid.endswith('__negative'):
            cases.append(dict(input=p, id=sid, style='negative', negative=True, truth=None))
    return cases


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from wm import pipeline, verdict
    from wm.contract import Thresholds

    cases = build_case_list()
    print(f'共 {len(cases)} 个输入：正例 {sum(not c["negative"] for c in cases)}，负例 {sum(c["negative"] for c in cases)}', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone().isoformat()
    results = []
    # 口径：桌面默认单模式（文字识别与笔画补全常开）。桌面封存基线的
    # desktop 阈值档 = Thresholds() 仅 watermark_recall_priority=True
    # （见 20261104_002213 全量封存 profiles.desktop）；首轮误用默认 False，
    # 导致笔画补全整体未启用、同输入较上次基线假性退化，本轮修正。
    thr = Thresholds(watermark_recall_priority=True)

    def save_reports():
        pos = [p for r in results if not r['negative'] for p in r['pages']]
        neg = [p for r in results if r['negative'] for p in r['pages']]
        by_style = collections.defaultdict(list)
        for r in results:
            if not r['negative']:
                for p in r['pages']:
                    by_style[r['style']].append(p)
        summary = dict(
            created_at=started, completed_at=datetime.now().astimezone().isoformat(),
            scope='常见水印专项基线（168 输入=145 正例+23 负例），桌面默认单模式；不替换官方全量基线指针',
            note='今天 112 张样本刚经可见度修正，本基线数值与 20261104 全量基线比较时需注意样本已变更；'
                 '真值在推理结束后加载，应用只收到输入图。',
            total_inputs=len(cases), completed_inputs=len(results),
            positive_pages=len(pos), negative_pages=len(neg),
            negative_safe=sum(bool(p.get('passed')) for p in neg),
            error_pages=sum(p['status'] == 'error' for r in results for p in r['pages']),
            by_style={s: dict(
                pages=len(ps),
                recall=round(float(np.nanmean([p['metrics'].get('mask_recall', np.nan) for p in ps if p.get('metrics')])), 4) if any(p.get('metrics') for p in ps) else None,
                precision=round(float(np.nanmean([p['metrics'].get('mask_precision', np.nan) for p in ps if p.get('metrics')])), 4) if any(p.get('metrics') for p in ps) else None,
            ) for s, ps in sorted(by_style.items())},
            results=results,
        )
        (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    for n, case in enumerate(cases, 1):
        rec = dict(case_id=case['id'], style=case['style'], negative=case['negative'], pages=[])
        try:
            processed = pipeline.process_file(case['input'], thr=thr, dpi=96)
            assert len(processed) == 1
            result = processed[0]
            folder = OUT / case['id']
            folder.mkdir(exist_ok=True)

            def png(name, array):
                dest = folder / name
                Image.fromarray(np.asarray(array, dtype=np.uint8)).save(dest)
                return dest.relative_to(OUT).as_posix()

            changed = float(np.any(result.source.image != result.output.image, axis=2).mean())
            item = dict(page=1, status='scored', passed=None,
                        input_preview=png('input.png', result.source.image),
                        output=png('output.png', result.output.image),
                        predicted_mask=png('mask.png', result.mask.binary()),
                        mask_method=result.mask.method, changed_fraction=changed,
                        elapsed_ms=result.verdict.elapsed_ms)
            if case['negative']:
                item['passed'] = bool(changed == 0 and result.mask.coverage() == 0)
            else:
                ds, truth_rec = case['truth']
                with Image.open(ds / truth_rec['paths']['alpha']) as a:
                    alpha = np.asarray(a.convert('L').resize(
                        (result.source.width, result.source.height),
                        Image.Resampling.LANCZOS), np.float32) / 255
                clean = np.asarray(Image.open(ds / truth_rec['paths']['clean']).convert('RGB'))
                if clean.shape[:2] != result.source.image.shape[:2]:
                    clean = np.asarray(Image.fromarray(clean).resize(
                        (result.source.width, result.source.height), Image.Resampling.LANCZOS))
                item['clean_preview'] = png('clean.png', clean)
                score = verdict.build_verdict(
                    source=result.source.image, output=result.output.image, mask=result.mask,
                    engine=result.verdict.engine, repairs=result.verdict.repairs,
                    elapsed_ms=result.verdict.elapsed_ms, stage_ms=result.verdict.stage_ms,
                    thr=thr, clean_reference=clean, alpha_reference=alpha)
                item['passed'] = bool(score.passed)
                item['metrics'] = {k: (float(v) if np.isfinite(v) else None) for k, v in score.metrics.items()}
                item['failures'] = [str(f) for f in score.failures]
            rec['pages'].append(item)
        except Exception as exc:
            rec['pages'].append(dict(page=1, status='error', passed=False, error=f'{type(exc).__name__}: {exc}'))
        results.append(rec)
        save_reports()
        print(f'[{n}/{len(cases)}] {case["id"]}: ' + ', '.join(p['status'] for p in rec['pages']), flush=True)

    # 简版报告页（图片相对路径，本地查看；对外发送需另行做自包含摘要）
    cards = []
    for r in results:
        for p in r['pages']:
            m = p.get('metrics') or {}
            cards.append(
                f'<article><h2>{html.escape(r["case_id"])}</h2>'
                f'<p>{p["status"]} / passed={p.get("passed")} / 通道 {html.escape(str(p.get("mask_method")))} / '
                f'召回 {m.get("recall")} / 精确率 {m.get("precision")} / 改动 {p.get("changed_fraction", 0):.4f} / '
                f'{p.get("elapsed_ms", 0)/1000:.1f}s</p>'
                f'<div class="pair">'
                + ''.join(f'<figure><img loading="lazy" src="{html.escape(p[k])}"><figcaption>{t}</figcaption></figure>'
                          for k, t in [('input_preview', '输入'), ('predicted_mask', '选区'), ('output', '输出'), ('clean_preview', '干净参考')]
                          if p.get(k))
                + f'</div><p>{html.escape(p.get("error", ""))}</p></article>')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>常见水印专项基线 '
            f'{started[5:10]} {started[11:16]}–{datetime.now().astimezone().strftime("%H:%M")}'
            '</title>'
            '<style>body{font:15px/1.8 system-ui;background:#f1f4f8;color:#25394e;padding:25px;max-width:1500px;margin:auto}'
            'article{background:white;border:1px solid #dae4ef;border-radius:10px;padding:24px;margin:20px 0}'
            '.pair{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0}'
            'img{width:100%;height:300px;object-fit:contain;background:#f3f5f7}</style>'
            '<h1>常见水印专项基线 '
            f'{started[5:10]} {started[11:16]}–{datetime.now().astimezone().strftime("%H:%M")}'
            '</h1>'
            '<p>168 输入（145 正例＋23 负例），桌面默认单模式。专项观察基线，不替换官方全量基线指针。</p>'
            + ''.join(cards) + '</html>')
    (OUT / 'report.html').write_text(page)
    print(OUT / 'report.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
