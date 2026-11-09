"""ocr_seed_stain 转正验收 · 独立集 v1 运行器（2026-11-05）。

按 转正验收门槛_独立集v1_20261105.json 执行（看结果前预登记，本轮不改参数）：
- 66 张全部跑 候选关闭/开启 两遍（桌面默认参数，dpi=200）；
- 61 张非 timestamp：开 vs 关 逐字节一致（零影响）；
- 5 张 timestamp：召回不回退、选区外零改动、逐通道误差不劣化；
- 负例中含日期样式正文的样本单列核对。

推理只读输入图；真值在推理完成后加载评分。证据写
独立集v1验收证据_20261105.json；每张图（原图/输出/选区）存 独立集v1产物/。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

DATA = ROOT / 'data/watermark_independent_v1'
OUT = ROOT / 'out/日期时间戳OCR种子候选_20261105'
PROD = OUT / '独立集v1产物'
EVIDENCE = OUT / '独立集v1验收证据_20261105.json'
DPI = 200


def load_rows():
    rows = []
    for line in (DATA / 'manifest.jsonl').read_text().splitlines():
        r = json.loads(line)
        r['_input'] = str(DATA / r['paths']['watermarked'])
        r['_clean'] = str(DATA / r['paths']['clean'])
        r['_alpha'] = str(DATA / r['paths']['alpha'])
        rows.append(r)
    return rows


def run_all(rows, experimental: bool, tag: str):
    from PySide6.QtCore import Qt
    from wm import documents
    from wm.batches import Batch, BatchWorker
    from wm.contract import Thresholds

    pages, lookup = [], []
    for r in rows:
        pages.extend(documents.load_pages(r['_input'], dpi=DPI))
        lookup.append(r)
    batch = Batch(f'独立集v1_{tag}', pages)
    results = {}
    t0 = time.time()

    def event(payload):
        if payload['kind'] != 'page':
            return
        r = lookup[payload['index']]
        results[r['id']] = payload['result']

    thr = Thresholds(experimental_ocr_line_seed=experimental)
    worker = BatchWorker([(batch, list(range(len(pages))), thr, 11, True, None)])
    worker.update.connect(event, Qt.DirectConnection)
    worker.run()
    print(f'[{tag}] {len(results)}/{len(rows)} 张，用时 {time.time() - t0:.0f}s', flush=True)
    return results


def main():
    PROD.mkdir(exist_ok=True)
    rows = load_rows()
    ts_ids = {r['id'] for r in rows if r['subtype'] == 'timestamp'}
    print(f'独立集 {len(rows)} 张：timestamp {len(ts_ids)}、负例 '
          f'{sum(1 for r in rows if r["is_negative"])}', flush=True)

    off = run_all(rows, False, '候选关闭')
    on = run_all(rows, True, '候选开启')

    report = {'零影响': {}, 'timestamp': {}, 'integrity': {}}
    for r in rows:
        rid = r['id']
        a, b = on.get(rid), off.get(rid)
        if a is None or b is None:
            report['零影响'][rid] = None
            continue
        same = bool(np.array_equal(a.output.image, b.output.image)
                    and np.array_equal(a.mask.alpha, b.mask.alpha))
        row = {'is_negative': bool(r['is_negative']), 'subtype': r['subtype'],
               'byte_identical': same,
               'on_method': a.mask.method,
               'on_selected_px': int((a.mask.alpha > 0).sum())}
        report['零影响'][rid] = row
        if rid in ts_ids:
            gt = cv2.imread(r['_alpha'], 0) > 0
            clean = cv2.imread(r['_clean'])
            out_a, out_b = a.output.image, b.output.image
            img = a.source.image
            sel_a = a.mask.binary(0.08) > 0
            sel_b = b.mask.binary(0.08) > 0
            rec_a, rec_b = float(sel_a[gt].mean()), float(sel_b[gt].mean())
            err_a = float(np.abs(out_a.astype(float) - clean.astype(float))[gt].mean())
            err_b = float(np.abs(out_b.astype(float) - clean.astype(float))[gt].mean())
            outside = int(np.abs(out_a.astype(int) - img.astype(int))[~a.mask.footprint].max()) \
                if a.mask.footprint is not None else None
            report['timestamp'][rid] = {
                'on_triggered_ocr_seed': a.mask.method == 'ocr_seed_stain',
                'recall_on': rec_a, 'recall_off': rec_b,
                'err_on_color': err_a, 'err_off_color': err_b,
                'outside_max_change_on': outside,
                'on_selected_px': int(sel_a.sum()), 'off_selected_px': int(sel_b.sum()),
                'pass_no_regress': bool(rec_a >= rec_b - 0.01 and err_a <= err_b + 0.5
                                        and outside == 0),
            }
            cv2.imwrite(str(PROD / f'{rid}__on.png'), out_a)
            overlay = img.copy()
            overlay[sel_a] = (overlay[sel_a].astype(float) * .55 + np.array([255, 0, 0]) * .45).astype('uint8')
            cv2.imwrite(str(PROD / f'{rid}__sel.jpg'), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

    ok_rows = [v for v in report['零影响'].values() if v]
    n_ok = sum(v['byte_identical'] for v in ok_rows)
    print('零影响逐字节一致：', n_ok, '/', len(ok_rows), flush=True)
    print('其中负例：', sum(v['byte_identical'] for v in ok_rows if v['is_negative']), '/',
          sum(1 for v in ok_rows if v['is_negative']), flush=True)
    for rid, row in report['timestamp'].items():
        print(f"{rid}: on触发={row['on_triggered_ocr_seed']} "
              f"召回 off {row['recall_off']:.3f} → on {row['recall_on']:.3f} | "
              f"误差 off {row['err_off_color']:.2f} → on {row['err_on_color']:.2f} | "
              f"区域外={row['outside_max_change_on']} | {'通过' if row['pass_no_regress'] else '未过'}",
              flush=True)

    report['meta'] = {'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                      'dataset': str(DATA), 'dpi': DPI,
                      'candidate': 'ocr_seed_stain', 'flag': 'experimental_ocr_line_seed',
                      'note': '门槛见 转正验收门槛_独立集v1_20261105.json；本轮未改候选参数'}
    EVIDENCE.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('证据已写入', EVIDENCE, flush=True)


if __name__ == '__main__':
    main()
