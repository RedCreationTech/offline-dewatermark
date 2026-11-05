"""OCR 种子通道（ocr_seed_stain）验收证据运行器（2026-11-05）。

按 out/日期时间戳OCR种子候选_20261105/验收门槛_20261105.json 执行：
1. 目标 5 张（document_029–032、charts_015 的 timestamp 变体）：候选开启
   逐张评分（召回/精确率/水印区误差前后/区域外改动/耗时）；
2. 其余 9 张 timestamp：候选开启 vs 关闭逐字节一致；
3. 负例全部：候选开启 vs 关闭逐字节一致；
4. 相邻样式（center_text/center_white/local_text 各 2）：逐字节一致；
5. 基线对照：候选关闭重跑 vs 20261104 冻结基线逐字节一致（分组/设置按
   冻结任务原样重现）。

推理只读输入图；真值在推理完成后加载评分。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import cv2

FREEZE = ROOT / 'out/项目防退化控制/20261104_002213_219680_freeze'
OUT = ROOT / 'out/日期时间戳OCR种子候选_20261105'
SAMPLES = OUT / '样本产物'
EVIDENCE = OUT / '验收证据_20261105.json'

TARGETS = ('document_029__timestamp.png', 'document_030__timestamp.png',
           'document_031__timestamp.png', 'document_032__timestamp.png',
           'charts_015__timestamp.png')
ADJACENT_STYLES = ('center_text', 'center_white', 'local_text')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def select_cases(manifest):
    ts, negatives, adjacent = [], [], {s: [] for s in ADJACENT_STYLES}
    seen = {s: set() for s in ADJACENT_STYLES}
    for c in manifest['cases']:
        if c['style'] == 'timestamp':
            ts.append(c)
        elif c.get('negative'):
            negatives.append(c)
        elif c['style'] in ADJACENT_STYLES:
            pool = adjacent[c['style']]
            if len(pool) < 2 and c.get('base_id') not in seen[c['style']]:
                pool.append(c)
                seen[c['style']].add(c.get('base_id'))
    return ts, negatives, adjacent


def freeze_tasks_and_records():
    tasks, records = {}, {}
    for task_path in sorted((FREEZE / '逐批运行').glob('*/task.json')):
        task = json.loads(task_path.read_text())
        tasks[task_path.parent] = task
        inf = task_path.parent / 'inference.json'
        if inf.exists():
            for rec in json.loads(inf.read_text()):
                if rec.get('status') == 'completed':
                    records[rec['id']] = rec
    return tasks, records


def run_config(tasks, case_ids, experimental: bool, tag: str):
    from PySide6.QtCore import Qt
    from wm import documents
    from wm.batches import Batch, BatchWorker
    from wm.contract import Thresholds

    results = {}
    t0 = time.time()
    for task_dir, task in tasks.items():
        wanted_ids = {i['id'] for i in task['inputs'] if i['id'] in case_ids}
        if not wanted_ids:
            continue
        settings = task['settings']
        thr = Thresholds(**settings['thresholds'], experimental_ocr_line_seed=experimental)
        pages, lookup = [], []
        for item in task['inputs']:
            loaded = documents.load_pages(item['input'], dpi=item['dpi'])
            pages.extend(loaded)
            lookup.extend((item, n) for n in range(1, len(loaded) + 1))
        indices = [n for n, (item, _) in enumerate(lookup) if item['id'] in wanted_ids]
        batch = Batch(f'验收_{tag}', pages)

        def event(payload, lookup=lookup):
            if payload['kind'] != 'page':
                return
            item, _ = lookup[payload['index']]
            if payload['result'] is not None:
                results[item['id']] = payload['result']

        worker = BatchWorker([(batch, indices, thr,
                               settings['close_px'], settings['ocr_enabled'], None)])
        worker.update.connect(event, Qt.DirectConnection)
        worker.run()
    print(f'[{tag}] {len(results)}/{len(case_ids)} 页，'
          f'用时 {time.time() - t0:.0f}s', flush=True)
    return results


def digest(result, mask_alpha_min):
    out = hashlib.sha256(np.ascontiguousarray(result.output.image).tobytes()).hexdigest()
    sel = hashlib.sha256(np.ascontiguousarray(
        (result.mask.binary(mask_alpha_min) > 0)).tobytes()).hexdigest()
    return out, sel


def score(case, result):
    ref = case['references'][0]
    clean = cv2.imread(ref['clean'])
    alpha = cv2.imread(ref['alpha'], 0)
    gt = alpha > 0
    img, out, mask = result.source.image, result.output.image, result.mask
    sel = mask.binary(0.08) > 0
    return {
        'method': mask.method,
        'triggered': mask.method == 'ocr_seed_stain',
        'recall_alpha_pos': float(sel[gt].mean()),
        'precision': float(sel[gt].sum() / max(sel.sum(), 1)),
        'wm_err_before': float(np.abs(img.mean(2) - clean.mean(2))[gt].mean()),
        'wm_err_after': float(np.abs(out.mean(2) - clean.mean(2))[gt].mean()),
        'outside_max_change': int(np.abs(out.astype(int) - img.astype(int))[~mask.footprint].max())
        if mask.footprint is not None else None,
        'selected_px': int(sel.sum()),
        'elapsed_ms': int(result.verdict.elapsed_ms),
    }


def main():
    SAMPLES.mkdir(exist_ok=True)
    manifest = json.loads((FREEZE / '样本清单.json').read_text())
    ts, negatives, adjacent = select_cases(manifest)
    adj_cases = [c for pool in adjacent.values() for c in pool]
    targets = [c for c in ts if c['id'].split('/')[-1] in TARGETS]
    rest_ts = [c for c in ts if c not in targets]
    print(f'timestamp {len(ts)}（目标 {len(targets)}）、负例 {len(negatives)}、'
          f'相邻 {len(adj_cases)}', flush=True)

    integrity = {}
    for group in (ts, negatives, adj_cases):
        for c in group:
            row = {'input_ok': sha256_file(c['input']) == c['input_sha256']}
            for ref in c.get('references', []):
                row.setdefault('refs_ok', True)
                row['refs_ok'] = (row['refs_ok']
                                  and sha256_file(ref['clean']) == ref['clean_sha256']
                                  and sha256_file(ref['alpha']) == ref['alpha_sha256'])
            integrity[c['id']] = row

    tasks, freeze_records = freeze_tasks_and_records()
    selected = {c['id'] for group in (ts, negatives, adj_cases) for c in group}

    off = run_config(tasks, selected, False, '候选关闭')
    on = run_config(tasks, selected, True, '候选开启')

    baseline_match = {}
    for cid in sorted(selected):
        rec, res = freeze_records.get(cid), off.get(cid)
        if rec is None or res is None:
            baseline_match[cid] = None
            continue
        out_sha, sel_sha = digest(res, rec['settings']['thresholds']['mask_alpha_min'])
        baseline_match[cid] = {'output_match': out_sha == rec['output_sha256'],
                               'mask_match': sel_sha == rec['mask_sha256']}

    zero_impact = {}
    for group, label in ((rest_ts, '其余时间戳'), (negatives, '负例'),
                         (adj_cases, '相邻样式')):
        for c in group:
            a, b = on.get(c['id']), off.get(c['id'])
            if a is None or b is None:
                zero_impact[c['id']] = None
                continue
            zero_impact[c['id']] = {
                'group': label, 'style': c['style'],
                'byte_identical': bool(np.array_equal(a.output.image, b.output.image)
                                       and np.array_equal(a.mask.alpha, b.mask.alpha)),
                'on_selected_px': int((a.mask.alpha > 0).sum()),
                'on_method': a.mask.method,
            }

    target_scores = {}
    for c in targets:
        res = on.get(c['id'])
        if res is None:
            target_scores[c['id']] = None
            continue
        row = score(c, res)
        row['baseline'] = baseline_match.get(c['id'])
        target_scores[c['id']] = row
        stem = c['id'].replace('/', '__')
        cv2.imwrite(str(SAMPLES / f'{stem}__on_output.png'), res.output.image)
        overlay = res.source.image.copy()
        sel = res.mask.binary(0.08) > 0
        overlay[sel] = (overlay[sel].astype(float) * .55 + np.array([255, 0, 0]) * .45).astype('uint8')
        cv2.imwrite(str(SAMPLES / f'{stem}__on_selection.jpg'), overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    git_head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, cwd=ROOT).stdout.strip()
    evidence = {
        'schema_version': 1,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'candidate': 'ocr_seed_stain（OCR 单行小字种子→局部染色实测）',
        'trigger': 'Thresholds.experimental_ocr_line_seed=True（实验开关，默认关闭）',
        'freeze_baseline': str(FREEZE),
        'code_head': git_head,
        'mode': 'desktop（桌面默认单口径）',
        'selection': {'目标': [c['id'] for c in targets],
                      '其余时间戳': [c['id'] for c in rest_ts],
                      '负例': [c['id'] for c in negatives],
                      '相邻样式': {s: [c['id'] for c in adjacent[s]] for s in ADJACENT_STYLES}},
        'integrity': integrity,
        'baseline_match_候选关闭_vs_冻结': baseline_match,
        'zero_impact_候选开启_vs_关闭': zero_impact,
        'target_scores_候选开启': target_scores,
    }
    EVIDENCE.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))

    ok_base = [v for v in baseline_match.values() if v]
    print('基线逐字节对照：', sum(v['output_match'] and v['mask_match'] for v in ok_base),
          '/', len(ok_base))
    zi = [v for v in zero_impact.values() if v]
    print('零影响：', sum(v['byte_identical'] for v in zi), '/', len(zi))
    for cid, row in target_scores.items():
        if row is None:
            print(cid, '未运行')
            continue
        print(f"{cid.split('/')[-1]}: triggered={row['triggered']} "
              f"recall={row['recall_alpha_pos']:.3f} prec={row['precision']:.3f} "
              f"err {row['wm_err_before']:.2f}→{row['wm_err_after']:.2f} "
              f"outside={row['outside_max_change']}")


if __name__ == '__main__':
    main()
