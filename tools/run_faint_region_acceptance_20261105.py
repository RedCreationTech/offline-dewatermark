"""极淡区域反染色候选 · 14 张全量验收证据运行器（2026-11-05）。

按 20261104 预登记门槛（验收门槛_20261104.json）补齐验证广度：
1. 极淡批 14 张（基础测试 9 + 人工增量 5）：候选开启（实验开关，默认样式路由）
   逐张评分（召回 alpha>0、精确率、水印区误差前后、区域内连带、区域外最大改动）；
2. 基线逐字节对照：候选关闭重跑，输出/选区 sha256 必须与 20261104_002213
   冻结基线一致（分组与设置按冻结任务原样重现，含跨页先验）；
3. 负例安全：全部无水印负例候选开启 vs 关闭逐字节一致（零选区零改动）；
4. 相邻样式零影响：居中大字/白色半透明大字/局部文字各 2 张（不同底图），
   候选开启 vs 关闭逐字节一致。

口径说明：预登记文件写"14 张 × 三模式"，但 2026-11-03 决策（
docs/decisions/2026-11-03-单模式基线与面板收敛.md 与
config/regression_control.json required_modes=["desktop"]）已把回归收敛为
桌面默认单模式；本轮按现行桌面默认口径执行并在结论中披露该口径变化。

推理只读取输入图；参考与标注在本页推理完成后才加载评分（沿用封存件，
评分与源图推理分离）。候选仍为隔离实验通道，不接入默认路径。
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
OUT = ROOT / 'out/极淡区域反染色验收_20261105'
SAMPLES = OUT / '样本产物'
EVIDENCE = OUT / '验收证据_20261105.json'

ADJACENT_STYLES = ('center_text', 'center_white', 'local_text')


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_manifest():
    return json.loads((FREEZE / '样本清单.json').read_text())


def select_cases(manifest):
    faint, negatives, adjacent = [], [], {s: [] for s in ADJACENT_STYLES}
    seen_base = {s: set() for s in ADJACENT_STYLES}
    for c in manifest['cases']:
        batch = c['batch']
        if batch.endswith('05_极淡文字'):
            faint.append(c)
        elif c.get('negative'):
            negatives.append(c)
        elif c['style'] in ADJACENT_STYLES:
            pool = adjacent[c['style']]
            if len(pool) < 2 and c.get('base_id') not in seen_base[c['style']]:
                pool.append(c)
                seen_base[c['style']].add(c.get('base_id'))
    return faint, negatives, adjacent


def freeze_tasks():
    """冻结逐批任务：task_dir -> task.json 内容；并建立 case id -> 记录索引。"""
    tasks, records = {}, {}

    def _scan(inference_path, task):
        try:
            data = json.loads(Path(inference_path).read_text())
        except Exception:
            return
        for rec in data:
            if rec.get('status') != 'completed':
                continue
            records[rec['id']] = rec

    for task_path in sorted((FREEZE / '逐批运行').glob('*/task.json')):
        task = json.loads(task_path.read_text())
        tasks[task_path.parent] = task
        _scan(task_path.parent / 'inference.json', task)
    return tasks, records


def run_config(tasks, case_ids, experimental: bool, tag: str):
    """按冻结任务原分组重跑选定案例。返回 {case_id: PageResult}。"""
    from PySide6.QtCore import Qt
    from wm import documents
    from wm.batches import Batch, BatchWorker
    from wm.contract import Thresholds

    results = {}
    t0 = time.time()
    for task_dir, task in tasks.items():
        inputs = [i for i in task['inputs'] if i['id'] in case_ids]
        if not inputs:
            continue
        settings = task['settings']
        thr = Thresholds(**settings['thresholds'], experimental_faint_region=experimental)
        wanted = {i['id'] for i in inputs}
        pages, lookup = [], []
        for item in task['inputs']:
            loaded = documents.load_pages(item['input'], dpi=item['dpi'])
            pages.extend(loaded)
            lookup.extend((item, n) for n in range(1, len(loaded) + 1))
        indices = [n for n, (item, _) in enumerate(lookup) if item['id'] in wanted]
        batch = Batch(f'验收_{tag}', pages)

        def event(payload, lookup=lookup):
            if payload['kind'] != 'page':
                return
            item, _page_no = lookup[payload['index']]
            if payload['result'] is not None:
                results[item['id']] = payload['result']

        worker = BatchWorker([(batch, indices, thr,
                               settings['close_px'], settings['ocr_enabled'], None)])
        worker.update.connect(event, Qt.DirectConnection)
        worker.run()
    print(f'[{tag}] {len(results)}/{len(case_ids)} 页完成，用时 {time.time() - t0:.0f}s',
          flush=True)
    return results


def digest(result, mask_alpha_min):
    out = np.ascontiguousarray(result.output.image).tobytes()
    sel = np.ascontiguousarray((result.mask.binary(mask_alpha_min) > 0)).tobytes()
    return (hashlib.sha256(out).hexdigest(), hashlib.sha256(sel).hexdigest(),
            int((result.mask.alpha > 0).sum()), result.mask.method,
            result.mask.repair_mode, int(result.verdict.elapsed_ms))


def score_against_reference(case, result):
    from wm.contract import Thresholds
    ref = case['references'][0]
    clean = cv2.imread(ref['clean'])
    alpha = cv2.imread(ref['alpha'], 0)
    thr = Thresholds()
    gt = alpha > 0
    img = result.source.image
    out = result.output.image
    mask = result.mask
    sel = mask.alpha > 0
    before = float(np.abs(img.mean(2) - clean.mean(2))[gt].mean())
    after = float(np.abs(out.mean(2) - clean.mean(2))[gt].mean())
    triggered = mask.method == 'faint_region_uniform_tint'
    return {
        'method': mask.method,
        'repair_mode': mask.repair_mode,
        'triggered_faint_region': triggered,
        'alpha_gt_mean': float(alpha[gt].mean() / 255.0),
        'alpha_hat': float(mask.overlay_opacity.max()) if triggered else None,
        'recall_alpha_pos': float(sel[gt].mean()),
        'precision': float(sel[gt].sum() / max(sel.sum(), 1)),
        'wm_err_before': before,
        'wm_err_after': after,
        'collateral_mean': (float(np.abs(out.mean(2) - clean.mean(2))[mask.footprint & ~gt].mean())
                            if triggered and (mask.footprint & ~gt).any() else None),
        'outside_max_change': int(np.abs(out.astype(int) - img.astype(int))[~mask.footprint].max()),
        'selected_px': int(sel.sum()),
        'selected_fraction': float(sel.mean()),
        'elapsed_ms': int(result.verdict.elapsed_ms),
    }


def main():
    OUT.mkdir(exist_ok=True)
    SAMPLES.mkdir(exist_ok=True)
    manifest = load_manifest()
    faint, negatives, adjacent = select_cases(manifest)
    adj_cases = [c for pool in adjacent.values() for c in pool]
    print(f'选定：极淡 {len(faint)}、负例 {len(negatives)}、相邻 {len(adj_cases)}', flush=True)

    # 输入与参考哈希核对（推理前只做完整性校验，不进入选区逻辑）
    integrity = {}
    for group in (faint, negatives, adj_cases):
        for c in group:
            row = {'input_ok': sha256_file(c['input']) == c['input_sha256']}
            for ref in c.get('references', []):
                row.setdefault('refs_ok', True)
                row['refs_ok'] = (row['refs_ok']
                                  and sha256_file(ref['clean']) == ref['clean_sha256']
                                  and sha256_file(ref['alpha']) == ref['alpha_sha256'])
            integrity[c['id']] = row

    tasks, freeze_records = freeze_tasks()
    selected = {c['id'] for group in (faint, negatives, adj_cases) for c in group}
    missing = selected - set(freeze_records)
    if missing:
        print('警告：以下案例缺少冻结记录，无法做逐字节对照：', sorted(missing), flush=True)

    off = run_config(tasks, selected, experimental=False, tag='候选关闭')
    on = run_config(tasks, selected, experimental=True, tag='候选开启')

    # 2. 基线逐字节对照（候选关闭 vs 冻结）
    baseline_match = {}
    for cid in sorted(selected):
        rec = freeze_records.get(cid)
        res = off.get(cid)
        if rec is None or res is None:
            baseline_match[cid] = None
            continue
        out_sha, sel_sha, _n, _m, _r, _e = digest(res, rec['settings']['thresholds']['mask_alpha_min'])
        baseline_match[cid] = {
            'output_match': out_sha == rec['output_sha256'],
            'mask_match': sel_sha == rec['mask_sha256'],
            'freeze_selected_pixels': rec['selected_pixels'],
        }

    # 3./4. 候选开启 vs 关闭 逐字节（负例 + 相邻样式）
    zero_impact = {}
    for group, label in ((negatives, '负例'), (adj_cases, '相邻样式')):
        for c in group:
            a = on.get(c['id'])
            b = off.get(c['id'])
            if a is None or b is None:
                zero_impact[c['id']] = None
                continue
            same_out = np.array_equal(a.output.image, b.output.image)
            same_mask = np.array_equal(a.mask.alpha, b.mask.alpha)
            zero_impact[c['id']] = {
                'group': label, 'style': c['style'],
                'byte_identical': bool(same_out and same_mask),
                'on_selected_px': int((a.mask.alpha > 0).sum()),
                'off_selected_px': int((b.mask.alpha > 0).sum()),
                'on_method': a.mask.method,
            }

    # 1. 极淡批 14 张评分（候选开启）
    faint_scores = {}
    for c in faint:
        res = on.get(c['id'])
        if res is None:
            faint_scores[c['id']] = None
            continue
        row = score_against_reference(c, res)
        row['integrity'] = integrity[c['id']]
        row['baseline'] = baseline_match.get(c['id'])
        faint_scores[c['id']] = row
        # 保存产物图（全分辨率留在磁盘供核查）
        stem = c['id'].replace('/', '__')
        cv2.imwrite(str(SAMPLES / f'{stem}__on_output.png'), res.output.image)
        overlay = res.source.image.copy()
        sel = res.mask.binary(0.08) > 0
        overlay[sel] = (overlay[sel].astype(float) * .55 + np.array([255, 0, 0]) * .45).astype('uint8')
        cv2.imwrite(str(SAMPLES / f'{stem}__on_selection.jpg'), overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    git_head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              cwd=ROOT).stdout.strip()
    git_dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'wm/', 'tools/'],
                               capture_output=True, text=True, cwd=ROOT).stdout.strip()

    evidence = {
        'schema_version': 1,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'candidate': 'faint_region_uniform_tint（极淡区域一致染色反变换）',
        'trigger': 'Thresholds.experimental_faint_region=True（实验开关；样式提示"极淡"等价），'
                   '默认路径不触发',
        'freeze_baseline': str(FREEZE),
        'code_head': git_head,
        'code_dirty_wm_tools': git_dirty or None,
        'mode': 'desktop（桌面默认单口径；预登记"三模式"按 2026-11-03 单模式决策收敛，见口径说明）',
        'selection': {
            '极淡批': [c['id'] for c in faint],
            '负例': [c['id'] for c in negatives],
            '相邻样式': {s: [c['id'] for c in adjacent[s]] for s in ADJACENT_STYLES},
        },
        'integrity': integrity,
        'baseline_match_候选关闭_vs_冻结': baseline_match,
        'zero_impact_候选开启_vs_关闭': zero_impact,
        'faint_scores_候选开启': faint_scores,
    }
    EVIDENCE.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    print(f'证据已写入 {EVIDENCE}', flush=True)

    # 控制台摘要
    ok_base = [v for v in baseline_match.values() if v]
    print('基线逐字节对照：', sum(v['output_match'] and v['mask_match'] for v in ok_base),
          '/', len(ok_base), '一致')
    zi = [v for v in zero_impact.values() if v]
    print('负例+相邻零影响：', sum(v['byte_identical'] for v in zi), '/', len(zi), '逐字节一致')
    for cid, row in faint_scores.items():
        if row is None:
            print(cid, '未运行')
            continue
        print(f"{cid}: triggered={row['triggered_faint_region']} "
              f"recall={row['recall_alpha_pos']:.3f} prec={row['precision']:.3f} "
              f"err {row['wm_err_before']:.2f}→{row['wm_err_after']:.2f} "
              f"outside={row['outside_max_change']}")


if __name__ == '__main__':
    main()
