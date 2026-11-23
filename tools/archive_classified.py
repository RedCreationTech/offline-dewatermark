#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""效果定档归档：把人工定档结果复制进三个数据集目录。

输入：明细报告里「导出任档结果」生成的 JSON（含 run_dir、record_sha256、每样本 user_class）。
输出：data/效果分类数据集/{效果好,提示词可处理,待解决}/<来源组>/<样本id>_p<页>/，
每个样本目录含输入图、处理结果、选区与核对图，以及 判定记录.json（机器指标＋人工定档＋代码版本）。

规则：
- 带水印样本必须全部定档才归档（漏定的直接报错列出）；运行错误页与无水印负例不参与定档。
- 目标目录已存在且定档不同：拒绝，除非 --force（旧判定记入 变更历史，不删除）。
- 只复制不移动：原始批次产物原样保留。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / 'data' / '效果分类数据集'
CLASSES = ('效果好', '提示词可处理', '待解决')
# 样本产物文件名（与封存目录内的记录一致；输入原图从样本清单的 input 路径复制）
ARTIFACTS = ('原图预览.jpg', '处理结果.png', '选区预览.jpg', '漏选多选核对.jpg')


def code_version() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, cwd=ROOT).stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def build_records(run_dir: Path) -> dict[str, dict]:
    run = load_json(run_dir / '运行记录.json')
    return {r['key']: r for r in run['records'] if r.get('key')}


def build_inputs(run_dir: Path) -> dict[str, str]:
    manifest_path = run_dir / '样本清单.json'
    if not manifest_path.is_file():
        return {}
    manifest = load_json(manifest_path)
    return {c['id']: c['input'] for c in manifest.get('cases', []) if c.get('id') and c.get('input')}


def archive(payload: dict, dataset_root: Path, force: bool) -> int:
    run_dir = Path(payload['run_dir']).resolve()
    if not (run_dir / '运行记录.json').is_file():
        raise SystemExit(f'运行目录缺少 运行记录.json：{run_dir}')
    records = build_records(run_dir)
    inputs = build_inputs(run_dir)
    stamp = datetime.now().astimezone().isoformat(timespec='seconds')
    version = code_version()

    samples = payload.get('samples') or []
    need = [s for s in samples if not s.get('negative') and s.get('status') != 'error']
    missing = [f"{s.get('id')} 第{s.get('page')}页" for s in need if not s.get('user_class')]
    if missing:
        raise SystemExit(f'有 {len(missing)} 张带水印样本未定档，先在报告里定档再导出：' + '、'.join(missing[:5])
                         + (' 等' if len(missing) > 5 else ''))

    plan = []
    for s in samples:
        cls = s.get('user_class')
        if not cls:
            continue  # 负例/错误页未定档，跳过
        if cls not in CLASSES:
            raise SystemExit(f'未知定档“{cls}”（应为：{"／".join(CLASSES)}）：{s.get("key")}')
        rec = records.get(s['key'])
        if rec is None:
            raise SystemExit(f'运行记录里找不到样本 {s.get("key")}')
        plan.append((s, rec))

    done = {c: 0 for c in CLASSES}
    changed = []
    for s, rec in plan:
        cls, key = s['user_class'], s['key']
        sid = str(s.get('id') or key).replace('/', '_')
        group = str(s.get('group') or '未分组')
        name = f'{sid}_p{s.get("page", 1)}'
        dest = dataset_root / cls / group / name
        dest.mkdir(parents=True, exist_ok=True)
        files = rec.get('files') or {}
        copied = []
        src_input = inputs.get(rec.get('id'))
        if src_input and Path(src_input).is_file():
            shutil.copy2(src_input, dest / ('输入' + Path(src_input).suffix))
            copied.append('输入')
        for art in ARTIFACTS:
            p = files.get(art)
            if p and Path(p).is_file():
                shutil.copy2(p, dest / art)
                copied.append(art)
        verdict = dict(
            schema_version=1,
            sample_key=key, id=rec.get('id'), page=rec.get('page'),
            group=s.get('group'), style=s.get('style'), label=s.get('label'),
            user_class=cls, machine_bucket=s.get('machine_bucket'),
            machine_candidate=s.get('machine_candidate'),
            preclass_thresholds=payload.get('preclass_thresholds'),
            metrics={k: s.get(k) for k in (
                'recall', 'precision', 'strong_recall', 'missed_pixels',
                'false_positive_pixels', 'fp_frac', 'watermark_mae',
                'background_mae', 'elapsed_ms', 'method', 'repair_mode')},
            classified_at=stamp, code_version=version,
            run_dir=str(run_dir), record_sha256=payload.get('record_sha256'),
            report_title=payload.get('report_title'),
        )
        record_path = dest / '判定记录.json'
        history = []
        # 改档检测：同一输入＋页在任一分类目录里已有判定都算，不能只看出目录。
        olds = [dataset_root / c / group / name / '判定记录.json' for c in CLASSES]
        olds = [p for p in olds if p.is_file()]
        old_dirs = [p.parent for p in olds]
        for old_path in olds:
            old = load_json(old_path)
            if old.get('user_class') != cls:
                if not force:
                    raise SystemExit(f'已定档为「{old.get("user_class")}」，改档需 --force：{old_path.parent}')
                changed.append(old_path.parent)
            history += old.get('变更历史') or [dict(
                user_class=old.get('user_class'), classified_at=old.get('classified_at'),
                code_version=old.get('code_version'), record_sha256=old.get('record_sha256'))]
        if history:
            verdict['变更历史'] = history
        record_path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding='utf-8')
        # 改档生效后移除旧目录，数据集内同一输入只保留一个归属。
        for d in old_dirs:
            if d != dest and d.is_dir():
                shutil.rmtree(d)
        done[cls] += 1

    print(f'归档完成（{len(plan)} 个样本）：' + '，'.join(f'{c} {n}' for c, n in done.items() if n))
    if changed:
        print(f'改档（--force，旧判定已记入变更历史）：{len(changed)} 个')
    print(f'数据集目录：{dataset_root}')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json', type=Path, required=True, help='报告里导出的效果定档 JSON')
    ap.add_argument('--dataset-root', type=Path, default=DATASET_ROOT, help='三个分类数据集的上级目录')
    ap.add_argument('--force', action='store_true', help='允许改档（旧判定记入变更历史）')
    args = ap.parse_args()
    payload = load_json(args.json)
    if payload.get('schema_version') != 1 or not payload.get('run_dir'):
        raise SystemExit('不是有效的定档导出文件（缺 schema_version/run_dir）')
    return archive(payload, args.dataset_root.resolve(), args.force)


if __name__ == '__main__':
    sys.exit(main())
