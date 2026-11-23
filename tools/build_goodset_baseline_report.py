#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""效果好数据集基线报告的生成与防回退对比。

用法：
  build [--out 目录]     生成自包含基线报告 HTML（图片 base64 内嵌），并把每张参数写入 _基线参数.json
  check --run-dir 轮次目录 | --json 定档导出.json
                         用新一轮控制轮的结果和基线逐张逐参数对比：
                         召回/精确率不许下降；漏选/多选/各类误差/耗时不得上升。任一违反即退出码 1。

基线口径：以人工定档「效果好」的样本为准（规则分数不参与判定，只作记录）。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GOOD = ROOT / 'data' / '效果分类数据集' / '效果好'
BASELINE = GOOD / '_基线参数.json'
PENDING = ROOT / 'data' / '效果分类数据集' / '待解决'

# 参数方向：up = 越高越好（不许下降）；down = 越低越好（不许上升）。
DIRECTIONS = {
    'recall': 'up', 'precision': 'up', 'strong_recall': 'up',
    'missed_pixels': 'down', 'false_positive_pixels': 'down', 'fp_frac': 'down',
    'watermark_mae': 'down', 'background_mae': 'down', 'elapsed_ms': 'down',
}


def _grain(metric: str, v):
    """按指标的最小比较单位取整（一位小数口径）：召回/精确率 0.1%、像素计数
    10 像素、误差 0.1（0-255 尺度）。低于该粒度的波动视为相同，不判回退。"""
    if metric in ('recall', 'precision', 'strong_recall', 'fp_frac'):
        return round(v * 100, 1)
    if metric in ('missed_pixels', 'false_positive_pixels'):
        return int(v / 10 + 0.5)  # 10 像素粒度，半进避免 banker's rounding 边界抖动
    if metric in ('watermark_mae', 'background_mae'):
        return round(v, 1)
    return v
METRIC_LABEL = {
    'recall': '笔画召回率', 'precision': '选区精确率', 'strong_recall': '明显笔画召回',
    'missed_pixels': '漏选像素', 'false_positive_pixels': '多选像素', 'fp_frac': '多选占全图比',
    'watermark_mae': '水印区参考误差', 'background_mae': '非水印区误差', 'elapsed_ms': '耗时(秒)',
}
SHOW = ['recall', 'precision', 'missed_pixels', 'false_positive_pixels',
        'watermark_mae', 'background_mae', 'elapsed_ms']


def load_verdicts(base: Path) -> list[dict]:
    rows = []
    for p in sorted(base.rglob('判定记录.json')):
        rows.append(json.loads(p.read_text(encoding='utf-8')))
    return rows


def fmt(metric: str, v):
    if v is None:
        return '—'
    if metric in ('recall', 'precision', 'strong_recall', 'fp_frac'):
        return f'{v * 100:.2f}%'
    if metric == 'elapsed_ms':
        return f'{v / 1000:.1f}s'
    if metric in ('watermark_mae', 'background_mae'):
        return f'{v:.4f}'
    return f'{v:,}'


def thumb(path: Path, max_px: int = 800, quality: int = 78) -> str | None:
    if not path.is_file():
        return None
    im = Image.open(path)
    if max(im.size) > max_px:
        r = max_px / max(im.size)
        im = im.resize((int(im.width * r), int(im.height * r)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.convert('RGB').save(buf, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def build(out_dir: Path | None) -> Path:
    rows = load_verdicts(GOOD)
    if not rows:
        raise SystemExit(f'效果好数据集为空：{GOOD}')
    now = datetime.now().astimezone()
    baseline = dict(
        schema_version=1, created_at=now.isoformat(timespec='seconds'),
        standard='防回退：任何一张的任何参数只允许变好——召回/精确率不许下降；漏选/多选/误差/耗时不得上升。',
        directions=DIRECTIONS, metric_label=METRIC_LABEL,
        source_run=rows[0].get('run_dir'), record_sha256=rows[0].get('record_sha256'),
        report_title=rows[0].get('report_title'),
        code_version=rows[0].get('code_version'),
        samples={r['id']: dict(group=r.get('group'), label=r.get('label'), metrics=r.get('metrics') or {})
                 for r in rows},
    )
    GOOD.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    out_dir = out_dir or ROOT / 'out' / '效果好数据集基线'
    out_dir.mkdir(parents=True, exist_ok=True)
    moment = now.strftime('%Y-%m-%d %H:%M')
    cards = []
    for r in rows:
        # 从归档目录取图：定位该样本目录
        sid = r['id'].replace('/', '_')
        sample_dir = GOOD / str(r.get('group') or '未分组') / f"{sid}_p{r.get('page', 1)}"
        imgs = {}
        for f in sample_dir.iterdir():
            if f.name.startswith('输入'):
                imgs['输入'] = thumb(f)
            elif f.name == '处理结果.png':
                imgs['处理结果'] = thumb(f)
            elif f.name == '漏选多选核对.jpg':
                imgs['核对图'] = thumb(f)
        m = r.get('metrics') or {}
        cells = ''.join(f'<td class="r">{fmt(k, m.get(k))}</td>' for k in SHOW)
        figs = ''.join(f'<figure><img src="{v}"><figcaption>{k}</figcaption></figure>' for k, v in imgs.items())
        cards.append(f'<article class="card"><h2>{r["id"]} <span class="muted">{r.get("label") or ""}</span></h2>'
                     f'<div class="figs">{figs}</div><table class="m"><tr>{"".join(f"<th>{METRIC_LABEL[k]}</th>" for k in SHOW)}</tr>'
                     f'<tr>{cells}</tr></table>'
                     f'<p class="muted">修复方式：{m.get("method")}／{m.get("repair_mode")} · '
                     f'定档时间 {r.get("classified_at", "")[:19]} · 代码 {r.get("code_version")}</p></article>')
    pending = load_verdicts(PENDING) if PENDING.is_dir() else []
    pend_rows = ''.join(f'<tr><td>{r["id"]}</td><td>{r.get("label") or ""}</td>'
                        f'<td class="r">{fmt("recall", (r.get("metrics") or {}).get("recall"))}</td></tr>' for r in pending)
    pend_html = (f'<h2>附：当前待解决（{len(pending)} 张，仅备查，不参与防回退）</h2><table><tr><th>样本</th><th>说明</th><th>召回率</th></tr>{pend_rows}</table>' if pending else '')
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>效果好数据集基线报告 {moment}</title>
<style>
body{{margin:0;background:#f4f6f4;color:#243c32;font:15px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1200px;margin:auto;padding:30px 22px 60px}}h1{{font-size:28px;margin:14px 0 4px}}h2{{font-size:18px}}
.note{{background:#eaf4ed;border-left:4px solid #287653;padding:14px 18px;border-radius:0 8px 8px 0;margin:14px 0}}
.muted{{color:#617267;font-size:13px}}table{{border-collapse:collapse;font-size:13.5px;background:white}}td,th{{border:1px solid #d8e1da;padding:7px 12px;text-align:left}}td.r,th.r{{text-align:right}}th{{background:#f0f4f0}}
.card{{background:white;border:1px solid #dde5df;border-radius:12px;padding:18px 22px;margin:16px 0}}.card h2{{margin:0 0 10px;font-size:17px}}
.figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}}figure{{margin:0}}img{{width:100%;height:260px;object-fit:contain;background:#f6f8f6;border-radius:6px}}figcaption{{text-align:center;font-size:12px;color:#617267;padding-top:4px}}
</style></head><body><main>
<h1>效果好数据集基线报告 <span class="muted">{moment}</span></h1>
<p class="muted">来源：{len(rows)} 张人工定档「效果好」样本 · 控制轮 {rows[0].get("report_title")} · 生成时间取自运行时刻</p>
<div class="note"><b>防回退标准（暂定）</b>：以上每张的每个参数都是基线值。以后任何一轮跑出来，
<b>召回率／精确率不许比基线低</b>；<b>漏选、多选、水印区误差、非水印区误差、耗时不得比基线高</b>。
任何一张任一参数违反即算回退，检查命令：<code>tools/build_goodset_baseline_report.py check --run-dir 新轮次目录</code>。
人工改档只增不减：新达标的补进来，已进基线的样本参数只许变好。</div>
{''.join(cards)}
{pend_html}
<footer class="muted">本报告自包含，图片以 base64 内嵌，可单文件发送。全分辨率原图在 data/效果分类数据集/ 各样本目录。</footer>
</main></body></html>'''
    dest = out_dir / f'效果好数据集基线报告_{now.strftime("%Y%m%d_%H%M")}.html'
    dest.write_text(page, encoding='utf-8')
    print(f'基线参数已写入：{BASELINE.relative_to(ROOT)}')
    print(f'基线报告：{dest}')
    return dest


def check(records: dict[str, dict]) -> int:
    if not BASELINE.is_file():
        raise SystemExit(f'基线不存在，先运行 build：{BASELINE}')
    baseline = json.loads(BASELINE.read_text(encoding='utf-8'))
    samples = baseline['samples']
    problems = []
    compared = 0
    warnings = []
    for sid, b in samples.items():
        new = records.get(sid)
        if new is None:
            problems.append(f'缺样本：{sid}（本轮没有它的记录，无法确认未回退）')
            continue
        compared += 1
        nm = new.get('metrics') or {}
        for metric, direction in DIRECTIONS.items():
            bv = (b.get('metrics') or {}).get(metric)
            nv = nm.get(metric)
            if bv is None or nv is None:
                continue
            if metric == 'elapsed_ms':
                # 耗时允许 10% 机器波动（超出才算回退）；其余参数按一位小数粒度
                # 比较：低于粒度的波动（如多选 5 个晕圈像素）不判回退。
                if nv > bv * 1.10:
                    problems.append(f'{sid} · {METRIC_LABEL[metric]} 上升：{fmt(metric, bv)} → {fmt(metric, nv)}')
                elif nv > bv:
                    warnings.append(f'{sid} · {METRIC_LABEL[metric]} 波动 +{(nv - bv) / bv * 100:.1f}%（10% 容差内）')
                continue
            gb, gn = _grain(metric, bv), _grain(metric, nv)
            if direction == 'up' and gn < gb:
                problems.append(f'{sid} · {METRIC_LABEL[metric]} 下降：{fmt(metric, bv)} → {fmt(metric, nv)}')
            if direction == 'down' and gn > gb:
                problems.append(f'{sid} · {METRIC_LABEL[metric]} 上升：{fmt(metric, bv)} → {fmt(metric, nv)}')
    print(f'对比完成：{compared}/{len(samples)} 张与基线比对。')
    for w in warnings:
        print(' ⚠', w)
    if problems:
        print('发现回退/缺失：')
        for p in problems:
            print(' -', p)
        return 1
    print('全部参数均未回退。')
    return 0


def _records_from_run_dir(run_dir: Path) -> dict[str, dict]:
    run = json.loads((run_dir / '运行记录.json').read_text(encoding='utf-8'))
    # 运行记录的指标在顶层；统一包成 {'metrics': {...}} 供 check 使用。
    return {r['id']: dict(metrics={k: r.get(k) for k in DIRECTIONS},
                         method=r.get('method'), repair_mode=r.get('repair_mode'))
            for r in run.get('records', []) if r.get('id')}


def _records_from_classify(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {s['id']: s for s in payload.get('samples', []) if s.get('id')}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='生成基线报告并写入基线参数')
    b.add_argument('--out', type=Path, help='报告输出目录（默认 out/效果好数据集基线/）')
    c = sub.add_parser('check', help='新一轮结果与基线对比，任一参数回退则退出码 1')
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument('--run-dir', type=Path, help='控制轮目录（含运行记录.json）')
    src.add_argument('--json', type=Path, help='报告里导出的效果定档 JSON')
    args = ap.parse_args()
    if args.cmd == 'build':
        build(args.out)
        return 0
    records = _records_from_run_dir(args.run_dir) if args.run_dir else _records_from_classify(args.json)
    return check(records)


if __name__ == '__main__':
    sys.exit(main())
