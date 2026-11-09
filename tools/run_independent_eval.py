"""One-look baseline evaluation on the locked independent set v1.

Rules honoured here (AGENTS.md sections 3 and 6):
- source-only inference: the application receives exactly one image path;
  clean references and alpha are loaded only AFTER inference returns;
- every expected sample stays in the denominator, even on failure/timeout;
- negatives must remain byte-identical (zero selected pixels, zero changes);
- metrics are reported raw: recall over alpha>0 AND recall over alpha>=64
  (明显笔画) separately, precision, false-selected background pixels,
  enclosed-hole false selections, mask share, method/repair, elapsed, errors.

The numbers this produces are a ONE-LOOK baseline. After the run, the set is
frozen: any later tuning informed by these numbers downgrades the set to
development/regression use and requires a new independent set.
"""
from __future__ import annotations
import argparse
import collections
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_ROOT = ROOT / 'data/watermark_independent_v1'


def read_rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate(root: Path) -> list[dict]:
    rows = read_rows(root / 'manifest.jsonl')
    qa = json.loads((root / 'qa/validation.json').read_text())
    assert qa['pass_all'], 'dataset QA did not pass; refusing to evaluate'
    for r in rows:
        for role, relpath in r['paths'].items():
            assert sha(root / relpath) == r['sha256'][role], f'hash mismatch {r["id"]} {role}'
    return rows


def enclosed_background(alpha: np.ndarray) -> np.ndarray:
    """Background pixels not reachable from the image border without crossing
    the watermark support (i.e. 字洞 / enclosed gaps)."""
    support = (alpha > 0).astype(np.uint8)
    free = (1 - support).astype(np.uint8)
    # flood fill from the border through background
    ff = free.copy()
    mask = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    for x in range(ff.shape[1]):
        for y in (0, ff.shape[0] - 1):
            if ff[y, x]:
                cv2.floodFill(ff, mask, (x, y), 2)
    for y in range(ff.shape[0]):
        for x in (0, ff.shape[1] - 1):
            if ff[y, x]:
                cv2.floodFill(ff, mask, (x, y), 2)
    enclosed = (ff == 0)  # background never reached by the flood
    return enclosed


def score_page(result, alpha: np.ndarray, clean: np.ndarray, neg: bool, thr) -> dict:
    from wm import verdict
    h, w = result.source.image.shape[:2]
    if alpha.shape[:2] != (h, w):
        alpha = np.asarray(Image.fromarray(alpha).resize((w, h), Image.Resampling.LANCZOS), np.float32) / 255.0
    gt = alpha > 0
    pred = result.mask.binary().astype(bool)
    selected = int(pred.sum())
    gt_px = int(gt.sum())
    hit = int(np.logical_and(pred, gt).sum())
    strong = alpha >= 64 / 255.0
    strong_px = int(strong.sum())
    strong_hit = int(np.logical_and(pred, strong).sum())
    false_bg = int(np.logical_and(pred, ~gt).sum())
    holes = enclosed_background(alpha.astype(np.float32))
    hole_false = int(np.logical_and(pred, holes).sum())
    vr = verdict.build_verdict(source=result.source.image, output=result.output.image, mask=result.mask,
                               engine=result.verdict.engine, repairs=result.verdict.repairs,
                               elapsed_ms=result.verdict.elapsed_ms, stage_ms=result.verdict.stage_ms,
                               thr=thr, clean_reference=clean, alpha_reference=alpha)
    changed = float(np.any(result.source.image != result.output.image, axis=2).mean())
    m = dict(mask_method=str(result.mask.method), coverage=float(result.mask.coverage()),
             selected_px=selected, gt_px=gt_px, mask_share=selected / (h * w),
             recall_all=(hit / gt_px) if gt_px else None,
             recall_strong=(strong_hit / strong_px) if strong_px else None,
             precision=(hit / selected) if selected else None,
             missed_px=gt_px - hit, false_bg_px=false_bg, hole_false_px=hole_false,
             changed_fraction=changed, elapsed_ms=int(result.verdict.elapsed_ms),
             psnr=vr.metrics.get('psnr'), ssim=vr.metrics.get('ssim'),
             fidelity_mean_delta=vr.metrics.get('fidelity_mean_delta'),
             fidelity_p99_delta=vr.metrics.get('fidelity_p99_delta'),
             residual_coverage=vr.metrics.get('residual_coverage'),
             ink_iou=vr.metrics.get('ink_iou'),
             passed=bool(vr.passed) if not neg else None)
    if neg:
        m['negative_safe'] = bool(selected == 0 and changed == 0)
    return m


def aggregate(pages: list[dict], key: str) -> dict:
    vals = collections.defaultdict(list)
    for p in pages:
        if p['metrics'] and p['metrics'].get(key) is not None:
            vals[p['metrics'].get('group', '')].append(p['metrics'][key])
    return {g: dict(n=len(v), mean=round(float(np.mean(v)), 4)) for g, v in vals.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--dpi', type=int, default=96)
    ap.add_argument('--render-only', action='store_true', help='只重建汇总与自包含报告，不重跑推理')
    args = ap.parse_args()
    if args.render_only:
        rerender(args.out)
        print(args.out / 'report.html')
        return 0
    rows = validate(args.root)
    out = args.out or ROOT / 'out' / ('独立验证基线_' + datetime.now().astimezone().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    from wm import pipeline
    from wm.contract import Thresholds
    # 桌面默认口径：笔画补全常开（watermark_recall_priority=True）。
    # 不能用 Thresholds() 裸默认——那是 False，会导致整轮独立验证与桌面封存基线口径不一致（假性退化）。
    thr = Thresholds(watermark_recall_priority=True)
    assert thr.watermark_recall_priority, '独立验证必须沿用桌面默认口径（笔画补全常开）'

    results = []
    for index, case in enumerate(rows, 1):
        rec = dict(id=case['id'], scenario=case['scenario'], subtype=case['subtype'], label=case['label'],
                   base_id=case['base_id'], background=case['background'], negative=case['is_negative'],
                   source_group=case['source_group'])
        try:
            processed = pipeline.process_file(args.root / case['paths']['watermarked'], dpi=args.dpi, thr=thr)
            assert len(processed) == 1
            result = processed[0]
            folder = out / case['id']
            folder.mkdir(exist_ok=True)

            def png(name, array):
                dest = folder / name
                Image.fromarray(np.asarray(array, dtype=np.uint8)).save(dest)
                return dest.relative_to(out).as_posix()

            item = dict(input_preview=png('input.png', result.source.image),
                        output=png('output.png', result.output.image),
                        predicted_mask=png('predicted_mask.png', result.mask.binary()),
                        clean_preview=png('clean.png', np.asarray(Image.open(args.root / case['paths']['clean']).convert('RGB'))),
                        status='scored')
            with Image.open(args.root / case['paths']['alpha']) as a:
                alpha = np.asarray(a.convert('L'), np.float32) / 255.0
            clean = np.asarray(Image.open(args.root / case['paths']['clean']).convert('RGB').resize(
                (result.source.width, result.source.height), Image.Resampling.LANCZOS))
            metrics = score_page(result, alpha, clean, case['is_negative'], thr)
            metrics['group'] = case['scenario'] + ' ' + case['label']
            item['metrics'] = metrics
            rec['pages'] = [item]
        except Exception as exc:  # failures stay in the denominator
            rec['pages'] = [dict(status='error', error=f'{type(exc).__name__}: {exc}')]
            print(f'[{index}/{len(rows)}] {case["id"]} ERROR {exc}', flush=True)
        results.append(rec)
        done = sum(len(r['pages']) for r in results)
        print(f'[{index}/{len(rows)}] {case["id"]}: {rec["pages"][0]["status"]}', flush=True)
        # incremental summary for crash safety
        (out / 'summary.partial.json').write_text(json.dumps(dict(results=results), ensure_ascii=False, indent=2))

    positives_pages = [p for r in results if not r['negative'] for p in r['pages'] if p.get('metrics')]
    positives = [p['metrics'] for p in positives_pages]
    negatives = [p for r in results if r['negative'] for p in r['pages']]
    neg_safe = sum(1 for p in negatives if p.get('metrics', {}).get('negative_safe'))
    by_scenario = collections.defaultdict(list)
    for r in results:
        for p in r['pages']:
            if p.get('metrics'):
                by_scenario[r['scenario'] + ' ' + r['label']].append(p['metrics'])

    def col(key, group):
        v = [m[key] for m in group if m.get(key) is not None]
        return round(float(np.mean(v)), 4) if v else None

    scenario_table = {g: dict(n=len(v), recall_all=col('recall_all', v), recall_strong=col('recall_strong', v),
                              precision=col('precision', v), false_bg_px=col('false_bg_px', v),
                              hole_false_px=col('hole_false_px', v), psnr=col('psnr', v),
                              ssim=col('ssim', v), elapsed_s=col('elapsed_ms', v) and round(col('elapsed_ms', v) / 1000, 2))
                      for g, v in by_scenario.items()}
    overall = dict(n=len(positives),
                   recall_all=col('recall_all', positives), recall_strong=col('recall_strong', positives),
                   precision=col('precision', positives), psnr=col('psnr', positives), ssim=col('ssim', positives),
                   fidelity_mean_delta=col('fidelity_mean_delta', positives),
                   residual_coverage=col('residual_coverage', positives),
                   elapsed_s=round(col('elapsed_ms', positives) / 1000, 2) if col('elapsed_ms', positives) else None)
    errors = [r['id'] for r in results for p in r['pages'] if p['status'] == 'error']
    summary = dict(created_at=datetime.now().astimezone().isoformat(), root=str(args.root),
                   manifest_sha256=sha(args.root / 'manifest.jsonl'), dpi=args.dpi,
                   code_snapshot=dict(git= os.popen('git rev-parse HEAD 2>/dev/null').read().strip() or None),
                   note='ONE-LOOK baseline. Source-only inference; references loaded after inference. '
                        'After this run the set is frozen; tuning on it downgrades it to development use.',
                   overall=overall, negatives=dict(n=len(negatives), safe=neg_safe),
                   errors=errors, by_scenario=scenario_table, results=results)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out / 'summary.partial.json').unlink(missing_ok=True)

    render_report(out, summary)
    print(out / 'report.html')
    print('ONE-LOOK baseline complete. Set is now frozen.')
    return int(bool(errors) or neg_safe < len(negatives))


def _data_uri(out: Path, rel: str, max_px: int = 800) -> str:
    """Downscale and embed an image as a base64 data URI so the delivered
    HTML is fully self-contained (a single file renders completely when sent
    to someone else). Full-resolution originals stay on disk for local work."""
    import base64, io
    im = Image.open(out / rel)
    if max(im.size) > max_px:
        ratio = max_px / max(im.size)
        im = im.resize((max(1, int(im.width * ratio)), max(1, int(im.height * ratio))),
                       Image.Resampling.NEAREST if im.mode == '1' else Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    if im.mode in ('L', '1'):
        im.convert('L').save(buf, format='PNG', optimize=True)
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    im.convert('RGB').save(buf, format='JPEG', quality=78)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def _pct(v):
    return '—' if v is None else f'{v * 100:.1f}%'


def _cls_recall(v):
    if v is None:
        return 'bad'
    return 'bad' if v < .3 else ('mid' if v < .6 else 'good')


def _cls_prec(v):
    if v is None:
        return 'bad'
    return 'bad' if v < .5 else ('mid' if v < .8 else 'good')


def _cls_bg(v):
    return 'bad' if v > 20000 else ('mid' if v > 5000 else 'good')


def render_report(out: Path, summary: dict) -> None:
    e = html.escape
    scenario_table = summary['by_scenario']
    overall = summary['overall']
    neg = summary['negatives']
    errors = summary['errors']

    # 每张的页指标按场景分组，并给一句话诊断
    pages_by_group = collections.defaultdict(list)
    for r in summary['results']:
        for p in r['pages']:
            m = p.get('metrics')
            if not m:
                continue
            entry = dict(rec=r['id'], bg=r['background'], page=p, m=m, neg=r['negative'])
            pages_by_group[m.get('group') or (r['scenario'] + ' ' + r['label'])].append(entry)

    def diagnose(m):
        parts = []
        ra = m.get('recall_all')
        if ra is not None:
            if ra < .3:
                parts.append(f'漏检严重（找回 {ra * 100:.0f}%）')
            elif ra < .6:
                parts.append(f'漏检偏多（找回 {ra * 100:.0f}%）')
        else:
            parts.append('未检出')
        fb = m.get('false_bg_px') or 0
        if fb > 20000:
            parts.append(f'误伤严重（多选背景 {fb:,}px）')
        elif fb > 5000:
            parts.append(f'有误伤（多选 {fb:,}px）')
        pr = m.get('precision')
        if pr is not None and pr < .5:
            parts.append(f'选得不准（精确率 {pr * 100:.0f}%）')
        if m.get('negative_safe'):
            parts.append('负例零改动 ✓')
        return '；'.join(parts) if parts else '表现较好'

    ordered = sorted(((g, d) for g, d in scenario_table.items()),
                     key=lambda kv: (kv[1]['recall_all'] is None, kv[1]['recall_all'] if kv[1]['recall_all'] is not None else 0))

    def bar(v, cls):
        w = 0 if v is None else max(2, min(100, v * 100))
        return f'<div class="bar {cls}"><i style="width:{w:.0f}%"></i><b>{_pct(v)}</b></div>'

    rows_html = ''.join(
        f'<tr><td><b>{e(g.split(" ", 1)[1] if " " in g else g)}</b><br/><span class="muted">{e(g.split(" ", 1)[0])} · {d["n"]}张</span></td>'
        f'<td>{bar(d["recall_all"], _cls_recall(d["recall_all"]))}</td>'
        f'<td>{bar(d["recall_strong"], _cls_recall(d["recall_strong"]))}</td>'
        f'<td>{bar(d["precision"], _cls_prec(d["precision"]))}</td>'
        f'<td class="{_cls_bg(d["false_bg_px"] or 0)}">{d["false_bg_px"]:,.0f}px</td>'
        f'<td>{d["psnr"]}</td><td>{d["elapsed_s"]}s</td></tr>'
        for g, d in ordered)

    worst_names = '、'.join(g.split(' ', 1)[1] for g, _ in ordered[:3])

    def figs_for(p, size=760, detail=False):
        return ''.join(f'<figure><img loading="lazy" src="{_data_uri(out, u, max_px=size)}"><figcaption>{t}</figcaption></figure>'
                       for u, t in [(p.get('input_preview'), '输入'), (p.get('output'), '实际输出'),
                                    (p.get('predicted_mask'), '预测掩码'), (p.get('clean_preview'), '干净参考')] if u)

    # 重点问题图集：每个非负例场景里“最差的一张”（先比找回率，再比误伤）
    gallery = []
    for g, d in ordered:
        if g.startswith('N00'):
            continue
        group_pages = [x for x in pages_by_group.get(g, []) if not x['neg']]
        if not group_pages:
            continue
        worst = min(group_pages, key=lambda x: ((x['m'].get('recall_all') if x['m'].get('recall_all') is not None else -1),
                                                -(x['m'].get('false_bg_px') or 0)))
        m = worst['m']
        gallery.append(
            f'<article class="case"><h3>{e(g.split(" ", 1)[1])}<span class="muted">　{e(g.split(" ", 1)[0])} · 最差样本</span></h3>'
            f'<p class="diag">{e(diagnose(m))}</p><div class="pair">{figs_for(worst["page"], 760)}</div></article>')

    detail_blocks = []
    for g, d in ordered:
        items = []
        for x in sorted(pages_by_group.get(g, []), key=lambda x: x['rec']):
            m = x['m']
            line = (f'<div class="mini"><b>{e(x["rec"])}</b><span class="muted">{e(x["bg"])}</span>'
                    f'<span>找回 {_pct(m.get("recall_all"))} · 明显笔画 {_pct(m.get("recall_strong"))} · '
                    f'精确 {_pct(m.get("precision"))} · 误选 {(m.get("false_bg_px") or 0):,}px · '
                    f'{m.get("elapsed_ms", 0) / 1000:.1f}s · {e(str(m.get("mask_method")))}</span>'
                    f'<span class="muted">{e(diagnose(m))}</span></div>'
                    f'<div class="pair small">{figs_for(x["page"], 460)}</div>')
            items.append(line)
        detail_blocks.append(
            f'<details><summary><b>{e(g)}</b>　找回 {_pct(d["recall_all"])} · 精确 {_pct(d["precision"])} · 误选 {d["false_bg_px"]:,.0f}px（{d["n"]}张，点击展开）</summary>'
            + ''.join(items) + '</details>')

    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>独立验证集一次性基线 2026-11-02</title><style>
*{{box-sizing:border-box}}body{{font:15px/1.8 system-ui,"PingFang SC",sans-serif;background:#f1f4f8;color:#25394e;padding:25px;max-width:1400px;margin:auto}}
.board{{display:flex;flex-wrap:wrap;gap:14px;margin:18px 0}}
.chip{{background:white;border:1px solid #dae4ef;border-radius:12px;padding:14px 22px;min-width:170px}}
.chip .v{{font-size:30px;font-weight:700;line-height:1.2}}.chip .k{{font-size:13px;color:#61727f}}
.bad{{color:#c0392b}}.mid{{color:#b0761a}}.good{{color:#1c7a4d}}
table{{border-collapse:collapse;font-size:14px;background:white;width:100%}}td,th{{border:1px solid #d5dfe8;padding:8px 12px;text-align:left}}th{{background:#eef3f8}}
.bar{{position:relative;min-width:150px;height:20px;background:#eef1f4;border-radius:5px;overflow:hidden}}
.bar i{{position:absolute;left:0;top:0;bottom:0;border-radius:5px}}.bar.bad i{{background:#e8b4ac}}.bar.mid i{{background:#ecd9a8}}.bar.good i{{background:#b7dcc6}}
.bar b{{position:absolute;left:8px;top:0;line-height:20px;font-size:12px}}
.case{{background:white;border:1px solid #dae4ef;border-radius:10px;padding:18px 22px;margin:16px 0}}
.case h3{{margin:0 0 6px;font-size:17px}}.diag{{margin:4px 0 12px;font-weight:600;color:#a03a2c}}
.pair{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}figure{{margin:0}}img{{width:100%;height:280px;object-fit:contain;background:#f3f5f7}}
.pair.small img{{height:200px}}
details{{background:white;border:1px solid #dae4ef;border-radius:10px;padding:14px 20px;margin:14px 0}}summary{{cursor:pointer;font-size:15px}}
.mini{{display:flex;gap:14px;flex-wrap:wrap;align-items:baseline;margin:16px 0 8px;font-size:13px}}
.note{{background:#fff8e6;border:1px solid #e8d9a0;border-radius:8px;padding:12px 18px;font-size:14px}}
.muted{{color:#7a8a94;font-size:12px;font-weight:400}}
</style></head><body>
<h1>独立验证集一次性基线（2026-11-02）</h1>
<p class="note"><b>一次性查看，已锁柜</b>：此后任何根据本报告调整算法的行为，都会使该独立集降级为开发用途，需要另建独立集 v2。推理只拿到输入图，干净参考与 alpha 在推理完成后才加载。本报告图片全部 base64 内嵌，单文件可直接发送。</p>
<details class="explain"><summary><b>这些数字怎么算的（指标口径）</b></summary>
<p>评测拿程序的"预测选区"和生成样本时已知的真实水印区域（alpha 图，0–255 不透明度）逐像素对比。每个像素只有四种情况：</p>
<table><tr><th></th><th>程序选了</th><th>程序没选</th></tr>
<tr><td><b>真是水印</b></td><td class="good">A 命中</td><td class="bad">B 漏检</td></tr>
<tr><td><b>真不是水印</b></td><td class="bad">C 误伤</td><td>D 正确忽略</td></tr></table>
<ul>
<li><b>找回率（全部笔画）</b> = A÷(A+B)：真实水印笔画被选中的比例；分母含淡笔画，不能用"淡的看不见"掩盖漏检。</li>
<li><b>明显笔画找回</b> = 只把 alpha≥64（不透明度约 25% 以上）的浓笔画代入同一公式。</li>
<li><b>精确率</b> = A÷(A+C)：选中的像素里真是水印的比例，衡量误圈背景/正文的多少。</li>
<li><b>误选背景</b> = C 的像素数（绝对面积）；<b>字洞误选</b> = C 中被水印笔画完全围死的空洞部分（从图边框做洪水填充，走不到的背景即封闭空洞），衡量"糊成实心块、填掉字内空白"的问题。</li>
<li><b>PSNR / SSIM</b> = 修复输出与干净原图的相似度（重建质量）。注意：它<b>不是</b>"水印去干净"的度量——找回率低时输出可以很像原图但水印还在。</li>
<li><b>无水印负例安全</b> = 无水印图必须选区为空且输出与输入逐字节一致。</li>
<li>颜色口径：找回率/精确率进度条，红 &lt;30%、黄 30–60%、绿 ≥60%；精确率红 &lt;50%、黄 50–80%、绿 ≥80%；误选背景红 &gt;2 万px、黄 5 千–2 万px。</li>
</ul>
<p class="muted">统计方式：场景行 = 该场景各张指标的算术平均；总体 = 全部正例的算术平均。失败/超时样本保留在分母中。</p>
</details>
<h2>一、一眼结论</h2>
<div class="board">
<div class="chip"><div class="v bad">{overall['recall_all'] * 100:.1f}%</div><div class="k">水印找回率（全部笔画）</div></div>
<div class="chip"><div class="v mid">{overall['recall_strong'] * 100:.1f}%</div><div class="k">明显笔画找回率</div></div>
<div class="chip"><div class="v mid">{overall['precision'] * 100:.1f}%</div><div class="k">选区精确率</div></div>
<div class="chip"><div class="v {_cls_bg(overall['false_bg_px'] or 0)}">{overall['false_bg_px']:,.0f}px</div><div class="k">平均误选背景/张</div></div>
<div class="chip"><div class="v good">{neg['safe']}/{neg['n']}</div><div class="k">无水印图零误伤</div></div>
<div class="chip"><div class="v">{overall['elapsed_s']}s</div><div class="k">平均耗时/张（本机CPU）</div></div>
</div>
<p style="font-size:17px">最需要修的三类：<b class="bad">{e(worst_names)}</b>。负例全部安全（无水印图没有被误改）。评测错误 {len(errors)} 个。</p>
<h2>二、场景榜单（按找回率从差到好）</h2>
<table><tr><th>场景</th><th>找回率(全部)</th><th>明显笔画找回</th><th>精确率</th><th>误选背景</th><th>PSNR</th><th>耗时</th></tr>{rows_html}</table>
<h2>三、重点问题样本（每类最差的一张：输入 → 输出 → 掩码 → 参考）</h2>
{''.join(gallery)}
<h2>四、逐张明细（默认折叠，点击展开）</h2>
{''.join(detail_blocks)}
</body></html>'''
    (out / 'report.html').write_text(page)


def rerender(out: Path) -> None:
    """Rebuild overall metrics + the self-contained report from a finished run
    without touching inference artifacts or numbers."""
    summary = json.loads((out / 'summary.json').read_text())
    positives = [p['metrics'] for r in summary['results'] if not r['negative']
                 for p in r['pages'] if p.get('metrics')]

    def col(key, group):
        v = [m[key] for m in group if m.get(key) is not None]
        return round(float(np.mean(v)), 4) if v else None
    summary['overall'] = dict(n=len(positives),
                              recall_all=col('recall_all', positives), recall_strong=col('recall_strong', positives),
                              precision=col('precision', positives), psnr=col('psnr', positives), ssim=col('ssim', positives),
                              fidelity_mean_delta=col('fidelity_mean_delta', positives),
                              fidelity_p99_delta=col('fidelity_p99_delta', positives),
                              residual_coverage=col('residual_coverage', positives),
                              false_bg_px=col('false_bg_px', positives), hole_false_px=col('hole_false_px', positives),
                              elapsed_s=round(col('elapsed_ms', positives) / 1000, 2) if col('elapsed_ms', positives) else None)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    render_report(out, summary)


if __name__ == '__main__':
    raise SystemExit(main())
