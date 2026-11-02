"""分步体检：只用检测入口（不修复）对固定回归集逐步打分。

测量第 0 步（有无水印判定）与第 1 步（水印像素定位）的真实表现：
- 存在性：预测选区有无 vs 真值 alpha 有无（含无水印负例的误触发统计）
- 定位：alpha>0 全部召回、明显笔画召回（alpha>=0.08）、精确率、
  背景多选、封闭字洞误选、选区占图比例
- 每次推理只给输入图；alpha 等参考在推理结束后才读取。

只读诊断，不修改检测、修复或路由逻辑。第 2~3 步（修复与验收）
由 tools/run_min_regression.py 的完整回归另行取证。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT = ROOT / 'data/watermark_regression_min_v1'


def read_rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def enclosed_background(alpha: np.ndarray) -> np.ndarray:
    """背景中从图边框走不到的部分（封闭字洞）。与独立评分入口同口径。"""
    import cv2
    support = (alpha > 0).astype(np.uint8)
    free = (1 - support).astype(np.uint8)
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
    return ff == 0


def b64(arr: np.ndarray, max_side: int = 700, quality: int = 68) -> str:
    """预览图降采样后 JPEG base64 内嵌，保证单个 HTML 完整可查看。"""
    img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert('RGB').save(buf, 'JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def tint(source: np.ndarray, region: np.ndarray, color: tuple, strength: float) -> np.ndarray:
    img = Image.fromarray(np.asarray(source, dtype=np.uint8)).convert('RGB')
    a = np.asarray(img).astype(np.float32)
    a[region] = (1 - strength) * a[region] + strength * np.array(color, np.float32)
    return a.astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=DEFAULT)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 个用例（调试用）')
    args = ap.parse_args()

    from tools.run_min_regression import validate
    from wm import documents, structure
    from wm.contract import Thresholds
    from wm import masks as masks_mod

    cases, checks = validate(args.root)
    if args.limit:
        cases = cases[:args.limit]

    out = args.out or ROOT / 'out' / ('分步体检_' + datetime.now().astimezone().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)

    thr = Thresholds()
    pages_out = []
    for index, case in enumerate(cases, 1):
        for i in range(case['page_count']):
            item = dict(case_id=case['case_id'], label=case['label'], suite=case['suite'],
                        subtype=case.get('subtype'), negative=case['negative'],
                        score_mode=case['score_mode'], page=i + 1)
            try:
                loaded = documents.load_pages(args.root / case['paths']['input'], dpi=96)
                assert len(loaded) == case['page_count']
                page_img = loaded[i]
                t0 = time.perf_counter()
                structural = structure.process_text_page(page_img, thr)
                if structural is not None:
                    mask = structural.mask
                    item['entry'] = 'structural'
                else:
                    mask = masks_mod.estimate_single_page(page_img.image, thr, use_ocr=True)
                    item['entry'] = 'raster'
                item['mask_ms'] = round((time.perf_counter() - t0) * 1000)
                pred = mask.binary() > 0
                item.update(method=mask.method, repair_mode=mask.repair_mode,
                            confidence=round(float(mask.confidence), 4),
                            selected_pixels=int(pred.sum()),
                            selected_ratio=round(float(pred.mean()), 6))
                # 推理结束，之后才允许读参考
                truth = case['pages'][i]['paths'] if case['kind'] == 'pdf' else case['paths']
                alpha_path = truth.get('alpha')
                alpha = None
                if alpha_path and (args.root / alpha_path).is_file():
                    with Image.open(args.root / alpha_path) as a:
                        alpha = np.asarray(a.convert('L').resize(
                            (page_img.width, page_img.height), Image.Resampling.LANCZOS),
                            np.float32) / 255
                src = np.asarray(page_img.image)
                if alpha is None:
                    item.update(gt_available=False, gt_present=None,
                                predicted_present=bool(pred.any()),
                                recall_all=None, recall_prominent=None, precision=None,
                                false_positive_pixels=None, missed_pixels=None,
                                hole_total=None, hole_selected_pixels=None)
                    item['viz'] = dict(input=b64(src))
                else:
                    gt = alpha > 0
                    prominent = alpha >= 0.08
                    inter = int((pred & gt).sum())
                    holes = enclosed_background(alpha)
                    item.update(
                        gt_available=True, gt_present=bool(gt.any()),
                        predicted_present=bool(pred.any()),
                        recall_all=round(inter / int(gt.sum()), 6) if gt.any() else None,
                        recall_prominent=(round(int((pred & prominent).sum()) / int(prominent.sum()), 6)
                                          if prominent.any() else None),
                        precision=round(inter / int(pred.sum()), 6) if pred.any() else None,
                        false_positive_pixels=int((pred & ~gt).sum()),
                        missed_pixels=int((~pred & gt).sum()),
                        hole_total=int(holes.sum()),
                        hole_selected_pixels=int((pred & holes).sum()))
                    item['viz'] = dict(
                        input=b64(src),
                        gt=b64(tint(src, gt, (220, 40, 60), 0.40)),
                        pred=b64(tint(src, pred, (30, 200, 120), 0.45)))
                    if item['missed_pixels'] or item['false_positive_pixels']:
                        diag = np.asarray(Image.fromarray(src).convert('RGB')).copy()
                        diag = tint(diag, gt & ~pred, (40, 90, 220), 0.85)   # 漏选：蓝
                        diag = tint(diag, pred & ~gt, (240, 140, 30), 0.85)  # 多选：橙
                        item['viz']['diagnostic'] = b64(diag, quality=72)
            except Exception as exc:  # 失败保留在清单与统计中
                item.update(status='error', error=f'{type(exc).__name__}: {exc}')
            pages_out.append({k: v for k, v in item.items() if not k.startswith('_')})
        print(f'[{index}/{len(cases)}] {case["case_id"]}', flush=True)

    payload = dict(created_at=datetime.now().astimezone().isoformat(), checks=checks,
                   note='只测检测阶段（第0~1步），未修复；alpha 在推理后读取。'
                        '无 alpha 的页召回/精确率为 null（未评估），只检查是否误触发。',
                   pages=pages_out)
    (out / 'stage01_metrics.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    write_report(out, payload)
    print(out / '分步体检报告.html')
    print('第 2~3 步（修复与验收）请看 tools/run_min_regression.py --suite all 的独立报告。')
    return 0


def write_report(out: Path, payload: dict) -> None:
    from html import escape
    pages = payload['pages']
    ok = [p for p in pages if p.get('status') != 'error']

    # ---- 第 0 步：存在性混淆矩阵（有真值的页）
    with_gt = [p for p in ok if p.get('gt_available')]
    tp = sum(1 for p in with_gt if p['gt_present'] and p['predicted_present'])
    fn = sum(1 for p in with_gt if p['gt_present'] and not p['predicted_present'])
    fp_p = [p for p in with_gt if not p['gt_present'] and p['predicted_present']]
    tn = sum(1 for p in with_gt if not p['gt_present'] and not p['predicted_present'])
    # 无真值的负例：只看是否误触发
    no_gt_neg = [p for p in ok if not p.get('gt_available') and p.get('negative')]
    no_gt_triggered = [p for p in no_gt_neg if p['predicted_present']]
    # manual_review 页不算存在性对错（无精确真值）
    review = [p for p in ok if p.get('score_mode') == 'manual_review']

    # ---- 第 1 步：定位指标
    def avg(key):
        vals = [p[key] for p in with_gt if p.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    stats = dict(
        recall_all=avg('recall_all'), recall_prominent=avg('recall_prominent'),
        precision=avg('precision'),
        fp_pixels=sum(p['false_positive_pixels'] or 0 for p in with_gt),
        fn_pixels=sum(p['missed_pixels'] or 0 for p in with_gt),
        hole_selected=sum(p['hole_selected_pixels'] or 0 for p in with_gt),
        hole_total=sum(p['hole_total'] or 0 for p in with_gt),
        mask_ms_avg=round(sum(p['mask_ms'] for p in ok) / len(ok)) if ok else None,
        mask_ms_max=max((p['mask_ms'] for p in ok), default=None))

    from collections import Counter
    methods = Counter(p.get('method', 'error') for p in ok)
    entries = Counter(p.get('entry', 'error') for p in ok)

    # ---- 分类型汇总
    by_subtype = {}
    for p in with_gt:
        if not p['gt_present']:
            continue
        row = by_subtype.setdefault(p.get('subtype') or p['label'], dict(n=0, recall=[], precision=[]))
        row['n'] += 1
        if p.get('recall_all') is not None:
            row['recall'].append(p['recall_all'])
        if p.get('precision') is not None:
            row['precision'].append(p['precision'])
    subtype_rows = ''.join(
        f'<tr><td>{escape(k)}</td><td>{v["n"]}</td>'
        f'<td>{escape(str(round(sum(v["recall"]) / len(v["recall"]), 4) if v["recall"] else "—"))}</td>'
        f'<td>{escape(str(round(sum(v["precision"]) / len(v["precision"]), 4) if v["precision"] else "—"))}</td></tr>'
        for k, v in sorted(by_subtype.items()))

    conf_rows = ''.join(
        f'<tr><td>真值有水印 ({sum(1 for p in with_gt if p["gt_present"])}页)</td>'
        f'<td>检出 {tp} 页</td><td>漏检 {fn} 页</td></tr>'
        f'<tr><td>真值无水印 ({sum(1 for p in with_gt if not p["gt_present"])}页)</td>'
        f'<td>误触发 {len(fp_p)} 页</td><td>正确不动 {tn} 页</td></tr>')

    neg_extra = ''
    if no_gt_neg:
        neg_extra = (f'<p>另有 {len(no_gt_neg)} 张无真值负例（无法算召回，只检查误触发）：'
                     f'误触发 {len(no_gt_triggered)} 张。</p>')
    if review:
        triggered = sum(1 for p in review if p['predicted_present'])
        neg_extra += (f'<p>真实扩展 manual_review {len(review)} 页（无精确真值，不计存在性对错）：'
                      f'系统选择处理 {triggered} 页，原样保留 {len(review) - triggered} 页。</p>')

    method_rows = ''.join(f'<tr><td>{escape(k)}</td><td>{v}</td></tr>'
                          for k, v in methods.most_common())

    cards = []
    for p in pages:
        if p.get('status') == 'error':
            cards.append(f'<article><h2>{escape(p["case_id"])} 第{p["page"]}页</h2>'
                         f'<p class="bad">异常：{escape(p.get("error", ""))}</p></article>')
            continue
        viz = p.get('viz', {})
        figs = ''.join(f'<figure><img src="{v}"><figcaption>{t}</figcaption></figure>'
                       for v, t in [(viz.get('input'), '输入'), (viz.get('gt'), '真值alpha'),
                                    (viz.get('pred'), '预测选区'), (viz.get('diagnostic'), '蓝=漏选 橙=多选')]
                       if v)
        m = {k: p.get(k) for k in ('method', 'entry', 'repair_mode', 'confidence', 'mask_ms',
                                   'selected_pixels', 'selected_ratio', 'recall_all',
                                   'recall_prominent', 'precision', 'false_positive_pixels',
                                   'missed_pixels', 'hole_total', 'hole_selected_pixels',
                                   'predicted_present', 'gt_present')}
        cards.append(f'<article><h2>{escape(p["case_id"] + " · " + str(p["label"]))} · 第 {p["page"]} 页</h2>'
                     f'<div class="pair">{figs}</div><pre>{escape(json.dumps(m, ensure_ascii=False, indent=1))}</pre></article>')

    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>分步体检报告 2026-11-02</title>
<style>body{{font:15px/1.8 system-ui;background:#f1f4f2;color:#253c35;margin:24px;max-width:1500px;margin:auto}}
section,article{{background:white;border:1px solid #dde5df;border-radius:10px;padding:22px;margin:18px 0}}
table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:8px;border-bottom:1px solid #e0e6e1;text-align:left;font-size:14px}}
.pair{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}}figure{{margin:0}}img{{width:100%;background:#f3f5f7}}
figcaption{{font-size:12px;color:#617267;text-align:center}}pre{{white-space:pre-wrap;font-size:12px;background:#f6f8f6;padding:10px;border-radius:6px}}
.bad{{color:#a33}}h1{{line-height:1.4}}</style></head><body>
<h1>分步体检报告（第 0~1 步：存在性判定与像素定位）</h1>
<p>固定回归集 43 输入／59 页，只运行检测入口、不修复；alpha 在推理结束后才读取。
本报告图片全部 base64 内嵌，单文件可完整查看。第 2~3 步（修复与验收）由完整回归另行报告。
诊断日期 2026-11-02。</p>
<section><h2>第 0 步：有无水印判定</h2><table>{conf_rows}</table>
<p>检出率（有水印的页中被检出）：{escape(str(round(tp/(tp+fn),4) if tp+fn else None))}；
误触发率（无水印页中被动手）：{escape(str(round(len(fp_p)/(len(fp_p)+tn),4) if len(fp_p)+tn else None))}。</p>
{neg_extra}
<details><summary>误触发清单</summary>{''.join(f'<p class="bad">{escape(p["case_id"])} 第{p["page"]}页：方法 {escape(str(p.get("method")))}，选中 {p["selected_pixels"]:,} 像素（占图 {p["selected_ratio"]:.4%}）</p>' for p in fp_p + no_gt_triggered) or '<p>无</p>'}</details>
</section>
<section><h2>第 1 步：水印像素定位（有真值的正例页平均）</h2>
<table><tr><th>全部召回(alpha&gt;0)</th><th>明显笔画召回(≥0.08)</th><th>精确率</th><th>总多选背景像素</th><th>总漏选像素</th><th>字洞误选/字洞总数</th><th>检测耗时均/最大</th></tr>
<tr><td>{escape(str(stats['recall_all']))}</td><td>{escape(str(stats['recall_prominent']))}</td><td>{escape(str(stats['precision']))}</td>
<td>{stats['fp_pixels']:,}</td><td>{stats['fn_pixels']:,}</td>
<td>{stats['hole_selected']:,} / {stats['hole_total']:,}</td>
<td>{stats['mask_ms_avg']} / {stats['mask_ms_max']} ms</td></tr></table>
<h3>分类型召回（有水印的正例）</h3><table><tr><th>子类型</th><th>页数</th><th>平均召回</th><th>平均精确率</th></tr>{subtype_rows}</table>
<h3>通道命中分布</h3><table><tr><th>方法</th><th>页数</th></tr>{method_rows}</table>
<p>结构剥离 {entries.get('structural', 0)} 页，光栅检测 {entries.get('raster', 0)} 页。</p>
</section>
<h2>逐页明细（59 页全保留，含异常）</h2>{''.join(cards)}
</body></html>'''
    (out / '分步体检报告.html').write_text(page)


if __name__ == '__main__':
    raise SystemExit(main())
