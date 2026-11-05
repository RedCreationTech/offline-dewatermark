#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""常见水印专项基线双报告生成工具。

读取 out/常见水印基线_20261104/summary.json 与逐页产物，产出：
- 常见水印基线摘要报告_<日期>_<时分>.html：自包含（图片一律 base64 内嵌，
  最长边 900px JPEG），可单文件外发；
- 常见水印基线明细报告_<日期>_<时分>.html：逐页明细（图片用相对路径，
  属全量逐页明细类豁免，对外只发同轮摘要）。

标题带主题＋日期＋时间（取自运行时刻）。生成后须跑 tools/build_report_index.py。
"""
from __future__ import annotations

import base64
import collections
import io
import json
from datetime import datetime
from html import escape as e
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'out' / '常见水印基线_20261104'
OUT = ROOT / 'out'

STYLE_LABEL = {
    'timestamp': '日期时间戳', 'website': '网址文字', 'logo_transparent': '透明几何 Logo',
    'logo_opaque': '不透明几何 Logo', 'local_text': '局部文字', 'tiny_corner': '高清小角标',
    'signature': '模拟手写签名', 'center_text': '居中大字', 'center_white': '白色半透明大字',
    'center_gray': '灰黑半透明大字', 'tiled_diagonal': '斜向平铺文字',
}

GLOSSARY = {
    '笔画召回率': '预测选区覆盖真实水印笔画像素（alpha>0）的比例。只衡量"找全没有"。',
    '精确率': '预测选区中真正属于水印笔画像素的比例。只衡量"多选没有"。',
    '负例': '确认无水印的对照图片。被改动（哪怕一个像素被修复）就算保护失败。',
    '零召回': '笔画召回率低于 5% 的页面，即水印基本没被找到。',
    'ink_iou': '识别条件：输出与干净参考在水印区域的吻合度，衡量"去完还认得出原内容吗"。',
    '兜底通道': '专用通道都没命中时的最后选区来源（记为 tonal）。证据不足时给出大掩码是当前误伤来源之一。',
}

TIP_JS = """
const tip = document.getElementById('term-tip');
document.querySelectorAll('button.term').forEach(btn => {
  btn.addEventListener('click', ev => {
    ev.stopPropagation();
    if (tip.style.display === 'block' && tip.textContent === btn.dataset.tip) { tip.style.display = 'none'; return; }
    tip.textContent = btn.dataset.tip;
    const r = btn.getBoundingClientRect();
    tip.style.left = Math.min(window.innerWidth - 360, Math.max(8, r.left - 160) + window.scrollX) + 'px';
    tip.style.top = (r.bottom + window.scrollY + 8) + 'px';
    tip.style.display = 'block';
  });
});
document.addEventListener('click', () => tip.style.display = 'none');
document.addEventListener('keydown', ev => { if (ev.key === 'Escape') tip.style.display = 'none'; });
"""


def term_html(text: str) -> str:
    for term, expl in GLOSSARY.items():
        if term in text and f'>{term}<' not in text:
            text = text.replace(term, f'{term}<button type="button" class="term" data-tip="{e(expl, quote=True)}">?</button>')
    return text


def b64(path: Path, max_side: int = 900) -> str:
    img = Image.open(path)
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert('RGB').save(buf, 'JPEG', quality=82)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def style_rows(results):
    agg = collections.defaultdict(list)
    for r in results:
        if r['negative']:
            continue
        for p in r['pages']:
            m = p.get('metrics') or {}
            agg[r['style']].append((m.get('mask_recall'), m.get('mask_precision'),
                                    m.get('ink_iou'), p.get('passed')))
    rows = []
    for s in sorted(agg, key=lambda k: float(np.nanmean([x[0] for x in agg[k]]))):
        v = agg[s]
        rr = float(np.nanmean([x[0] for x in v]))
        pp = float(np.nanmean([x[1] for x in v]))
        ii = [x[2] for x in v if x[2] is not None]
        rows.append((STYLE_LABEL.get(s, s), len(v), rr, pp,
                     float(np.nanmean(ii)) if ii else None,
                     sum(1 for x in v if x[0] is not None and x[0] < 0.05)))
    return rows


def pct(x):
    return '—' if x is None else f'{x * 100:.1f}%'


def main():
    # 标题时间一律取测试当时的时间（summary.json 记录的开始/结束），不是报告生成时间。
    s = json.loads((RUN / 'summary.json').read_text())
    t0 = datetime.fromisoformat(s['created_at'])
    t1 = datetime.fromisoformat(s['completed_at'])
    stamp = f"{t0.strftime('%Y-%m-%d')} 测试 {t0.strftime('%H:%M')}–{t1.strftime('%H:%M')}"
    day = t0.strftime('%Y%m%d')
    hm = t0.strftime('%H%M')
    results = s['results']
    rows = style_rows(results)

    neg_pages = [p for r in results if r['negative'] for p in r['pages']]
    neg_bad = [r['case_id'] for r in results if r['negative'] for p in r['pages'] if not p.get('passed')]
    zero_total = sum(r[5] for r in rows)
    pos_pages = s['positive_pages']

    table = ''.join(
        f'<tr><td>{e(name)}</td><td>{n}</td><td class="num {"bad" if rr<0.3 else "ok"}">{pct(rr)}</td>'
        f'<td class="num">{pct(pp)}</td><td class="num">{pct(ii)}</td><td>{z}</td></tr>'
        for name, n, rr, pp, ii, z in rows)
    style_table = (
        '<table><thead><tr><th>常见水印样式</th><th>页数</th><th>笔画召回率 ?</th>'
        '<th>精确率 ?</th><th>ink_iou ?</th><th>零召回 ?页</th></tr></thead>'
        f'<tbody>{table}</tbody></table>').replace('?', '<button type="button" class="term" data-tip="占位"></button>') if False else (
        f'<table><thead><tr><th>常见水印样式</th><th>页数</th><th>笔画召回率'
        f'<button type="button" class="term" data-tip="{e(GLOSSARY["笔画召回率"], quote=True)}">?</button></th>'
        f'<th>精确率<button type="button" class="term" data-tip="{e(GLOSSARY["精确率"], quote=True)}">?</button></th>'
        f'<th>ink_iou<button type="button" class="term" data-tip="{e(GLOSSARY["ink_iou"], quote=True)}">?</button></th>'
        f'<th>零召回<button type="button" class="term" data-tip="{e(GLOSSARY["零召回"], quote=True)}">?</button>页</th>'
        f'</tr></thead><tbody>{table}</tbody></table>')

    findings = f"""
<ul>
<li><b>整体</b>：{pos_pages} 张正例里 <b>{zero_total} 张零召回</b>——接近一半的页面水印基本没被找到，双极分布与旧全量基线一致。</li>
<li><b>最好</b>：斜向平铺 {pct(rows[-1][2])}、居中大字 {pct(rows[-2][2])}、灰黑半透明大字（新增样式首秀）{pct([r for r in rows if r[0]=="灰黑半透明大字"][0][2])}。</li>
<li><b>最差（明天归因靶子）</b>：{ '、'.join(f'{r[0]} {pct(r[2])}' for r in rows[:4]) }。</li>
<li><b>负例 ?</b>：{s['negative_safe']}/{s['negative_pages']} 安全；未安全：{'、'.join(neg_bad) if neg_bad else '无'}（暗照片被兜底通道改动，已登记，与"漏检后 tonal 乱给掩码"同根）。</li>
<li><b>口径</b>：今天 112 张样本刚经可见度修正，与 11-04 全量基线比较时须注意样本已变更；本报告是专项观察基线，不替换官方全量基线指针。</li>
</ul>"""
    findings = term_html(findings)

    best_bad = []
    for name, n, rr, pp, ii, z in rows[:4]:
        cand = [r for r in results if not r['negative'] and STYLE_LABEL.get(r['style']) == name]
        cand.sort(key=lambda r: (r['pages'][0].get('metrics') or {}).get('mask_recall') or 0)
        for r in cand[:2]:
            p = r['pages'][0]
            figs = ''.join(
                f'<figure><img src="{b64(RUN / p[k])}"><figcaption>{t}</figcaption></figure>'
                for k, t in [('input_preview', '输入'), ('predicted_mask', '选区'), ('output', '输出')]
                if p.get(k))
            m = p.get('metrics') or {}
            best_bad.append(
                f'<div class="case"><h3>{e(r["case_id"])}</h3>'
                f'<p>召回 {pct(m.get("mask_recall"))} ｜ 精确率 {pct(m.get("mask_precision"))} ｜ '
                f'通道 {e(str(p.get("mask_method")))} ｜ {p.get("elapsed_ms", 0)/1000:.1f}s</p>'
                f'<div class="figs">{figs}</div></div>')

    summary_html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>常见水印基线摘要报告 {stamp}</title>
<style>
:root{{--ink:#203047;--muted:#63758a;--accent:#24639f;--border:#dce5ee;--page:#f2f5f8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--page);color:var(--ink);font:15px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1060px;margin:0 auto;padding:32px 24px 70px}}
section{{background:#fff;border:1px solid var(--border);border-radius:13px;padding:26px 30px;margin-bottom:22px}}
h1{{font-size:28px;margin:4px 0 10px}}h2{{font-size:20px;border-left:5px solid var(--accent);padding-left:11px;margin:22px 0 10px}}
.eyebrow{{font-size:12px;color:var(--muted)}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}}th,td{{border:1px solid #e2e8f0;padding:10px 12px;text-align:left}}th{{background:#edf3f8}}
.num{{font-weight:700}}.bad{{color:#b3382e}}.ok{{color:#1e7a46}}
ul{{padding-left:22px}}li{{margin-bottom:7px}}
.case{{border:1px solid var(--border);border-radius:10px;padding:14px 18px;margin:14px 0}}
.case h3{{font-size:14px;margin:0 0 6px}}
.figs{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}figure{{margin:0}}img{{width:100%;height:230px;object-fit:contain;background:#f0f2f4;border:1px solid #e1e6eb;border-radius:4px}}figcaption{{font-size:11px;text-align:center;color:#60788e;margin-top:4px}}
button.term{{border:1px solid #bcccdc;border-radius:50%;width:17px;height:17px;font-size:10px;color:#486581;background:#f0f4f8;cursor:pointer;padding:0;vertical-align:super}}
#term-tip{{position:absolute;background:#243b53;color:#fff;font-size:13px;padding:10px 14px;border-radius:8px;max-width:340px;display:none;z-index:9;line-height:1.6}}
</style></head><body>
<div id="term-tip"></div><main>
<p class="eyebrow">观察基线 · 生成于 {stamp} · 桌面默认单模式 · 168 输入（145 正例＋23 负例）</p>
<h1>常见水印基线摘要报告 {stamp}</h1>
<section><h2>一、按样式结果</h2>{style_table}</section>
<section><h2>二、要点与口径</h2>{findings}</section>
<section><h2>三、最差四类的实例（每类抽样 2 张）</h2>{''.join(best_bad)}</section>
<section><h2>四、产物与追溯</h2>
<ul>
<li>完整逐页明细：同目录《常见水印基线明细报告》（图片为相对路径，本地查看；对外发送以本摘要为准）。</li>
<li>逐页数值与哈希：常见水印基线_20261104/summary.json；报告由 tools/build_common_baseline_report.py 生成，可重跑。</li>
</ul></section>
</main><script>{TIP_JS}</script></body></html>"""

    detail_cards = []
    # 图片相对路径：明细报告在 常见水印基线报告_*/ 下，产物在 常见水印基线_20261104/ 下。
    IMG_PREFIX = '../常见水印基线_20261104/'
    for r in results:
        for p in r['pages']:
            m = p.get('metrics') or {}
            recall = m.get('mask_recall')
            problem = (not r['negative'] and (recall is None or recall < 0.30)) or (r['negative'] and not p.get('passed'))
            figs = ''.join(
                f'<figure><img loading="lazy" src="{IMG_PREFIX}{e(p[k])}"><figcaption>{t}</figcaption></figure>'
                for k, t in [('input_preview', '输入'), ('predicted_mask', '选区'), ('output', '输出'), ('clean_preview', '干净参考')]
                if p.get(k))
            detail_cards.append(
                f'<article data-style="{e(STYLE_LABEL.get(r["style"], "负例"))}" data-search="{e(r["case_id"]).lower()}">'
                f'<h2>{e(r["case_id"])}</h2>'
                f'<p class="{"problem" if problem else "fine"}">{"▲ 问题档 ｜ " if problem else ""}'
                f'{p["status"]} ｜ 召回 {pct(m.get("mask_recall"))} ｜ 精确率 {pct(m.get("mask_precision"))} ｜ '
                f'ink_iou {pct(m.get("ink_iou"))} ｜ 通道 {e(str(p.get("mask_method")))} ｜ '
                f'改动 {p.get("changed_fraction", 0):.4f} ｜ {p.get("elapsed_ms", 0)/1000:.1f}s ｜ passed={p.get("passed")}</p>'
                f'<div class="figs">{figs}</div><p>{e(p.get("error", ""))}</p></article>')
    # 结论区：测试张数、各类型结果、问题最大样式（模版要求结论置顶）
    style_stats = rows  # (名称, 页数, 召回, 精确率, ink_iou, 零召回数)
    def problem_ratio(row):
        name, n, rr, pp, ii, z = row
        if name == '负例':
            return 1 - s['negative_safe'] / max(s['negative_pages'], 1)
        return (z + sum(1 for r in results if not r['negative'] and STYLE_LABEL.get(r['style']) == name
                         for p in r['pages']
                         if (p.get('metrics') or {}).get('mask_recall') is not None
                         and 0.05 <= (p.get('metrics') or {}).get('mask_recall') < 0.30)) / n
    stats_sorted = sorted(style_stats, key=problem_ratio, reverse=True)
    worst_name = stats_sorted[0][0]
    conclusion = (
        f'<div class="conclusion"><h2>结论</h2>'
        f'<p>本次测试 <b>{s["total_inputs"]}</b> 个输入（正例 {s["positive_pages"]} 页＋'
        f'负例 {s["negative_pages"]} 页），桌面默认单模式，'
        f'负例 {s["negative_safe"]}/{s["negative_pages"]} 安全，运行错误 {s["error_pages"]} 页。'
        f'问题档口径：笔画召回 &lt;30% 或负例受损。'
        f'测试时间 {t0.strftime("%Y-%m-%d %H:%M")}–{t1.strftime("%H:%M")}。</p>'
        f'<h3>各类型结果（按问题占比排序）</h3>'
        f'<table><thead><tr><th>水印类型</th><th>页数</th><th>均笔画召回率</th><th>均精确率</th>'
        f'<th>问题档占比</th></tr></thead><tbody>'
        + ''.join(f'<tr><td>{e(name)}</td><td>{n}</td><td class="num">{pct(rr)}</td>'
                   f'<td class="num">{pct(pp)}</td><td class="num">{problem_ratio(row)*100:.0f}%（{int(problem_ratio(row)*n)}/{n}）</td></tr>'
                   for row in stats_sorted for name, n, rr, pp, ii, z in [row])
        + f'</tbody></table>'
        f'<p><b>问题最大的样式：{e(worst_name)}</b>——初步分析：'
        + ('兜底通道在证据不足时仍给出大掩码，且该类字形特征未命中任何专用通道；'
           '具体根因待明天的漏检归因（已登记靶子：不透明 Logo → 透明 Logo → 时间戳 → 白色半透明大字）。'
           if worst_name != '负例' else '暗照片底图被兜底通道改动；与漏检后兜底乱给掩码同根，已登记。')
        + '</p><p class="note">本报告为专项观察基线，不替换官方全量基线指针；'
          '今天 112 张样本刚经可见度修正，与 11-04 全量基线比较时需注意样本已变更。</p></div>')
    tabs = ['全部'] + [e(r[0]) for r in style_stats] + ['负例']
    tabs_html = ''.join(f'<button type="button" class="{"on" if i == 0 else ""}" data-tab="{t}">{t}</button>'
                        for i, t in enumerate(tabs))
    detail_html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>常见水印基线明细报告 {stamp}</title>
<style>
body{{font:15px/1.8 system-ui,"PingFang SC","Microsoft YaHei",sans-serif;background:#f1f4f8;color:#25394e;padding:24px;max-width:1500px;margin:auto}}
article{{background:#fff;border:1px solid #dae4ef;border-radius:10px;padding:22px;margin:18px 0}}
h1{{font-size:24px}}
h2{{font-size:15px;margin:0 0 8px}}
.figs{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}figure{{margin:0}}img{{width:100%;height:280px;object-fit:contain;background:#f3f5f7}}
.tabs{{display:flex;gap:6px;flex-wrap:wrap;margin:14px 0;align-items:center}}
.tabs button{{padding:8px 16px;border:1px solid #d8e1e8;border-radius:8px;background:#e9eef2;cursor:pointer;font:inherit}}
.tabs button.on{{background:#fff;font-weight:700;border-bottom:2.5px solid #2c7a4f}}
#search{{padding:8px 12px;border:1px solid #d8e1e8;border-radius:8px;font:inherit;min-width:240px}}
table{{border-collapse:collapse;font-size:13px;background:#fff;width:100%}}td,th{{border:1px solid #d5dfe8;padding:6px 10px;text-align:left}}
.num{{font-weight:700}}
.conclusion{{background:#fff;border:1px solid #dae4ef;border-radius:10px;padding:22px;margin:18px 0}}
.problem{{color:#b3382e}}
.note{{color:#63758a;font-size:12.5px}}
</style></head><body>
<h1>常见水印基线明细报告 {stamp}</h1>
{conclusion}
<div class="tabs">{tabs_html}<span style="margin-left:auto"><input id="search" type="search" placeholder="搜索样本编号…"></span></div>
{''.join(detail_cards)}
<script>
const tabs = document.querySelectorAll('.tabs button[data-tab]');
const arts = document.querySelectorAll('article[data-style]');
const box = document.getElementById('search');
function apply(){{
  const tab = document.querySelector('.tabs button.on')?.dataset.tab || '全部';
  const q = box.value.trim().toLowerCase();
  arts.forEach(a => {{
    const okTab = tab === '全部' || a.dataset.style === tab;
    const okQ = !q || a.dataset.search.includes(q);
    a.style.display = okTab && okQ ? '' : 'none';
  }});
}}
tabs.forEach(b => b.addEventListener('click', () => {{
  tabs.forEach(x => x.classList.remove('on')); b.classList.add('on'); apply();
}}));
box.addEventListener('input', apply);
</script>
</body></html>"""

    out_dir = OUT / f'常见水印基线报告_{day}'
    out_dir.mkdir(exist_ok=True)
    sum_path = out_dir / f'常见水印基线摘要报告_{day}_{hm}.html'
    det_path = out_dir / f'常见水印基线明细报告_{day}_{hm}.html'
    sum_path.write_text(summary_html, 'utf-8')
    det_path.write_text(detail_html, 'utf-8')
    print(sum_path)
    print(det_path)


if __name__ == '__main__':
    main()
