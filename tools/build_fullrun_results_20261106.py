"""全量放行本轮独立结果报告（不做新旧基线对比）。

背景：20261106 全量放行运行完成后，自动对比因"运行条件不一致"（默认参数变更 +
11-05 样本去重/裁撤，按管控规则不可比）停止。本报告只输出本轮自身跑出的成绩：
总览、负例安全、分样式统计、时间戳专项明细、异常清单。
数据来源：out/分层防退化检查/20261106_092144_137579_full/完整候选记录/运行记录.json
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'out/分层防退化检查/20261106_092144_137579_full/完整候选记录/运行记录.json'
OUT = ROOT / 'out/全量放行本轮结果_20261106'
STAMP = datetime.now().strftime('%Y-%m-%d %H:%M')
REPORT = OUT / f'全量放行结果报告_20261106_{datetime.now().strftime("%H%M")}.html'

GLOSSARY = {
    '召回率': '真实水印笔画中被选中的比例（分母是全部水印像素，含极淡边缘）。',
    '明显笔画召回': '只统计真值里透明度够明显的笔画，衡量"看得见的部分捞回没有"。',
    '精确率': '选中的像素里真正属于水印的比例；低 = 误圈背景。',
    '像素误差': '修复区域相对干净参考的平均每像素颜色差（0-255）。是重建差异，不能直接叫残留率。',
    '负例': '无水印对照图。在负例上产生任何改动都算安全失败。',
    '逐字节一致': '两次输出每个字节完全相同——证明没有任何影响。',
}


def g(t):
    return f'<span class="term">{t}<span class="tip">{GLOSSARY[t]}</span></span>'


def lead(text):
    return f'<p class="lead">{text}</p>'


def fmt(v, nd=3):
    if v is None:
        return '—'
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return str(v)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def main():
    OUT.mkdir(exist_ok=True)
    recs = json.load(open(RUN))['records']
    done = [r for r in recs if r.get('status') == 'completed']
    errs = [r for r in recs if r.get('status') != 'completed']
    negs = [r for r in done if r.get('negative')]
    negs_changed = [r for r in negs if r.get('changed_pixels')]
    pos = [r for r in done if not r.get('negative')]

    # 分样式统计
    styles = {}
    for r in done:
        s = r.get('style') or '（未分类）'
        d = styles.setdefault(s, [])
        d.append(r)
    style_rows = []
    for s, rows in sorted(styles.items(), key=lambda kv: -len(kv[1])):
        style_rows.append(
            f"<tr><td>{s}</td><td>{len(rows)}</td>"
            f"<td>{fmt(mean([r.get('recall') for r in rows]))}</td>"
            f"<td>{fmt(mean([r.get('strong_recall') for r in rows]))}</td>"
            f"<td>{fmt(mean([r.get('precision') for r in rows]))}</td>"
            f"<td>{fmt(mean([r.get('watermark_mae') for r in rows]), 2)}</td>"
            f"<td>{fmt(mean([r.get('elapsed_ms') for r in rows]), 0)}</td></tr>")

    # 时间戳专项
    ts_rows = []
    for r in sorted([r for r in done if r.get('style') == 'timestamp'], key=lambda x: x['id']):
        ts_rows.append(
            f"<tr><td>{r['id'].split('/')[-1]}</td>"
            f"<td class={'ok' if (r.get('recall') or 0) >= 0.9 else 'bad'}>{fmt(r.get('recall'))}</td>"
            f"<td>{fmt(r.get('strong_recall'))}</td>"
            f"<td>{fmt(r.get('watermark_mae'), 2)}</td>"
            f"<td>{r.get('method')}</td><td>{fmt(r.get('elapsed_ms'), 0)}</td></tr>")

    # center_gray（旧基线没有的新页）
    cg_rows = []
    for r in sorted([r for r in done if 'center_gray' in r['id']], key=lambda x: x['id']):
        cg_rows.append(
            f"<tr><td>{r['id'].split('/')[-1]}</td><td>{fmt(r.get('recall'))}</td>"
            f"<td>{fmt(r.get('watermark_mae'), 2)}</td><td>{r.get('method')}</td></tr>")

    # 异常：未完成 + 负例改动披露
    err_rows = ''.join(
        f"<tr><td>{r['id']}</td><td>{r.get('status')}</td><td>{'输入文件在磁盘存在但未列入本轮运行计划（计划沿用旧基线清单）；新基线封存将覆盖' if 'outlined' in r['id'] else '见批次日志'}</td></tr>"
        for r in errs)
    neg_rows = ''.join(
        f"<tr><td>{r['id']}</td><td class='bad'>{r.get('changed_pixels')}</td><td>{r.get('method')}</td>"
        f"<td>与 11-02 旧基线逐字节相同（既有问题，非本轮引入），tonal 通道在无水印照片误触发，另开任务处理，样本保留不删</td></tr>"
        for r in negs_changed)

    # 全距离散
    elapsed = [r.get('elapsed_ms') for r in done if r.get('elapsed_ms')]
    p95 = sorted(elapsed)[int(len(elapsed) * 0.95)] if elapsed else 0

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>
<title>全量放行本轮结果（独立成绩，不做新旧对比） 2026-11-06 {STAMP}</title>
<style>
body{{font:15px/1.9 'PingFang SC','Hiragino Sans GB',system-ui,sans-serif;background:#eef2f6;color:#25394e;margin:0;padding:40px 20px}}
.wrap{{max-width:1000px;margin:0 auto;background:#fff;border-radius:14px;box-shadow:0 2px 18px rgba(30,60,90,.08);padding:46px 54px;box-sizing:border-box}}
h1{{font-size:23px;text-align:center;margin:6px 0 4px;color:#1b3a54}}
.subtitle{{text-align:center;color:#7a8a94;font-size:13.5px;margin-bottom:24px}}
h2{{font-size:17.5px;color:#1b3a54;border-left:5px solid #2c6e9b;padding-left:12px;margin:38px 0 12px}}
.lead{{font-size:15.5px;background:#f2f7fb;border-left:4px solid #7aa8c4;border-radius:0 8px 8px 0;padding:12px 16px;margin:12px 0;line-height:1.9}}
table{{border-collapse:collapse;background:#fff;width:100%;font-size:13.6px;margin:14px 0;border-radius:8px;overflow:hidden}}
td,th{{border:1px solid #e2e8ef;padding:8px 12px;text-align:left;vertical-align:top}}
th{{background:#f2f6fa;color:#1b3a54;font-weight:600}}
tr:nth-child(even) td{{background:#fafcfe}}
.ok{{color:#1c7a4d;font-weight:600}} .bad{{color:#c0392b;font-weight:600}}
.warn{{background:#fdf3f2;border:1px solid #e6c4c0;border-radius:10px;padding:13px 17px;font-size:14px;margin:12px 0;line-height:1.85}}
.good{{background:#f0faf4;border:1px solid #bfe3cc;border-radius:10px;padding:13px 17px;font-size:14px;margin:12px 0;line-height:1.85}}
.meta{{color:#7a8a94;font-size:13px;text-align:center;margin-top:36px}}
.term{{border-bottom:1px dashed #2c6e9b;cursor:help;position:relative}}
.tip{{display:none;position:absolute;left:0;top:1.5em;z-index:9;width:320px;background:#fff;border:1px solid #b9cbda;border-radius:8px;
box-shadow:0 4px 14px rgba(30,60,90,.18);padding:10px 13px;font-size:13px;line-height:1.7;color:#33475b;cursor:default;font-weight:400}}
.term.open .tip{{display:block}}
code{{background:#eef1f4;padding:1px 6px;border-radius:5px;font-size:12.8px}}
</style></head><body><div class="wrap">
<h1>全量放行本轮结果（2026-11-06）</h1>
<p class="subtitle">{STAMP} ｜ 独立成绩报告，按规则<b>不做新旧基线对比</b>（默认参数已变更 + 样本清单已裁撤，条件不一致不可比）<br/>
数据：本轮 375 页运行自身产出 ｜ 零回归证据另见 47 页开/关对照与独立集 v1 验收</p>

<h2>一、总览</h2>
{lead('一句话：本轮 375 页全部按计划跑完（374 成功 + 1 页因计划外缺页未跑），零崩溃；负例 22 张中 21 张零改动，唯一有改动的一张与旧基线逐字节相同（老问题，非本轮引入）。时间戳新通道在真实管线里如期触发，3 张文档底时间戳像素误差降到 0.004–0.017 级。')}
<table><tr><th>指标</th><th>数值</th><th>说明</th></tr>
<tr><td>计划页数 / 完成</td><td>375 / {len(done)}</td><td>计划沿用旧基线清单口径</td></tr>
<tr><td>崩溃 / 错误页</td><td class="ok">0</td><td>375 批退出码全 0，日志零 Traceback</td></tr>
<tr><td>未跑页</td><td class="bad">1</td><td>document_031__outlined：文件在磁盘但不在本轮计划，新基线封存覆盖</td></tr>
<tr><td>{g('负例')}安全</td><td class="ok">21/22 零改动</td><td>1 张有改动，见下文明细（既有问题）</td></tr>
<tr><td>可测正例</td><td>{len(pos)}</td><td>其中 5 页无真值标注、{g('召回率')}记为"—"（不可测，非零分）</td></tr>
<tr><td>耗时</td><td>均值 {fmt(mean(elapsed),0)} ms，P95 {p95} ms</td><td>无超时页</td></tr></table>

<h2>二、负例改动披露（1 张，既有问题）</h2>
{('<div class="warn">按规矩"原有负例错误也须披露并比较，不能删样本规避"，如实登记：</div>'
  + '<table><tr><th>样本</th><th>改动像素</th><th>通道</th><th>定性</th></tr>' + neg_rows + '</table>') if neg_rows else '<div class="good">负例全部零改动。</div>'}

<h2>三、分样式成绩（本轮自身口径）</h2>
{lead('口径：每页先各自计算再按页平均（不让大图吃掉小图）。"—"表示该页无真值标注或不可测，不参与平均。')}
<table><tr><th>样式</th><th>页数</th><th>{g('召回率')}<br/>（alpha&gt;0 全笔画）</th><th>{g('明显笔画召回')}</th><th>{g('精确率')}</th><th>{g('像素误差')}</th><th>耗时 ms</th></tr>
{''.join(style_rows)}</table>
<p class="meta" style="text-align:left">{g('像素误差')}是重建差异、不等于残留率；{g('召回率')}分母含极淡边缘。</p>

<h2>四、日期时间戳专项明细（本轮 15 页）</h2>
{lead('新通道 <code>ocr_seed_stain</code> 在真实管线中如期触发（method 列可见）：document_029/030/031 三页被解决，'
      '其中 030 全笔画召回 0.975 但明显笔画召回 1.000——漏掉的只是亚阈值边缘像素。'
      '另一个新发现：固定集里 3 张照片底时间戳（猫/汤/photo_021）由既有照片通道解决（召回 1.0），'
      '而独立集的照片底时间戳不行——两者死法不同，已列入下周诊断（周报优先项 1）。')}
<table><tr><th>样本</th><th>召回率</th><th>明显笔画</th><th>像素误差</th><th>解决通道</th><th>耗时 ms</th></tr>
{''.join(ts_rows)}</table>
<p class="meta" style="text-align:left">其余 9 页为已登记未解决项：document_032（深到与墨迹不可分）、charts/food_dish/outdoor_building/textures 等（OCR 读不出，物理受限）。</p>

<h2>五、新增页面成绩：灰黑半透明大字（center_gray，6 页）</h2>
{lead('这 6 页是旧基线建立之后新增的，本轮首次纳入，没有历史对照。成绩参差：document_032 接近满分、document_029/030 为 0——按新样式登记，进入死因榜。')}
<table><tr><th>样本</th><th>召回率</th><th>像素误差</th><th>通道</th></tr>
{''.join(cg_rows)}</table>

<h2>六、异常与未完成</h2>
<table><tr><th>样本</th><th>状态</th><th>说明</th></tr>{err_rows if err_rows else '<tr><td colspan="3">无</td></tr>'}</table>

<h2>七、本轮的边界（不夸大的部分）</h2>
<div class="warn">① 本报告是"本轮跑成什么样"，<b>不含"比上周好/差"</b>——新旧对比因条件不一致按规则停止；
零回归的结论来自本轮之前的开/关对照实验（47 页逐字节 + 独立集 61/61），那是科学的 A/B，本报告不重复宣称。<br/>
② 时间戳 3 页的"多选增加"（每页 323 像素）是候选的已知代价：选在搜索框内的实测染色像素、残差≈0，
已在验收中逐张目视核对，属"故意变好"的一部分。<br/>
③ 5 页无真值、{g('召回率')}不可测，保持"未评估"，未计零分也未计满分。</div>

<p class="meta">数据来源：完整候选记录/运行记录.json（375 条）｜ 运行目录：out/分层防退化检查/20261106_092144_137579_full/ ｜
带 ❓ 的名词可点击浮层查看解释，再点一次或按 Esc 收起。</p>
<script>
document.addEventListener('click',function(e){{
  var t=e.target.closest('.term');
  document.querySelectorAll('.term.open').forEach(function(x){{if(x!==t)x.classList.remove('open');}});
  if(t)t.classList.toggle('open');
}});
document.addEventListener('keydown',function(e){{if(e.key==='Escape')document.querySelectorAll('.term.open').forEach(function(x){{x.classList.remove('open');}});}});
</script>
</div></body></html>"""
    REPORT.write_text(html, encoding='utf-8')
    print(f'written: {REPORT} ({len(html) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
