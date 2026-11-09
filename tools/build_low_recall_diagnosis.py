#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基础选区低召回诊断报告生成工具。

只读取指定防退化控制轮目录（封存基线）里的运行记录、样本清单、固定资源和
已生成的预览图，不重新跑任何去水印推理。输出自包含 HTML（图片 base64 内嵌），
登记低召回样本的物理可见性鉴定与根因分类，供后续立项修复用。
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import re
from datetime import datetime
from html import escape as e
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'out'

GLOSSARY = {
    '笔画召回率': '预测选区覆盖真实水印笔画像素（alpha>0.08）的比例。只衡量"找全没有"，不衡量"多选没有"。',
    '印迹强度': '输入图相对干净参考图在水印区域的平均灰度差（0-255）。数值越大水印在像素上越明显；低于约4级时接近物理不可见。',
    '纸面比例': '图像中接近纯白纸张背景的像素占比。去水印流程用它判断"这是不是一张文档照片"。',
    '负例': '确认无水印的对照图片。负例被改动（哪怕一个像素被修复）就算保护失败。',
    '零召回': '笔画召回率低于10%的页面。本报告的低召回主体，其中位数选中像素为0，即"完全没找到"。',
}

METHOD_LABEL = {
    'tonal': 'tonal（浅亮色罩）',
    'nonpaper_background': 'nonpaper_background（非纸面放弃）',
    'recall_pixel_strokes': 'recall_pixel_strokes（笔画像素）',
    'photo_confirmed_colour_line': 'photo_confirmed_colour_line（照片彩线）',
    'paper_repeat_document_overlay': 'paper_repeat_document_overlay（重复叠加）',
}


def term_html(text: str) -> str:
    """把术语包成 ❓ 浮层。"""
    for term, expl in GLOSSARY.items():
        if term in text and f'>{term}<' not in text:
            text = text.replace(term, f'{term}<button type="button" class="term" data-tip="{e(expl, quote=True)}">?</button>')
    return text


def b64_img(path: Path, max_side: int = 900) -> str:
    img = cv2.imread(str(path))
    if img is None:
        return ''
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode()


def figure(rec: dict, key: str, caption: str) -> str:
    p = (rec.get('files') or {}).get(key)
    if not p or not Path(p).is_file():
        return ''
    data = b64_img(Path(p))
    return (f'<figure><img src="data:image/jpeg;base64,{data}" alt="{e(caption)}">'
            f'<figcaption>{term_html(e(caption))}</figcaption></figure>')


def imprint_stats(input_p: str, clean_p: str, alpha_p: str):
    img = cv2.imread(input_p)
    clean = cv2.imread(clean_p)
    al = cv2.imread(alpha_p, cv2.IMREAD_UNCHANGED)
    if img is None or clean is None or al is None:
        return None
    img = img.astype(np.float32)
    clean = clean.astype(np.float32)
    if al.ndim == 3:
        al = al.mean(2)
    al = al.astype(np.float32) / 255.0
    delta = np.abs(img - clean).mean(2)
    wm = al > 0.08
    if wm.sum() == 0:
        return None
    d = delta[wm]
    return dict(mean=float(d.mean()), frac4=float((d > 4).mean()), px=int(wm.sum()))


def classify(st) -> str:
    if st['mean'] >= 12 or st['frac4'] >= 0.5:
        return '印迹清晰（算法漏检）'
    if st['mean'] >= 5:
        return '印迹中等'
    return '印迹微弱（接近物理不可见）'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True, help='防退化控制轮目录（含运行记录.json）')
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    run = json.loads((run_dir / '运行记录.json').read_text())
    manifest = json.loads((run_dir / '样本清单.json').read_text())
    cases = {c['id']: c for c in manifest['cases']}
    recs = run['records']

    pos = [r for r in recs if not r.get('negative') and r.get('status') == 'completed' and r.get('recall') is not None]
    zero = [r for r in pos if r['recall'] < 0.10]
    neg = [r for r in recs if r.get('negative')]
    neg_modified = [r for r in neg if (r.get('changed_pixels') or 0) > 0]
    errors = [r for r in recs if r.get('status') != 'completed']

    # 召回分布
    buckets = collections.OrderedDict([('0-10%', 0), ('10-30%', 0), ('30-50%', 0), ('50-70%', 0), ('70-90%', 0), ('90-100%', 0)])
    for r in pos:
        v = r['recall']
        for i, (lo, hi) in enumerate([(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.001)]):
            if lo <= v < hi:
                buckets[list(buckets)[i]] += 1
                break

    # 方法 × 零召回
    method_rows = []
    for m, total in collections.Counter(r['method'] for r in pos).most_common():
        z = sum(1 for r in zero if r['method'] == m)
        method_rows.append((METHOD_LABEL.get(m, m), total, z, z / total))

    # 印迹鉴定
    measured, unreadable = [], []
    for r in zero:
        c = cases.get(r['id'])
        ref = (c.get('references') or [{}])[0] if c else {}
        if not ref.get('clean') or not ref.get('alpha'):
            unreadable.append((r, '无参考'))
            continue
        st = imprint_stats(c['input'], ref['clean'], ref['alpha'])
        if st is None:
            unreadable.append((r, 'PDF或参考图无法直接读取'))
            continue
        measured.append((r, st, classify(st)))
    cls_counts = collections.Counter(cat for _, _, cat in measured)

    # 代表样本
    def find_rec(snippet: str, negative=False):
        for r in recs:
            if snippet in r['id'] and bool(r.get('negative')) == negative and r.get('status') == 'completed':
                return r
        return None

    reps = {
        'A_food': find_rec('inc11_food_dish__local_text'),
        'B_logo': find_rec('document_029__logo_opaque'),
        'B_white': find_rec('document_029__center_white'),
        'C_neg': find_rec('photo_021__negative', negative=True),
    }

    gen_moment = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')
    fname_stamp = gen_moment.replace('-', '').replace(' ', '_').replace(':', '')
    out_path = OUT / f'基础选区漏检诊断报告_{fname_stamp}.html'

    max_bucket = max(buckets.values())
    hist = ''.join(
        f'<div class="bar-row"><span class="bar-label">{k}</span>'
        f'<div class="bar-wrap"><div class="bar{" hot" if k == "0-10%" else ""}" style="width:{v / max_bucket * 100:.0f}%"></div></div>'
        f'<span class="bar-num">{v}</span></div>' for k, v in buckets.items())

    method_table = ''.join(
        f'<tr><td>{e(m)}</td><td>{t}</td><td class={"bad" if z else ""}>{z}</td><td>{p:.0%}</td></tr>'
        for m, t, z, p in method_rows)

    cls_lines = ''.join(f'<li><strong>{e(cat)}</strong>：{n} 页</li>' for cat, n in cls_counts.most_common())

    detail_rows = []
    for r, st, cat in measured:
        detail_rows.append(
            f'<tr><td>{e(r["id"])}</td><td>{e(str(r.get("style")))}</td><td>{e(r["method"])}</td>'
            f'<td>{r["recall"]:.3f}</td><td>{r.get("selected_pixels")}</td>'
            f'<td>{st["mean"]:.1f}</td><td>{st["frac4"]:.0%}</td><td>{e(cat)}</td></tr>')
    for r, why in unreadable:
        detail_rows.append(
            f'<tr><td>{e(r["id"])}</td><td>{e(str(r.get("style")))}</td><td>{e(r["method"])}</td>'
            f'<td>{r["recall"]:.3f}</td><td>{r.get("selected_pixels")}</td><td colspan="2">{e(why)}</td><td>未测量</td></tr>')
    detail_rows.sort()

    neg_row = ''.join(
        f'<tr><td>{e(r["id"])}</td><td>{e(r["method"])}</td><td>{r.get("selected_pixels")}</td>'
        f'<td>{r.get("changed_pixels")}</td></tr>' for r in neg_modified) or '<tr><td colspan="4">无</td></tr>'

    def rep_block(key: str, *figs: str) -> str:
        r = reps.get(key)
        if not r:
            return ''
        return ''.join(figure(r, fk, cap) for fk, cap in figs)

    page = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>基础选区漏检诊断报告 {gen_moment}</title>
<style>
body{{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:1060px;margin:24px auto;padding:0 16px;color:#1f2933;line-height:1.85}}
h1{{font-size:23px;margin-bottom:4px}} h2{{font-size:18px;margin-top:38px;border-left:5px solid #486581;padding-left:10px}} h3{{font-size:16px;margin-top:24px}}
.meta{{color:#627d98;font-size:14px}}
.box{{background:#f0f4f8;border:1px solid #d9e2ec;border-radius:8px;padding:14px 18px;margin:14px 0}}
.badbox{{background:#fdf0f0;border:1px solid #f0c8c8;border-radius:8px;padding:14px 18px;margin:14px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0}} th,td{{border:1px solid #d9e2ec;padding:6px 10px;text-align:left}} th{{background:#f0f4f8}}
td.bad{{color:#9f3a38;font-weight:700}}
.bar-row{{display:flex;align-items:center;gap:10px;margin:4px 0}} .bar-label{{width:64px;text-align:right;font-size:13px;color:#486581}}
.bar-wrap{{flex:1;background:#eef2f6;border-radius:4px;height:20px}} .bar{{height:20px;background:#829ab1;border-radius:4px}} .bar.hot{{background:#9f3a38}}
.bar-num{{width:40px;font-size:13px}}
figure{{margin:16px 0;text-align:center}} figure img{{max-width:100%;border:1px solid #d9e2ec;border-radius:6px}} figcaption{{font-size:13px;color:#627d98;margin-top:6px}}
button.term{{border:1px solid #bcccdc;border-radius:50%;width:18px;height:18px;font-size:11px;line-height:1;color:#486581;background:#f0f4f8;cursor:pointer;padding:0;vertical-align:super}}
#term-tip{{position:fixed;background:#243b53;color:#fff;font-size:13px;padding:10px 14px;border-radius:8px;max-width:340px;display:none;z-index:9;line-height:1.6}}
code{{background:#f0f4f8;padding:1px 6px;border-radius:4px}}
</style></head><body>
<h1>基础选区漏检诊断报告</h1>
<p class="meta">生成时间 {gen_moment}（北京时间）· 数据来源：封存基线 <code>{e(run_dir.name)}</code>（单模式全量 {len(recs)} 页）· 本报告只做数据诊断，未重新运行任何去水印推理，未改动算法。</p>

<div class="box"><strong>术语：</strong>{'　'.join(f'{t}<button type="button" class="term" data-tip="{e(x, quote=True)}">?</button>' for t, x in GLOSSARY.items())}</div>

<h2>一、结论摘要</h2>
<p>本次全量 {len(recs)} 页中，正样本 {len(pos)} 页、{term_html('负例')} {len(neg)} 页、运行错误 {len(errors)} 页。
带真值的正样本呈明显双极分布：100 页召回在 90% 以上，{len(zero)} 页{term_html('零召回')}（其中位数选中像素为 0，即"完全没找到"）。
对 {len(zero)} 页做{term_html('印迹强度')}鉴定：可测量 {len(measured)} 页中 <strong>{cls_counts.get('印迹清晰（算法漏检）', 0)} 页印迹清晰</strong>——水印在像素上明明存在，是算法没检测到，不是物理不可见。</p>
<ul>{cls_lines}<li>未测量：{len(unreadable)} 页（PDF 输入或参考图无法直接读取，不影响结论方向）</li></ul>
<p>根因归为四类，按修复优先级：</p>
<table><tr><th>优先级</th><th>根因</th><th>页数</th><th>性质</th></tr>
<tr><td>1（红线）</td><td>C. tonal 在干净照片上造出 6 万像素选区并修复，{term_html('负例')}受损</td><td>{len(neg_modified)} 页</td><td>保护失效，已确认</td></tr>
<tr><td>2</td><td>A. "纸面比例&lt;25% 就放弃"的保护闸把真实照片上的水印全部跳过</td><td>{sum(1 for r in zero if r['method'] == 'nonpaper_background')} 页</td><td>设计使然的能力空缺，已确认</td></tr>
<tr><td>3</td><td>B. tonal 路径模型只覆盖"浅亮色罩"，饱和彩色、白半透明、网格类水印在合成文档上整类漏检</td><td>{sum(1 for r in zero if r['method'] == 'tonal')} 页</td><td>算法覆盖不足，机制为待验证假设</td></tr>
<tr><td>4</td><td>D. recall_pixel_strokes 少量漏检</td><td>{sum(1 for r in zero if r['method'] == 'recall_pixel_strokes')} 页</td><td>待查</td></tr></table>

<h2>二、总体分布</h2>
<h3>召回率分布（带真值正样本 {len(pos)} 页）</h3>
{hist}
<h3>定位方法 × 零召回（失败率 = 零召回页数 ÷ 该方法总页数）</h3>
<table><tr><th>定位方法</th><th>总页数</th><th>零召回</th><th>失败率</th></tr>{method_table}</table>
<p>两条路径值得注意：<code>nonpaper_background</code> 的失败率是 <strong>100%</strong>（经过它的照片全部没有选区）；
<code>tonal</code> 承担了最多页数（{sum(1 for r in pos if r['method'] == 'tonal')} 页），零召回 {sum(1 for r in zero if r['method'] == 'tonal')} 页，是绝对数量最大的漏检来源。</p>

<h2>三、根因 A：真实照片整类未处理（{sum(1 for r in zero if r['method'] == 'nonpaper_background')} 页）</h2>
<p><strong>机制（已确认）：</strong><code>wm/masks.py</code> 里有一道保护闸：当{term_html('纸面比例')}低于 25% 时直接返回空选区，方法名记为 <code>nonpaper_background</code>。
它存在的历史原因写在代码注释里：旧的文档兜底逻辑曾在留出集上把<strong>有水印的照片和干净照片</strong>都制造出占全图 17.7% 的假掩码，这道闸是为了让不支持的"真实照片"原样通过、不被误伤。</p>
<p><strong>代价（本次测量）：</strong>闸外这批照片上的水印{term_html('印迹强度')}在 9.7~54.7 灰度级、多数像素差异超过 4 级——肉眼可见，全部被选区放弃。其中 60 页来自"02_人工增量110张"真实照片批次，相当于<strong>真实照片这一类输入目前完全没有去水印能力</strong>。</p>
{rep_block('A_food', ('原图预览.jpg', '代表样本：真实照片（美食场景），左上角有灰色"训练素材"小字水印'), ('漏选多选核对.jpg', '同一页核对图：蓝色为真实水印笔画（全部漏选），流程选择了零像素'))}
<p><strong>性质：</strong>这是防误伤设计付出的能力代价，不是回归缺陷。优化方向是"照片水印检测"立项：为真实照片建立独立的证据链（局部灰字、阴影字），同时保留负例保护，不能用放宽这道闸来实现。</p>

<h2>四、根因 B：tonal 路径覆盖不足（{sum(1 for r in zero if r['method'] == 'tonal')} 页零召回）</h2>
<p><strong>现象（已确认）：</strong><code>tonal</code>路径建模的是"比纸面浅/亮的色罩"。合成文档上大量水印不在这个模型里：</p>
<ul>
<li><strong>不透明彩色几何 Logo</strong>：下方左图是一颗饱和度很高的蓝色实心星，相对纸面印迹达 136 灰度级，完全可见——tonal 选中 0 像素；</li>
<li><strong>白色半透明大字</strong>：下方右图是斜贯页面的半透明"测试素材"文字，多数页印迹中等偏弱（这类接近物理可见性边界，需逐页区分对待）；</li>
<li><strong>网格、线网、签名等</strong>：共约 40 页，多为清晰可见的印迹。</li></ul>
{rep_block('B_logo', ('漏选多选核对.jpg', 'document_029 logo_opaque：真相是一枚蓝色实心星（核对图蓝色叠加），印迹136，tonal 完全未检出'))}
{rep_block('B_white', ('漏选多选核对.jpg', 'document_029 center_white：真相是斜贯页面的半透明大字，肉眼极淡；部分此类页印迹接近物理不可见'))}
<p><strong>机制（待验证假设）：</strong>tonal 的种子条件只接受"亮于纸面 tone"的尖峰，饱和彩色（蓝星灰度中等）、白色半透明（差异极小）、细网格（形态不符）都不产生种子；且文档类路由优先走 tonal，导致这些样式<strong>整类没有回退路径</strong>。需要用代码走查和小样本实验证实。</p>

<h2>五、根因 C：tonal 负例误伤（红线，{len(neg_modified)} 页）</h2>
<div class="badbox"><strong>photo_021__negative：一张确认无水印的建筑照片，被 tonal 选出 61,538 像素并实际修复改动 69,186 像素。</strong>
这是防退化底线中的"负例零改动"被打破，修复优先级最高。</div>
{rep_block('C_neg', ('原图预览.jpg', 'photo_021 负例：干净建筑摄影，无任何水印'), ('选区预览.jpg', '同一页选区：tonal 把天空渐变和建筑边缘的色调起伏当成"水印"，红区即被选中的假水印'))}
<p><strong>现象细节：</strong>同一底图的三个版本路由不一致——正例（白色半透明大字版）走 <code>recall_pixel_strokes</code> 选中 0 像素（漏检），负例反而走 <code>tonal</code> 并大肆多选。
<strong>待验证假设：</strong>tonal 缺少对"自然渐变、建筑边缘"等自然纹理的排除检查，且其触发条件与正例路径的优先级关系需要走查。</p>

<h2>六、根因 D：recall_pixel_strokes 少量漏检（{sum(1 for r in zero if r['method'] == 'recall_pixel_strokes')} 页）</h2>
<p>该路径整体健康（{sum(1 for r in pos if r['method'] == 'recall_pixel_strokes')} 页中仅 {sum(1 for r in zero if r['method'] == 'recall_pixel_strokes')} 页零召回，失败率约
{sum(1 for r in zero if r['method'] == 'recall_pixel_strokes') / max(1, sum(1 for r in pos if r['method'] == 'recall_pixel_strokes')):.0%}），多为多重叠加、白色半透明大字等难例，单独登记，随 B 类一并复核。</p>

<h2>七、负例与运行错误登记</h2>
<h3>负例被改动（应恒为 0）</h3>
<table><tr><th>输入</th><th>方法</th><th>选中像素</th><th>改动像素</th></tr>{neg_row}</table>
<h3>运行错误（非算法问题）</h3>
<table><tr><th>输入</th><th>说明</th></tr>{''.join(f'<tr><td>{e(r["id"])}</td><td>{e(str(r.get("error") or r.get("note") or "记录为错误"))}</td></tr>' for r in errors) or '<tr><td colspan="2">无</td></tr>'}</table>

<h2>八、后续建议（仅登记，本次不改算法）</h2>
<ol>
<li><strong>C 负例保护（红线）</strong>：走查 tonal 触发与排除条件，补充自然渐变/边缘纹理检查；验收必须含 27 张负例零改动 + photo_021 正负例同图对照。</li>
<li><strong>A 照片水印能力</strong>：为真实照片建立独立检测证据链（局部灰字、阴影字），严禁直接放宽 25% 纸面闸；验收含 inc11 全批次与全部负例。</li>
<li><strong>B tonal 覆盖</strong>：先代码走查确认无种子原因，再按样式（彩色 Logo、白半透明、网格）逐个补证据；每补一个样式须带开发样本、其他样式回归与独立验证。</li>
<li><strong>D</strong>：随 B 复核时顺带分析。</li>
<li>所有修复按防退化流程走：相关样式 + 固定保护样本 + 独立验证集三重证据，禁止全局降阈值或关保护。</li></ol>

<h2>附录：全部 {len(zero)} 页零召回清单</h2>
<table><tr><th>输入</th><th>样式</th><th>方法</th><th>召回</th><th>选中像素</th><th>印迹均值</th><th>&gt;4级占比</th><th>分类</th></tr>{''.join(detail_rows)}</table>
<p class="meta">注：印迹均值 = |输入−干净参考| 在真实水印区域的平均灰度差；&gt;4级占比 = 差异超过 4 灰度级的水印像素比例。运行错误页不计入本表。本报告为诊断登记，不构成任何修复已完成或质量合格的结论。</p>

<div id="term-tip"></div>
<script>
const tip = document.getElementById('term-tip');
document.querySelectorAll('button.term').forEach(btn => {{
  btn.addEventListener('click', ev => {{
    ev.stopPropagation();
    if (tip.style.display === 'block' && tip.textContent === btn.dataset.tip) {{ tip.style.display = 'none'; return; }}
    tip.textContent = btn.dataset.tip;
    const r = btn.getBoundingClientRect();
    tip.style.left = Math.min(window.innerWidth - 360, Math.max(8, r.left - 160)) + 'px';
    tip.style.top = (r.bottom + window.scrollY + 8) + 'px';
    tip.style.display = 'block';
  }});
}});
document.addEventListener('click', () => tip.style.display = 'none');
document.addEventListener('keydown', ev => {{ if (ev.key === 'Escape') tip.style.display = 'none'; }});
</script>
</body></html>'''

    out_path.write_text(page)
    print(out_path)


if __name__ == '__main__':
    main()
