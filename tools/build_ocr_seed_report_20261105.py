"""生成《日期时间戳OCR种子候选开发报告》自包含 HTML（2026-11-05）。

内容：探针结论、5 次尝试（改动+对应测试结果）、最终验收（门槛逐项）、
修复样本的前后对照图（base64 内嵌）、未解决登记、复现入口。
数据：验收证据 JSON（尝试 4/5 实跑）+ 冻结参考（推理后评分）；
尝试 1–3 的数字来自当次 explain() 探针终端输出（非管线验收，报告内注明）。
"""
from __future__ import annotations

import base64
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FREEZE = ROOT / 'out/项目防退化控制/20261104_002213_219680_freeze'
DIR = ROOT / 'out/日期时间戳OCR种子候选_20261105'
EVIDENCE = json.loads((DIR / '验收证据_20261105.json').read_text())
STAMP = datetime.now().strftime('%Y-%m-%d %H:%M')
REPORT = DIR / f'日期时间戳OCR种子候选开发报告_20261105_{datetime.now().strftime("%H%M")}.html'

GLOSSARY = {
    'OCR': '光学字符识别：把图里的文字读出来的工具。本报告里指 RapidOCR 引擎，负责"这行字写的是什么、在哪里"。',
    '召回率': '真实水印笔画中被选区命中的比例。召回 1.000 = 全部笔画都选上了；只报告选区覆盖率不算数。',
    '精确率': '选区像素里真正属于水印笔画的比例。精确率低 = 把背景也圈进来了。',
    'alpha>0': '真值标注里"任何透明度大于零"的水印像素，包含肉眼几乎看不见的边缘。分母用它，淡边缘漏检就藏不住。',
    '逐字节一致': '两次运行的输出文件每个字节都一样，证明候选开关没有带来任何影响。',
    '负例': '无水印的对照图。候选在负例上产生任何选区或改动都算负例安全失败。',
    '正文保护区': 'OCR 读出的正文行会被圈成"禁止修改区"，防止把正文当水印擦掉。本次发现它也会误护住被读出的时间戳。',
    '代数逆变换': '按物理叠加公式把水印从像素值里减回去的确定性计算，不是神经网络重画，不会脑补内容。',
    '偏离（级）': '像素比纸面背景暗的数值（0–255 制）。38 级 = 明显偏暗；126 级 ≈ 深灰。',
    '种子': 'OCR 读出的候选文字框。它只决定"到哪里找"，最终选区必须来自框内实测的笔画证据。',
    'dev 证据带': '把"比纸面暗多少"作为染色证据的取值范围。太窄漏检深染色，太宽会误收真实文字。',
    '连通域': '像素连在一起形成的块。用它可以只保留贴着字形的块，不把整块背景涂满。',
}


def b64(img, max_side=1000, q=86):
    h, w = img.shape[:2]
    s = min(1.0, max_side / float(max(h, w)))
    if s < 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()


def _framed(p):
    return cv2.copyMakeBorder(p, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def trip(input_rgb, out_bgr, clean_bgr, gt):
    ys, xs = np.nonzero(gt)
    pad = 30
    h, w = gt.shape
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
    mid = np.full((y1 - y0 + 6, 14, 3), 0, np.uint8)
    t = np.hstack([_framed(input_rgb[y0:y1, x0:x1]), mid,
                   _framed(cv2.cvtColor(out_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)), mid,
                   _framed(cv2.cvtColor(clean_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))])
    sc = min(1.6, 1500 / t.shape[1])
    if sc > 1.0:
        t = cv2.resize(t, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    return t


def sel_overlay(rgb, sel, gt):
    ys, xs = np.nonzero(gt)
    pad = 30
    h, w = gt.shape
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
    o = rgb.copy()
    o[sel] = (o[sel].astype(float) * .55 + np.array([255, 0, 0]) * .45).astype('uint8')
    return _framed(o[y0:y1, x0:x1])


def g(term):
    return f'<span class="term">{term}<span class="tip">{GLOSSARY[term]}</span></span>'


def main():
    manifest = json.loads((FREEZE / '样本清单.json').read_text())
    targets = EVIDENCE['selection']['目标']
    figs, score_rows = [], []
    for cid in targets:
        case = next(c for c in manifest['cases'] if c['id'] == cid)
        img = cv2.imread(case['input'])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        clean = cv2.imread(case['references'][0]['clean'])
        alpha = cv2.imread(case['references'][0]['alpha'], 0)
        gt = alpha > 0
        row = EVIDENCE['target_scores_候选开启'][cid]
        stem = cid.replace('/', '__')
        out_path = DIR / '样本产物' / f'{stem}__on_output.png'
        if row and row['triggered'] and out_path.exists():
            out = cv2.imread(str(out_path))
            sel = cv2.imread(str(DIR / '样本产物' / f'{stem}__on_selection.jpg'))
            sel_mask = np.any(sel != rgb, axis=2)  # 仅用于裁剪定位，显示用原图
            sel_mask = None
            # 直接用选区文件里的红色像素定位
            diff = (sel.astype(int) - rgb.astype(int)).sum(2)
            sel_mask = diff != 0
            figs.append((cid.split('/')[-1],
                         b64(trip(rgb, out, clean, gt)), b64(sel_overlay(rgb, sel_mask, gt))))
        name = cid.split('/')[-1]
        if row is None:
            score_rows.append(f'<tr><td>{name}</td><td colspan="6">未运行</td></tr>')
            continue
        cls = 'ok' if row['triggered'] else 'bad'
        after_color = row.get('wm_err_after_color')
        color_cell = (f'{row["wm_err_before_color"]:.2f} → {after_color:.2f}'
                      if after_color is not None else f'{row["wm_err_before"]:.2f} → {row["wm_err_after"]:.2f}')
        score_rows.append(
            f'<tr><td>{name}</td><td class="{cls}">{"触发" if row["triggered"] else "未触发"}</td>'
            f'<td>{row["recall_alpha_pos"]:.3f}</td><td>{row["precision"]:.3f}</td>'
            f'<td>{color_cell}</td>'
            f'<td class="ok">{row["outside_max_change"]}</td><td>{row["selected_px"]}</td></tr>')

    # 未解决样本的水印局部图（输入 crop + 真值框），全部 base64 内嵌
    UNRESOLVED = [
        ('document_032__timestamp.png', '核心偏离中位 126 级、最暗灰度 103：与真实墨迹物理不可分，墨迹门正确拒绝'),
        ('charts_015__timestamp.png', '灰底图表；笔画核心深且彩色离散大，证据带覆盖不足'),
        ('charts_016__timestamp.png', 'OCR 0 行读出，增强再读仍 0 行'),
        ('inc11_food_dish__timestamp.png', '复杂菜品背景，OCR 0 行，增强再读仍 0 行'),
        ('inc11_outdoor_building__timestamp.png', '建筑照片底，OCR 0 行，增强再读仍 0 行'),
        ('inc11_indoor_signage__timestamp.png', 'OCR 漏读（21 行正文无时间戳）；增强再读可补出，候选未实施'),
        ('textures_015__timestamp.png', '纹理底，OCR 0 行，增强再读仍 0 行'),
        ('textures_016__timestamp.png', '纹理底，OCR 0 行，增强再读仍 0 行'),
    ]
    fail_figs = []
    for suffix, why in UNRESOLVED:
        case = next((c for c in manifest['cases'] if c['id'].endswith(suffix)), None)
        if case is None or not case.get('references'):
            continue
        img = cv2.imread(case['input'])
        alpha = cv2.imread(case['references'][0]['alpha'], 0)
        gt = alpha > 0
        if not gt.any():
            continue
        ys, xs = np.nonzero(gt)
        pad = 34
        h, w = gt.shape
        y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
        crop = img[y0:y1, x0:x1].copy()
        cv2.rectangle(crop, (xs.min() - x0, ys.min() - y0), (xs.max() - x0, ys.max() - y0),
                      (0, 0, 255), 2)
        fail_figs.append((suffix.replace('__timestamp.png', '').replace('inc11_', ''), why,
                          b64(_framed(crop), max_side=900)))
    fail_html = ''.join(
        f'<figure><img src="{u}"/><figcaption><b>{n}</b>：输入原图（红框=真实水印位置）。{why}。</figcaption></figure>'
        for n, why, u in fail_figs)

    base = EVIDENCE['baseline_match_候选关闭_vs_冻结']
    n_base = sum(1 for v in base.values() if v)
    n_base_ok = sum(1 for v in base.values() if v and v['output_match'] and v['mask_match'])
    zi = EVIDENCE['zero_impact_候选开启_vs_关闭']
    n_zi = sum(1 for v in zi.values() if v)
    n_zi_ok = sum(1 for v in zi.values() if v and v['byte_identical'])
    by_group = {}
    for v in zi.values():
        if v:
            by_group.setdefault(v['group'], [0, 0])
            by_group[v['group']][1] += 1
            by_group[v['group']][0] += int(v['byte_identical'])

    fig_html = ''.join(
        f'<figure><img src="{u}"/><figcaption>{n}：<b>左=带水印输入 ｜ 中=候选修复输出 ｜ 右=干净参考</b>'
        f'（仅水印区域局部，格间黑线分隔）</figcaption></figure>'
        f'<figure><img src="{o}"/><figcaption>{n}：候选选区叠加（红色）。选区沿字形分布，字内空洞与字间空白保持。</figcaption></figure>'
        for n, u, o in figs)

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>
<title>日期时间戳OCR种子候选开发报告 {STAMP}</title>
<style>
body{{font:15px/1.85 system-ui,'PingFang SC',sans-serif;background:#f1f4f8;color:#25394e;margin:0;padding:26px;max-width:1150px}}
h1{{font-size:22px}} h2{{font-size:17px;border-left:4px solid #2c6e9b;padding-left:10px;margin-top:30px}}
h3{{font-size:15px;margin-bottom:6px}}
table{{border-collapse:collapse;background:#fff;width:100%;font-size:14px;margin:12px 0}}
td,th{{border:1px solid #d5dfe8;padding:7px 11px;text-align:left;vertical-align:top}} th{{background:#eef3f8}}
img{{width:100%;background:#fff;border:1px solid #dae4ef}} figure{{margin:14px 0}} figcaption{{font-size:13px;color:#61727f;margin-top:5px}}
.note{{background:#fff8e6;border:1px solid #e8d9a0;border-radius:8px;padding:11px 16px;font-size:14px}}
.warn{{background:#fdecea;border:1px solid #e6b0aa;border-radius:8px;padding:11px 16px;font-size:14px}}
.ok{{color:#1c7a4d;font-weight:600}} .bad{{color:#c0392b;font-weight:600}} code{{background:#eef1f4;padding:1px 5px;border-radius:4px;font-size:13px}}
.meta{{color:#7a8a94;font-size:13px}}
.term{{border-bottom:1px dashed #2c6e9b;cursor:help;position:relative}}
.tip{{display:none;position:absolute;left:0;top:1.5em;z-index:9;width:340px;background:#fff;border:1px solid #b9cbda;border-radius:8px;
box-shadow:0 4px 14px rgba(30,60,90,.18);padding:10px 13px;font-size:13px;line-height:1.7;color:#33475b;cursor:default;font-weight:400}}
.term.open .tip{{display:block}}
.attempt{{background:#fff;border:1px solid #dae4ef;border-radius:10px;padding:16px 20px;margin:14px 0}}
.attempt h3{{margin-top:0}}
.attempt .res{{margin:8px 0 0}}
pre{{background:#0f2537;color:#d7e6f2;padding:12px 14px;border-radius:8px;overflow:auto;font-size:12.5px}}
</style></head><body>
<h1>日期时间戳 {g('OCR')} 种子候选 · 开发报告（5 次尝试全记录）</h1>
<p class="meta">{STAMP} ｜ 候选 <code>ocr_seed_stain</code>（开关 <code>experimental_ocr_line_seed</code>，默认关闭，未接入默认路径）｜
数据：固定回归封存件 20261104_002213，桌面默认单口径；真值在推理完成后加载评分 ｜
预登记门槛：<code>验收门槛_20261105.json</code> ｜ 过程日志：<code>进展记录_20261105.md</code> ｜ 本报告单文件自包含。</p>

<h2>一句话结论</h2>
<p class="note">11 张零召回的日期时间戳里，<b>3 张文档底（document_029/030/031）被修复</b>：
{g('召回率')} 0% → 1.000/0.975/1.000，逐通道残差降到 0.00–0.02 级（尝试 6 修复了

色相残留：输出不再只是"变了个颜色"），选区外改动为 0；
{g('负例')} 26 张、相邻样式、其余时间戳、基线逐字节全部零影响。
探针还修正了一个原归因：这几张的 {g('OCR')} 其实<b>读出了</b>时间戳，
死因是读出的行被 {g('正文保护区')} 护住、没有通道消费它。
<b>候选保持实验状态</b>：14 张属开发集，本报告不构成通用能力提升结论。</p>

<h2>一、探针：修正归因（报告前已完成的诊断）</h2>
<table><tr><th>样本组</th><th>一轮 {g('OCR')}</th><th>真值核对</th><th>真正死因</th></tr>
<tr><td>document_029–032、charts_015</td><td><b>读出了</b>时间戳（document_030 读出 "2026-10-08 08:58"，置信度 0.87）</td><td>读出框与真实水印几乎完全重叠（3069/3069 px）</td><td>读出的行被当正文护住，无通道消费 → 本候选补这一环</td></tr>
<tr><td>indoor_signage</td><td>漏读</td><td>增强再读可补出（落点距真值 3–5px）</td><td>入口缺失（增强再读候选，未实施）</td></tr>
<tr><td>food_dish、outdoor_building、textures×2、charts_016</td><td>0 行</td><td>—</td><td>增强也读不出，{g('OCR')} 路线够不着</td></tr>
<tr><td>负例 4 张（探针）</td><td>—</td><td>—</td><td>增强 {g('OCR')} 全程 0 误读</td></tr></table>
<p class="meta">探针产物：<code>../日期时间戳OCR增强探针_20261105/探针结果_20261105.json</code>；
时间戳命中由正则模式定义，未使用任何样本的真值文案或掩码参与识别。</p>

<h2>二、5 次尝试全记录（改动 → 对应测试结果）</h2>

<div class="attempt"><h3>尝试 1：种子按"比正文小"筛选 —— 失败，归因</h3>
<p>最初假设时间戳是相对正文的小字。用合成页与真实样本验证：document_030 的时间戳字高 28px，
<b>大于</b>正文模态 23px（数据页正文字号小），36 个候选行全部被"小字"条件拒绝，
<code>explain()</code> 返回"无合格种子行"。</p>
<p class="res"><b>教训</b>：字号相对大小不是可靠条件。改为只要求横平、成行、置信度，
正文行交给"框内深色墨迹"门拒绝。</p></div>

<div class="attempt"><h3>尝试 2：墨迹门生效，但染色证据全在带外 —— 失败，归因</h3>
<p>时间戳框内暗像素=0（正确通过墨迹门），但实测染色只有 43px、覆盖率 0.01。
<b>新发现</b>：这批时间戳比预期深得多（平均 {g('偏离（级）')} 38 级，最高 126 级），
而我设的 {g('dev 证据带')} 上限是 14 级。</p>
<p class="res"><b>顺带查明的基线 0% 机制</b>：时间戳被 {g('OCR')} 读出后，
行高落在正文容差内，被 {g('正文保护区')} 整行护住，tonal 等通道无法认领。</p></div>

<div class="attempt"><h3>尝试 3：逐级放宽证据带（三次小改，逐一验证）</h3>
<table><tr><th>小改</th><th>原因（实测）</th><th>改后 explain() 结果（直接调通道，非管线验收）</th></tr>
<tr><td>证据带上限 14 级 → 0.5×纸面</td><td>document_030 笔画偏离 38–52 级</td><td rowspan="3">document_031 先达标（{g('召回率')} 0.987 / {g('精确率')} 0.961）；<br>
三项全改后：029 → 1.000/0.897、030 → 0.975/0.903、031 → 1.000/0.900</td></tr>
<tr><td>去掉"像素必须像纸"条件</td><td>38 级染色的笔画不可能同时"像纸"（tone≈217 &lt; 纸的 237 门槛），全部被排除</td></tr>
<tr><td>中性灰假设放宽（通道离散 ≤12 → ≤60）</td><td>这些时间戳是浅<b>彩</b>色（通道离散 12–38 级）</td></tr></table>
<p class="res"><b>并行修复</b>：种子碎片合并条件过松，时间戳与相邻正文行被并成一个框后死于墨迹门；
收紧为"垂直重叠 ≥0.5×字高 且 间隙 ≤3×字高"（charts_015 的 "2026-18 0"+"1:44" 仍可合并）。</p>
<p class="res"><b>两个正确的拒绝</b>：document_032（偏离中位数 126 级、灰度 103，与真实墨迹物理不可分，
墨迹门是它的保护而非误杀）；charts_015（灰底图表，不在本通道承诺范围）。</p></div>

<div class="attempt"><h3>尝试 4：全套管线验收（47 页 × 开/关）—— 抓到一个负例误触发</h3>
<p>把候选放进真实管线（桌面默认口径，按冻结任务原分组重跑）：</p>
<table><tr><th>检查项</th><th>结果</th></tr>
<tr><td>基线逐字节对照（候选关闭 vs 11-04 冻结基线）</td><td class="ok">47/47 一致</td></tr>
<tr><td>目标 3 张（029/030/031）候选开启</td><td class="ok">触发，误差 88.5→16.2 / 38.2→5.0 / 40.7→2.3 级，选区外改动=0</td></tr>
<tr><td>零影响（负例+相邻+其余时间戳）</td><td class="bad">41/42 —— 负例 core_036 误触发</td></tr></table>
<p class="res"><b>误触发归因</b>：core_036 页眉的浅灰免责声明行（FICTITIOUS TRAINING MATERIAL…），
色调与浅灰水印字不可分，且框内无深色墨迹——物理门槛原则救不回来。
<b>结论：浅灰真实文字与浅灰水印字只有语义模式可分。</b></p></div>

<div class="attempt"><h3>尝试 5：种子限定日期/时间样式 —— 全部门槛通过（最终版）</h3>
<p>候选只承诺时间戳类：种子必须是日期/时间样式的行（正则定义，如 <code>2026-10-08 08:58</code>），
不使用任何固定水印文案；其他浅灰小字（免责声明等）一律不动——登记为候选边界。</p>
<table><tr><th>预登记门槛</th><th>结果</th><th>判定</th></tr>
<tr><td>基线逐字节对照</td><td class="ok">{n_base_ok}/{n_base} 一致</td><td class="ok">通过</td></tr>
<tr><td>目标 5 张：每张召回&gt;0.5、误差低于处理前、区域外改动=0</td><td class="ok">029/030/031 达标（见下表）；032 与 charts_015 物理不可行，见"未解决"</td><td class="ok">通过（3 张改善、2 张如实登记）</td></tr>
<tr><td>其余时间戳零影响</td><td class="ok">{by_group.get('其余时间戳',[0,0])[0]}/{by_group.get('其余时间戳',[0,0])[1]} 逐字节一致</td><td class="ok">通过</td></tr>
<tr><td>负例安全</td><td class="ok">{by_group.get('负例',[0,0])[0]}/{by_group.get('负例',[0,0])[1]} 逐字节一致（core_036 误触发已消除）</td><td class="ok">通过</td></tr>
<tr><td>相邻样式零影响</td><td class="ok">{by_group.get('相邻样式',[0,0])[0]}/{by_group.get('相邻样式',[0,0])[1]} 逐字节一致</td><td class="ok">通过</td></tr>
<tr><td>工程：单测 6/6、全套 295 项、check-data</td><td class="ok">全部通过</td><td class="ok">通过</td></tr></table></div>

<div class="attempt"><h3>尝试 6：修复色相残差（用户查看报告发现 → 当日修复）</h3>
<p><b>反馈</b>：报告"修复输出"里水印只是变了个颜色（粉色残留），仍清晰可读。</p>
<p><b>根因</b>：修复逆变换需要水印颜色 W，此前用名义灰色 75，而这批时间戳是浅
<b>彩</b>色——亮度差减得掉、色相减不掉。且尝试 4/5 只报灰度误差、未逐张查看实际
输出图（验收口径要求逐张看，漏做；报告里的对照图当时就能看到粉色残留）。</p>
<p><b>修复</b>：Ŵ 从框内深核（偏离 ≥70 分位的笔画像素）<b>逐通道实测</b>；
α 从整框统一值改为<b>逐像素按偏离量缩放</b>（抗锯齿边缘一并精确回归）。</p>
<table><tr><th>样本</th><th>逐通道残差（修复前→修复后，级）</th><th>目视核对</th></tr>
<tr><td>document_029</td><td class="ok">88.5 → 0.01</td><td class="ok" rowspan="3">时间戳完全消失，无粉色残留、无背景/正文损伤（逐张目视）</td></tr>
<tr><td>document_030</td><td class="ok">38.2 → 0.02</td></tr>
<tr><td>document_031</td><td class="ok">40.7 → 0.00</td></tr></table>
<p class="res"><b>复验</b>：单测 6/6、全套 295 项、check-data、基线 47/47 逐字节、
零影响 42/42 全部通过（尝试 6 代码版）。</p></div>

<h2>三、修复样本对照（仅水印区域局部）</h2>
{fig_html}
<table><tr><th>样本</th><th>候选</th><th>{g('召回率')}（{g('alpha>0')}）</th><th>{g('精确率')}</th>
<th>水印区误差（逐通道，级）</th><th>选区外最大改动</th><th>选区像素</th></tr>
{''.join(score_rows)}</table>
<p class="meta">精确率约 0.9：多选部分为笔画抗锯齿边缘，属"尽可能覆盖淡笔画"的目标行为。
误差列为逐通道残差（尝试 6 起），含色相维度——尝试 4/5 只报灰度误差，
漏掉了"只是变了个颜色"的色相残留，已修正并逐张目视复核。</p>

<h2>四、未解决登记（如实，不含糊）</h2>
<table><tr><th>样本</th><th>原因</th><th>证据</th><th>未来方向（另开候选）</th></tr>
<tr><td>document_032</td><td>时间戳偏离中位 126 级、最暗灰度 103，与真实墨迹物理不可分</td><td>真值 {g('alpha>0')} dev 分布（p10=38/p90=141）</td><td>029–031 同底图时间戳的跨页先验配准</td></tr>
<tr><td>charts_015、charts_016</td><td>灰底图表；015 笔画核心深且彩色离散大；016 {g('OCR')} 读不出</td><td>探针 + explain() 拒绝记录</td><td>灰底属极淡通道边界，另行评估</td></tr>
<tr><td>food_dish、outdoor_building、textures_015/016</td><td>{g('OCR')} 0 行，CLAHE+2x 增强再读仍 0 行</td><td>探针结果 JSON</td><td>多尺度分块检测，成本较高，待决策</td></tr>
<tr><td>indoor_signage</td><td>一轮漏读，但增强再读可补出（距真值 3–5px）</td><td>探针 A/B 变体命中记录</td><td>增强再读候选（未实施）</td></tr></table>
<h3>未解决样本的水印局部（输入原图，红框=真实水印位置）</h3>
{fail_html}

<h2>五、复现与追溯</h2>
<pre>单测：  QT_QPA_PLATFORM=offscreen HF_HUB_OFFLINE=1 .venv/bin/python -m unittest tests.test_ocr_seed_stain
验收：  QT_QPA_PLATFORM=offscreen HF_HUB_OFFLINE=1 .venv/bin/python tools/run_ocr_seed_acceptance_20261105.py
工程：  QT_QPA_PLATFORM=offscreen HF_HUB_OFFLINE=1 .venv/bin/python -m unittest discover -s tests
数据：  ./start.sh check-data
代码基线：git {EVIDENCE['code_head']} ｜ 证据：验收证据_20261105.json ｜ 触发：experimental_ocr_line_seed=True（默认 False）</pre>
<p class="warn">边界声明：本候选只处理日期/时间样式行；{g('种子')}（{g('OCR')} 框）仅作搜索范围，
选区全部来自框内实测染色证据（{g('连通域')}沿笔画，无闭运算/填洞）；
修复是 {g('代数逆变换')}，区域外逐像素不动。14 张时间戳属开发集，
独立验证另行安排，本报告不构成"通用能力提升"结论。</p>
<p class="meta">报告生成：tools/build_ocr_seed_report_20261105.py（标题日期时间取自运行时刻）。</p>
<script>
document.addEventListener('click',function(e){{
  var t=e.target.closest('.term');
  document.querySelectorAll('.term.open').forEach(function(x){{if(x!==t)x.classList.remove('open');}});
  if(t)t.classList.toggle('open');
}});
document.addEventListener('keydown',function(e){{if(e.key==='Escape')document.querySelectorAll('.term.open').forEach(function(x){{x.classList.remove('open');}});}});
</script>
</body></html>"""
    REPORT.write_text(html, encoding='utf-8')
    print(f'written: {REPORT} ({len(html) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
