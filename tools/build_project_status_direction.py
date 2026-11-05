#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""项目现状与开发方向整理报告生成工具。

把当前项目的定位、管线、基线效果分层、工程管控状态和下一步开发方向整理成
一份自包含 HTML（纯文字，无图片依赖），标题带运行日期时间，术语用 ❓ 浮层。
数据源为磁盘上已封存的 20261104 基线摘要与项目文档，不重新跑任何推理。
"""
from __future__ import annotations

from datetime import datetime
from html import escape as e
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'out'

GLOSSARY = {
    '笔画召回率': '预测选区覆盖真实水印笔画像素的比例。只衡量"找全没有"，不衡量"多选没有"。',
    '精确率': '预测选区中真正属于水印笔画像素的比例。只衡量"多选没有"，不衡量"找全没有"。',
    '负例': '确认无水印的对照图片。负例被改动（哪怕一个像素被修复）就算保护失败。',
    '笔画补全': '在 OCR 框基础上沿真实笔画逐像素扩展选区的技术，桌面默认常开。',
    '透明度反解': '已知水印颜色和透明度时，用数学减法把染色从原像素中扣除，不是重画。',
    '兜底通道': '专用通道都没命中时的最后选区来源（报告里记为 tonal）。证据不足时给出大掩码是当前误伤来源之一。',
    '防退化控制': '冻结代码快照、跑固定回归、新旧逐页对比、门槛不达标不放行的版本管控机制。',
    '封存基线': '用冻结代码对固定回归集跑出的完整结果包，记录当时状态（含缺陷），作为以后所有改动的对照基准。',
    '独立验证': '从未参与过调参的样本集上的验证。项目独立集 v1 已因查看过而作废，当前无可用独立验证。',
    '运行错误': '处理过程中抛异常、无指标无产物的页次，保留在统计中，不删除。',
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
    """把术语包成 ❓ 浮层。"""
    for term, expl in GLOSSARY.items():
        if term in text and f'>{term}<' not in text:
            text = text.replace(term, f'{term}<button type="button" class="term" data-tip="{e(expl, quote=True)}">?</button>')
    return text


def table(headers, rows):
    th = ''.join(f'<th>{term_html(e(h))}</th>' for h in headers)
    trs = []
    for r in rows:
        trs.append('<tr>' + ''.join(f'<td>{term_html(e(str(c)))}</td>' for c in r) + '</tr>')
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'


EFFECT_ROWS = [
    ('斜向平铺文字', '76.7%', '66.0%', '基本能用，守住不退化'),
    ('居中大字', '74.4%', '71.1%', '基本能用，守住不退化'),
    ('模拟手写签名', '42.7%', '86.6%', '检出弱但不误伤，次优先'),
    ('高清小角标', '44.7%', '58.2%', '检出弱，次优先'),
    ('局部文字', '51.8%', '33.3%', '漏检和误伤双高，次优先'),
    ('白色半透明大字', '24.7%', '43.3%', '基本没检出，主攻'),
    ('网址文字', '29.7%', '28.1%', '基本没检出且误伤，主攻'),
    ('日期时间戳', '20.0%', '15.5%', '基本没检出，主攻'),
    ('透明几何 Logo', '16.6%', '87.5%', '基本没检出，主攻'),
    ('不透明几何 Logo', '1.2%', '66.5%', '基本没检出，主攻'),
]

PDF_ROWS = [
    ('PDF 独立透明图片', '94.2%', '99.7%', '好'),
    ('PDF 数字栅格化压平', '89.3%', '81.0%', '好'),
    ('PDF 中文编码文字', '80.4%', '97.0%', '好'),
    ('PDF 共用内容流', '79.5%', '96.4%', '好'),
    ('PDF 矢量路径', '5.1%', '100.0%', '结构类里的明显缺口'),
]

DIRECTION_ROWS = [
    ('1', '主攻：常见高频类型漏检归因＋修复', '时间戳、网址、Logo、白色半透明大字。先逐张归因（OCR 没找到／阈值没过／路由走错），拿证据再动算法。这类是招标成功率的分母。'),
    ('2', '次攻：局部文字', '召回和误伤双高，含兜底通道误伤背景的问题。'),
    ('3', '极淡文字保持隔离实验', '区域一致染色反变换候选仅在白纸验证过，走预登记门槛，不进默认路径，白纸之外如实标未解决。'),
    ('4', '守住：不退化清单', '斜向平铺、居中大字、PDF 结构类、负例；每轮改动过 ./start.sh test 分层回归。'),
    ('5', '择机补的工程账', 'inc11 止血闸门跨样式验证、墨色一致性预实验评审、A6000 实测；要宣称泛化需重建独立验证 v2（已决定不采真实样本，此账挂起）。'),
]


def main():
    now = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')
    title = f'项目现状与开发方向整理 {now}'
    out_dir = OUT / f'项目现状与开发方向整理_{now.split()[0].replace("-", "")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / '项目现状与开发方向整理.html'

    body = f"""
<h2>这份文档是什么</h2>
<p>把项目的定位、管线、当前效果、工程管控状态和下一步开发方向整理成一页，供对齐共识使用。
效果数字全部来自 2026-11-04 封存的固定回归基线（当前代码、桌面默认单模式、395 个输入共 423 页，
平均 15.3 秒/页，{term_html(e('负例'))} 26/27 安全，1 页{term_html(e('运行错误'))}），
不重新跑任何推理。生成时间 {now}。</p>

<h2>一、这个项目是什么、要做到什么</h2>
<ul>
<li><strong>目标</strong>：离线（本地、CPU 可跑）智能去水印，面向招标要求：成功率 &gt;60%、&lt;2 分钟/张（RTX A6000）、流式批处理。</li>
<li><strong>产品形态</strong>：一张图进、一张净图出；桌面默认单模式（文字识别与{term_html(e('笔画补全'))}已从界面删除开关、常开）。</li>
<li><strong>技术管线</strong>：OCR 定位＋笔画补全 → 周期/重复模板 → {term_html(e('透明度反解'))} → LaMa 修复；PDF 走结构对象删除（不走像素修复）。</li>
</ul>

<h2>二、现状：及格区</h2>
<p>图片类里只有两类基本能用：</p>
{table(['常见类型', '笔画召回率', '精确率', '结论'], EFFECT_ROWS[:2])}
<p>PDF 结构类普遍好（本质是对象级删除），矢量路径是唯一明显缺口：</p>
{table(['PDF 类型', '笔画召回率', '精确率', '结论'], PDF_ROWS)}

<h2>三、现状：失败区（恰是高频类型）</h2>
{table(['常见类型', '笔画召回率', '精确率', '结论'], EFFECT_ROWS[2:])}
<p>这批漏检页的共同特征是掉到{term_html(e('兜底通道'))}并给出大掩码，是背景误伤的来源之一。
极淡文字 15/15 全灭（历史既有缺陷），有"区域一致染色反变换"候选、仅在白纸验证过，未接入默认路径。</p>

<h2>四、口径限制（避免误读数字）</h2>
<ul>
<li>以上是<strong>开发集</strong>基线，数字偏乐观；{term_html(e('独立验证'))} v1 已因查看过而作废，当前没有独立验证，不能宣称泛化。</li>
<li>招标"水印去除成功率"的统计公式未与组委会书面确认。</li>
<li>像素误差不是水印残留率；423 页基线是观察基线，记录含缺陷的现状，不代表质量合格。</li>
</ul>

<h2>五、工程管控状态</h2>
<ul>
<li>{term_html(e('防退化控制'))}已运转：20261104 {term_html(e('封存基线'))}（单模式首个基线），今后改算法必须过 ./start.sh test 分层回归。</li>
<li>已落地未验证：inc11 止血闸门（修"补全整体替换选区"根因）未做跨样式回归；墨色一致性预实验待评审，未接默认路径。</li>
<li>环境欠账：A6000 从未实测（项目纪律"不测不冻结"）；本机缺中文字体，合成集评估跑不了。</li>
</ul>

<h2>六、接下来的开发方向（按优先级）</h2>
{table(['优先级', '方向', '说明'], DIRECTION_ROWS)}
<p><strong>明确不做</strong>：不补真实样本采集；不为开发集分数全局降阈值或扩大掩码；极淡、渐变、多色等低频合成样式不投入。</p>

<h2>七、一句话记住全局</h2>
<p class="lead">管线骨架成立、管控机制成立；真正的战场是常见高频类型的检出召回，其余是守住和挂账。</p>

<h2>八、依据与出处</h2>
<ul>
<li>效果数字：out/项目防退化控制/20261104_002213_219680_freeze/ 摘要报告（2026-11-04 16:00 版）。</li>
<li>样式与样本覆盖：docs/水印分类样本与最新测试.html、data/应用测试输入 清单。</li>
<li>极淡候选边界：docs/极淡水印方案与适用边界讨论_20261104.md。</li>
<li>管控口径：AGENTS.md 第 7、9 节，config/regression_control.json。</li>
</ul>
"""

    glossary_bar = '　'.join(
        f'{t}<button type="button" class="term" data-tip="{e(x, quote=True)}">?</button>'
        for t, x in GLOSSARY.items())

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title>
<style>
:root{{--ink:#203047;--muted:#63758a;--accent:#24639f;--border:#dce5ee;--page:#f2f5f8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--page);color:var(--ink);font:15px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:34px 26px 80px}}
section{{background:#fff;border:1px solid var(--border);border-radius:13px;padding:28px 32px;margin-bottom:24px}}
h1{{font-size:30px;line-height:1.35;margin:6px 0 14px}}
h2{{font-size:21px;margin:26px 0 12px;border-left:5px solid var(--accent);padding-left:12px;line-height:1.5}}
h2:first-child{{margin-top:0}}
.lead{{font-size:18px;color:#2c557f;background:#eff5fb;border-radius:8px;padding:14px 18px}}
.eyebrow{{font-size:12px;color:var(--muted)}}
ul{{padding-left:22px}}li{{margin-bottom:7px}}
.table-wrap{{overflow:auto;margin:14px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{text-align:left;vertical-align:top;padding:11px 12px;border-bottom:1px solid #e2e8f0;border-right:1px solid #eef2f6}}
th{{background:#edf3f8;color:#415e7b;white-space:nowrap}}
.glossary{{background:#f6f9fc;border:1px solid var(--border);border-radius:9px;padding:14px 18px;font-size:12px;color:#48627d;margin:18px 0}}
button.term{{border:1px solid #bcccdc;border-radius:50%;width:18px;height:18px;font-size:11px;line-height:1;color:#486581;background:#f0f4f8;cursor:pointer;padding:0;vertical-align:super}}
#term-tip{{position:absolute;background:#243b53;color:#fff;font-size:13px;padding:10px 14px;border-radius:8px;max-width:340px;display:none;z-index:9;line-height:1.6}}
code{{background:#eef2f6;border-radius:4px;padding:1px 6px;font-size:13px}}
</style></head><body>
<div id="term-tip"></div>
<main>
<p class="eyebrow">项目状态 · 生成于 {now} · 效果数字出自 20261104 封存基线，开发集口径</p>
<h1>{e(title)}</h1>
<div class="glossary"><strong>术语速查（点击 ❓ 查看解释，再点一次、点空白处或按 Esc 收起）：</strong><br>{glossary_bar}</div>
<section>{body}</section>
</main>
<script>{TIP_JS}</script>
</body></html>"""

    path.write_text(html, encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
