"""PK 赛去水印批次测试报告生成器（skill 资产，固定评定模板）。

用法：
    python build_pk_report.py --config run_config.json

config 字段见同目录 run_config.example.json。输出自包含 HTML 到
<test_root>/<生成时间戳>/，并归档 manifest.json、queue.json 与 config。
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

GLOSSARY = {
    "T/N": "评比速度指标：批处理总时间 T（秒）÷ 批处理文档数量 N，即单张平均处理时间。T/N≤120 秒时按 (120−T/N)/6+15 计分，超过 120 秒不得分。",
    "水印去除成功率": "评比质量指标：被裁判判定为去除成功的文档数 A ÷ 输入文档数 B，得分=A/B×35。成功规则：去除后文档整体风格不变、水印区域不影响内容识别、非水印区域与原文保持一致。",
    "WeVisOne": "本地视觉大模型（WeVisOne-VL-2B），按提示词定位水印实例框，常驻服务方式运行。",
    "提示词定位": "找水印方式：把中文描述交给本地大模型给出水印实例框，框内再按像素证据挑笔画；模型框只作搜索范围。",
    "OCR 正文保护": "用文字识别圈出正文区域，检测时避开、修复时保护，防止正文被误选误改。",
    "LaMa": "深度学习修复模型（ONNX）。有正文保护的页面走 LaMa 按上下文重建；交叠处小号文字容易重建出错字（造字）。",
    "选区": "判定为水印、允许修复的像素集合；选区外像素逐像素保持原图。",
    "交叠残留": "水印笔画与正文交叉处遗留的少量水印像素；有时是为保护正文故意保留，有时是没选到的漏检。",
    "瑕疵": "本评测固定口径：水印已去除、文档可正常使用，但存在不影响识读的小墨点、淡痕或个别字符改花。",
    "纸面直填": "白底无正文页的修复方式：选区直接填纸面色，快但对压在正文上的笔画无能为力。",
    "置信度": "检测层对选区的把握（0~1），低于 0.12 触发证据闸门拦截（输出保持原图）。",
    "证据闸门": "修复前安全检查：选区为空/过小/置信度不足/重绘范围过大时不动原图并转人工。",
}

CSS = """
body{font-family:"Microsoft YaHei",system-ui,sans-serif;margin:0;background:#f2f5f7;color:#22303a}
main{max-width:1180px;margin:0 auto;padding:26px 30px 60px}
h1{font-size:24px;margin:6px 0 2px}h2{font-size:19px;margin:34px 0 10px;border-left:5px solid #243b46;padding-left:10px}
h3{font-size:16px;margin:18px 0 6px}
.sub{color:#7a8a94;font-size:13px}
.chips{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.chip{background:white;border:1px solid #d8e1e8;border-radius:12px;padding:12px 18px;min-width:160px}
.chip .v{font-size:24px;font-weight:700}.chip .k{font-size:12.5px;color:#7a8a94;margin-top:2px}
.good{color:#287d55}.mid{color:#967329}.bad{color:#ae4940}
table{border-collapse:collapse;font-size:13px;background:white;width:100%}
td,th{border:1px solid #d5dfe8;padding:6px 9px;text-align:left}td.r{text-align:right}
.card{background:white;border:1px solid #d8e1e8;border-radius:12px;padding:16px 20px;margin:14px 0}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:10px 0}
figure{margin:0}img{width:100%;height:320px;object-fit:contain;background:#f6f8f9;border-radius:6px}
figcaption{text-align:center;font-size:12px;color:#7a8a94;padding-top:4px}
.note{background:#fff8e6;border:1px solid #e8d9a0;border-radius:8px;padding:10px 16px;font-size:13px;margin:14px 0}
code{background:#eef2f5;border-radius:4px;padding:1px 5px;font-size:12.5px}
.term{border-bottom:1.5px dotted #b0761a;cursor:help}
.term i{display:inline-block;width:15px;height:15px;line-height:15px;text-align:center;font-size:10px;font-style:normal;background:#b0761a;color:white;border-radius:50%;margin-left:3px;vertical-align:2px}
#term-pop{position:fixed;display:none;max-width:360px;background:#243b46;color:#f2f5f7;font-size:13px;line-height:1.7;padding:14px 18px;border-radius:10px;box-shadow:0 8px 30px rgba(0,0,0,.35);z-index:99}
#term-pop b{color:#ffd58a}
ol li,ul li{margin:6px 0;font-size:14px;line-height:1.75}
p{font-size:14px;line-height:1.8}
.score{font-size:30px;font-weight:700}
"""

TERMS_JS = """
const GLOSSARY = __GLOSSARY__;
const pop = document.getElementById('term-pop');
document.addEventListener('click', e => {
  const t = e.target.closest('.term');
  if (t) {
    e.stopPropagation();
    const name = t.dataset.term || t.textContent.replace('?','').trim();
    const def = GLOSSARY[name] || '（暂无解释）';
    pop.innerHTML = '<b>' + name + '</b><br>' + def + '<br><span style="opacity:.6;font-size:11px">点击任意处收起</span>';
    pop.style.display = 'block';
    const r = t.getBoundingClientRect();
    pop.style.left = Math.min(window.innerWidth - 380, Math.max(8, r.left)) + 'px';
    pop.style.top = Math.min(window.innerHeight - 160, r.bottom + 8) + 'px';
    return;
  }
  if (pop.style.display === 'block') pop.style.display = 'none';
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') pop.style.display = 'none'; });
""".replace("__GLOSSARY__", json.dumps(GLOSSARY, ensure_ascii=False))


def esc(x) -> str:
    return html.escape("—" if x is None else str(x))


def mark_terms(text: str) -> str:
    for term in sorted(GLOSSARY, key=len, reverse=True):
        text = text.replace(term, f'<span class="term" data-term="{term}">{term}<i>?</i></span>')
    return text


def img64(path: Path, max_edge: int = 800, quality: int = 80) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_edge, max_edge))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def crop64(path: Path, box: tuple[int, int, int, int], max_edge: int = 760) -> str:
    im = Image.open(path).convert("RGB").crop(box)
    im.thumbnail((max_edge, max_edge))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def match_judgment(stem: str, judgments: dict[str, dict]) -> tuple[str, dict]:
    if stem in judgments:
        return stem, judgments[stem]
    for key in judgments:
        if stem.endswith(key) or key in stem:
            return key, judgments[key]
    raise KeyError(f"config.judgments 中没有与页 {stem!r} 匹配的判定（键需为页名或其后缀）")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="指定报告子目录（默认在 test_root 下新建时间戳目录）")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    exp = Path(cfg["export_dir"])
    manifest = json.loads((exp / "manifest.json").read_text(encoding="utf-8"))
    queue = json.loads((exp / "queue.json").read_text(encoding="utf-8"))
    states = {q["page"]: q["state"] for q in queue}
    now = datetime.now()
    stamp = f"{now:%Y-%m-%d %H:%M}"
    judgments: dict[str, dict] = cfg["judgments"]

    N = len(manifest["pages"])
    T_ms = sum(p["verdict"]["elapsed_ms"] for p in manifest["pages"])
    T = float(cfg.get("T_wall_seconds") or T_ms / 1000)
    tn = T / N
    speed_score = (120 - tn) / 6 + 15 if tn <= 120 else 0.0
    crop_box = tuple(cfg.get("crop_box", [500, 700, 1150, 1450]))

    rows, n_ok = [], 0
    for p in manifest["pages"]:
        stem = p["page"].split("#")[0]
        _, j = match_judgment(stem, judgments)
        grade = j["verdict"]
        if grade == "ok":
            n_ok += 1
        v = p["verdict"]
        det = p["processing"].get("prompt_detection", {})
        src = np.asarray(Image.open(exp / p["files"]["source"]).convert("RGB"), np.int16)
        out = np.asarray(Image.open(exp / p["files"]["clean"]).convert("RGB"), np.int16)
        z = np.load(exp / p["files"]["mask_data"])
        sel = z["alpha"] >= 0.08
        changed = (np.abs(out - src).max(axis=2) > 0)
        rows.append(dict(p=p, stem=stem, name=j.get("name", stem), v=v, det=det, j=j,
                         grade=grade, state=states.get(p["page"], "?"),
                         sel=int(sel.sum()), changed=int(changed.sum()),
                         out_sel=int((changed & ~sel).sum()),
                         src_mean=float(src[sel].mean()) if sel.any() else 0,
                         out_mean=float(out[sel].mean()) if sel.any() else 0))
    A_strict = n_ok
    A_lenient = A_strict
    for key in cfg.get("lenient_extra_pages", []):
        for r in rows:
            if r["stem"] == key or r["stem"].endswith(key) or key in r["stem"]:
                if r["grade"] != "ok":
                    A_lenient += 1
    rate_score = A_strict / N * 35
    rate_score_lenient = A_lenient / N * 35

    svc = sum(r["det"].get("service_ms", 0) for r in rows) / 1000
    mask_other = sum(r["v"]["stage_ms"].get("mask_ms", 0) - r["det"].get("service_ms", 0)
                     for r in rows) / 1000
    repair = sum(r["v"]["stage_ms"].get("repair_ms", 0) for r in rows) / 1000

    trs = []
    for r in rows:
        cls = "good" if r["grade"] == "ok" else "bad"
        trs.append(
            f"<tr><td>{esc(r['name'])}</td>"
            f"<td class='r'>{r['p']['mask_confidence']:.2f}</td>"
            f"<td class='r'>{r['sel']:,}</td>"
            f"<td class='r'>{r['p']['coverage']*100:.2f}%</td>"
            f"<td>{esc('+'.join(r['v']['repairs']) or '无')}</td>"
            f"<td class='r'>{r['changed']:,}</td>"
            f"<td class='r'>{r['out_sel']}</td>"
            f"<td class='r'>{r['src_mean']:.0f}→{r['out_mean']:.0f}</td>"
            f"<td class='r'>{r['v']['elapsed_ms']/1000:.1f}s</td>"
            f"<td class='{cls}'>{esc(r['j']['label'])}</td></tr>")
    page_table = ("<table><tr><th>页</th><th>置信度</th><th>选区px</th><th>覆盖率</th>"
                  "<th>修复方式</th><th>改动px</th><th>选区外改动</th><th>选区均色(原→出)</th>"
                  "<th>耗时</th><th>成功判定</th></tr>" + "".join(trs) + "</table>")

    cards = []
    for r in rows:
        f = r["p"]["files"]
        cls = "good" if r["grade"] == "ok" else "bad"
        figs = [
            (img64(exp / f["source"]), "输入原图"),
            (img64(exp / f["selection_overlay"]), "水印选区（红）"),
            (img64(exp / f["clean"]), "去水印输出"),
            (crop64(exp / f["source"], crop_box), "原图·水印区放大"),
            (crop64(exp / f["clean"], crop_box), "输出·水印区放大"),
        ]
        fig_html = "".join(f'<figure><img src="{u}"><figcaption>{esc(c)}</figcaption></figure>'
                           for u, c in figs)
        cards.append(
            f'<div class="card"><h3>{esc(r["name"])}　<span class="{cls}">{esc(r["j"]["label"])}</span></h3>'
            f'<p>{mark_terms(esc(r["j"]["note"]))}</p>'
            f'<p><small>应用状态：{esc(r["state"])} · 置信度 {r["p"]["mask_confidence"]:.2f} · 选区 '
            f'{r["sel"]:,}px · 改动 {r["changed"]:,}px（选区外 {r["out_sel"]}）· 修复 '
            f'{esc("+".join(r["v"]["repairs"]) or "无")} · 耗时 {r["v"]["elapsed_ms"]/1000:.1f}s'
            f'（模型服务 {r["det"].get("service_ms",0)/1000:.1f}s / 修复 '
            f'{r["v"]["stage_ms"].get("repair_ms",0)/1000:.1f}s）</small></p>'
            f'<div class="figs">{fig_html}</div></div>')

    cmp_html = ""
    comp = cfg.get("comparison")
    if comp:
        cmp_rows = ""
        for row in comp["rows"]:
            tds = "".join(
                f"<td class='r'>{esc(c)}</td>" if i else f"<td>{esc(c)}</td>"
                for i, c in enumerate(row))
            cmp_rows += f"<tr>{tds}</tr>"
        cmp_html = (f'<h2>六、{esc(comp["caption"])}</h2><div class="card"><table>'
                    f'<tr>' + "".join(f"<th>{esc(h)}</th>" for h in comp["header"]) + "</tr>"
                    + cmp_rows + "</table></div>")

    notes = cfg.get("notes", {})
    speed_extra = notes.get("speed", "")
    problems = notes.get("problems", [])
    limits = notes.get("limits", [
        "单批次成绩不代表多样式/多组合数据集成绩。",
        "成功判定为按规则原文的目视判定，无裁判真值。",
        "硬件差异：本机数字需在竞赛 RTX A6000 平台现场复测。",
        "残留/损伤为目视 + 像素代理指标，非精确召回率。",
    ])
    problems_html = "".join(f"<li>{mark_terms(esc(x))}</li>" for x in problems)
    limits_html = "".join(f"<li>{esc(x)}</li>" for x in limits)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(cfg.get("test_root", r"D:\gaocunbin\pk-test\test")) / f"{now:%Y%m%d_%H%M%S}"
        out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exp / "manifest.json", out_dir / "manifest.json")
    shutil.copy2(exp / "queue.json", out_dir / "queue.json")
    shutil.copy2(args.config, out_dir / "run_config.json")

    title = f"PK 赛两项科目批次测试报告 {stamp}"
    body = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{esc(title)}</title><style>{CSS}</style></head><body><main>
<h1>{esc(title)}</h1>
<p class="sub">批次：{esc(cfg.get('batch_label', exp.parent.name))} · 处理：{esc(str(exp))}</p>

<h2>一、本轮成绩（按评比要求两项科目）</h2>
<div class="chips">
<div class="chip"><div class="v">{T:.1f} 秒 / {N} 张</div><div class="k">批处理总时间 T 与数量 N（{esc(cfg.get('T_source_note', '应用批次总用时'))}）</div></div>
<div class="chip"><div class="v">{tn:.1f} 秒/张</div><div class="k">{mark_terms("T/N")}（≤120 秒有分）</div></div>
<div class="chip"><div class="v score">{speed_score:.1f}<span class="sub">/35</span></div><div class="k">水印去除速度得分</div></div>
<div class="chip"><div class="v">{A_strict} / {N}</div><div class="k">{mark_terms("水印去除成功率")} A/B（主口径）</div></div>
<div class="chip"><div class="v score">{rate_score:.1f}<span class="sub">/35</span></div><div class="k">成功率得分（主口径）</div></div>
<div class="chip"><div class="v score">{speed_score + rate_score:.1f}<span class="sub">/70</span></div><div class="k">本轮测试总成绩（主口径）</div></div>
</div>
<div class="note"><b>口径说明：</b>成功规则原文为「去除后文档整体风格不变、水印区域不影响内容识别、非水印区域与原文保持一致」。
主口径按字面从严：判<b>成功（有瑕疵）</b>的页必须水印去除干净、内容可识读，瑕疵（{mark_terms("瑕疵")}）逐页披露；
判<b>不通过</b>的页存在可见水印残留或非水印内容被改动。
若按宽松口径（{A_lenient}/{N}），成功率得分 {rate_score_lenient:.1f}/35，总分 {speed_score + rate_score_lenient:.1f}/70。最终以裁判判定为准。</div>

<h2>二、测试对象与口径</h2>
<div class="card"><p>
· 被测软件：{mark_terms(esc(cfg['software_note']))}<br>
· 硬件：{mark_terms(esc(cfg['hardware_note']))}<br>
· 软件设置：{mark_terms(esc(cfg['settings_note']))}<br>
· 输入图片：{esc(cfg['input_note'])}<br>
· 数据来源：应用「导出结果」目录（manifest.json 逐页判定/耗时 + 原图/选区/输出 PNG），未改动应用文件；改动像素、选区内外均色由导出图复算。manifest.json 与 queue.json 已随报告归档，供一致性核验。</p></div>

<h2>三、科目一：水印去除速度（满分 35）</h2>
<div class="card"><p>
· T = <b>{T:.1f} 秒</b>（逐页处理耗时合计 {T_ms/1000:.1f} 秒，两者互证），N = {N}。<br>
· {mark_terms("T/N")} = {T:.1f} ÷ {N} = <b>{tn:.1f} 秒/张</b>，得分 = (120 − {tn:.1f}) ÷ 6 + 15 = <b>{speed_score:.1f} 分</b>。<br>
· 耗时分解（合计 {T_ms/1000:.0f} 秒）：{mark_terms("WeVisOne")}模型服务 {svc:.0f} 秒（{svc/max(T_ms/1000,1)*100:.0f}%）＋ 框内笔画提取 {mask_other:.0f} 秒 ＋ 修复（{mark_terms("LaMa")}/纸面直填）{repair:.0f} 秒（{repair/max(T_ms/1000,1)*100:.0f}%）。{mark_terms(esc(speed_extra)) if speed_extra else ''}</p></div>

<h2>四、科目二：水印去除成功率（满分 35）</h2>
<div class="card"><p>
· 判定结果（主口径）：成功 A = <b>{A_strict}</b>，不通过 {N - A_strict}。成功率 A/B = <b>{A_strict/N*100:.1f}%</b>，得分 = <b>{rate_score:.1f} 分</b>。<br>
· 全部页的改动像素应落在选区内（选区外改动 = 0 为达标线），下表逐页列出核验值。</p>
{page_table}</div>

<h2>五、逐张证据（原图 / 选区 / 输出 / 放大）</h2>
{''.join(cards)}

{cmp_html}

<h2>七、问题清单与优化方向</h2>
<div class="card"><ol>{problems_html}</ol></div>

<h2>八、局限</h2>
<div class="card"><ul>{limits_html}</ul></div>

<footer><p class="sub">本报告自包含（图片已内嵌），可单文件发送。报告子目录归档 manifest.json / queue.json / run_config.json 供核验。数据源：{esc(str(exp))}</p></footer>
</main><div id="term-pop"></div><script>{TERMS_JS}</script></body></html>"""

    out = out_dir / f"PK两项科目批次测试报告_{now:%Y%m%d_%H%M%S}.html"
    out.write_text(body, encoding="utf-8")
    print(f"report -> {out} ({out.stat().st_size/1024/1024:.1f} MB)")
    print(f"dir    -> {out_dir}")


if __name__ == "__main__":
    main()
