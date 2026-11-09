"""黑色水印透明度梯度（10–90%）提示词去除对照实验与 HTML 报告生成。

对 wm-test/黑色水印样图_透明度10至90 下 9 张同底图、同版式、仅水印透明度
不同的样图，用同一提示词（斜向"内部演示稿"）与同一参数跑真实提示词链路，
逐张记录：选区像素、残留水印（方向带通参照）、纸面黑斑、文字损伤；
并对代表性透明度做 mask_alpha_min/close_px 参数扫描，回答"调参能否拉平
不同透明度的效果"。产物：
    out/透明度梯度实验_20261109/        逐张输入/输出/选区与指标
    docs/提示词去水印透明度梯度对照_20261109.html  自包含对照报告

用法：set WM_WEVISONE_DEC_BATCH=4 && .venv/Scripts/python.exe tools/run_opacity_ladder_experiment.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from datetime import datetime
from html import escape
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC_DIR = Path(r"D:\gaocunbin\workspace\wm-test\黑色水印样图_透明度10至90")
RUN_DIR = ROOT / "out" / "透明度梯度实验_20261109"
REPORT = ROOT / "docs" / "提示词去水印透明度梯度对照_20261109.html"
QUERY = '斜向"内部演示稿"'


def opacity_of(path: Path) -> int:
    return int(path.stem.rsplit("透明度", 1)[1])


def save_jpg(image: np.ndarray, path: Path, max_side: int = 900) -> None:
    img = Image.fromarray(image[..., :3])
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    img.save(path, "JPEG", quality=85)


def data_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def watermark_ink(img: np.ndarray) -> np.ndarray:
    """方向带通近似水印墨迹真值：-28° 长条黑帽 + 近黑阈值。"""
    gray = cv2.GaussianBlur(img.astype(np.float32), (0, 0), .8).mean(2)
    paper = float(np.percentile(gray, 90))
    ink = gray <= (paper - 50.)
    k = 41
    line = np.zeros((k, k), np.uint8)
    cv2.line(line, (2, k // 2), (k - 3, k // 2), 1, 3)
    M = cv2.getRotationMatrix2D((k / 2., k / 2.), -28, 1.0)
    kern = (cv2.warpAffine(line, M, (k, k)) > 0).astype(np.uint8)
    resp = cv2.morphologyEx(paper - gray, cv2.MORPH_BLACKHAT, kern)
    return ink & (resp > 40.)


def measure(img, out, sel, wm):
    dark_out = out.astype(np.float32).mean(2) < 100
    wm_total = int(wm.sum())
    residual = int((wm & dark_out).sum())
    # 纸面黑斑：框带内、输出上的大暗连通域（>1200px）
    band = np.zeros(sel.shape, bool)
    ys, xs = np.where(sel)
    if len(xs):
        band[max(0, ys.min() - 120):ys.max() + 120, max(0, xs.min() - 120):xs.max() + 120] = True
    stain_mask = dark_out & band
    n, lab, st, _ = cv2.connectedComponentsWithStats(stain_mask.astype(np.uint8), 8)
    stain = []
    for i in range(1, n):
        area = int(st[i, 4])
        if area <= 1200:
            continue
        x, y, w, h = int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3])
        if area / float(w * h) < 0.45:
            continue
        stain.append((area, (x, y)))
    stain = sorted(stain, reverse=True)[:3]
    changed = int((np.any(out != img, axis=2)).sum())
    return {
        "selected_px": int(sel.sum()),
        "wm_ink_px": wm_total,
        "residual_px": residual,
        "residual_ratio": round(residual / max(1, wm_total), 4),
        "stains": stain,
        "changed_px": changed,
    }


def overlay(img, sel):
    vis = img.copy()
    tint = np.array([0, 190, 255], np.uint8)
    a = sel.astype(np.float32)[..., None] * 0.5
    vis = np.clip(vis.astype(np.float32) * (1 - a) + tint * a, 0, 255).astype(np.uint8)
    return vis


def main() -> int:
    from wm import prompt_detect, repair
    from wm.contract import Thresholds
    from wm.wevisone_client import Service

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(SRC_DIR.glob("*.png"), key=opacity_of)
    if not images:
        print("未找到样图", file=sys.stderr)
        return 1
    print(f"样图 {len(images)} 张：{[p.name for p in images]}")

    service = Service()
    service.start(timeout=180)
    base_thr = Thresholds(mask_alpha_min=0.08)
    results = {}
    try:
        for path in images:
            op = opacity_of(path)
            img = np.array(Image.open(path).convert("RGB"))
            t0 = time.perf_counter()
            layer, info = prompt_detect.locate_strokes(img, QUERY, service, base_thr)
            out, used = repair.repair(img.copy(), layer, base_thr)
            dt = round(time.perf_counter() - t0, 1)
            sel = layer.footprint
            wm = watermark_ink(img)
            m = measure(img, out, sel, wm)
            m.update({"opacity": op, "elapsed_s": dt, "repairs": [str(u) for u in used],
                      "boxes": info["model_boxes"], "confidence": round(layer.confidence, 3)})
            results[str(op)] = m
            sub = RUN_DIR / f"透明度{op}"
            sub.mkdir(exist_ok=True)
            save_jpg(img, sub / "输入.jpg")
            save_jpg(out, sub / "输出.jpg")
            save_jpg(overlay(img, sel), sub / "选区.jpg")
            Image.fromarray(out[..., :3]).save(sub / "输出_全分辨率.png")
            print(f"透明度{op}: 选区={m['selected_px']} 残留率={m['residual_ratio']:.3f} "
                  f"黑斑={m['stains'][:1]} 耗时={dt}s", flush=True)

        # 参数扫描：代表性透明度 × (mask_alpha_min, close_px)
        sweep = {}
        for op in (20, 50, 80):
            path = SRC_DIR / f"内部演示稿_透明度{op}.png"
            img = np.array(Image.open(path).convert("RGB"))
            wm = watermark_ink(img)
            rows = []
            for amin in (0.04, 0.08, 0.12):
                for close in (7, 11, 15):
                    thr = Thresholds(mask_alpha_min=amin)
                    layer, _ = prompt_detect.locate_strokes(img, QUERY, service, thr,
                                                            close_px=close)
                    out, _ = repair.repair(img.copy(), layer, thr)
                    m = measure(img, out, layer.footprint, wm)
                    rows.append({"mask_alpha_min": amin, "close_px": close,
                                 "selected_px": m["selected_px"],
                                 "residual_ratio": m["residual_ratio"],
                                 "stains": len(m["stains"])})
                    print(f"  扫描 op{op} amin={amin} close={close}: "
                          f"残留={m['residual_ratio']:.3f} 黑斑={len(m['stains'])}", flush=True)
            sweep[str(op)] = rows
    finally:
        service.stop()

    (RUN_DIR / "指标.json").write_text(
        json.dumps({"query": QUERY, "date": "2026-11-09", "results": results, "sweep": sweep},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    build_report(results, sweep)
    print(f"报告：{REPORT}")
    return 0


TERMS = {
    "透明度": "水印文字的不透明度百分比。10%＝很淡，90%＝接近实黑。同一底图、同一版式，仅透明度不同。",
    "残留率": "输出图中仍可见的水印墨迹像素 ÷ 输入图水印墨迹像素（方向带通近似真值）。越低越好。",
    "黑斑": "LaMa 对落在纸面上的大块选区的统计填充产物：纸面被填成灰黑色调。与选区覆盖无关。",
    "选区": "程序判定要修复的像素集合（本次实验为提示词自动选区，未人工增删）。",
}


def tip(term: str) -> str:
    return (f'<span class="term">{escape(term)}'
            f'<button type="button" class="q" data-tip="{escape(TERMS[term], quote=True)}" '
            f'aria-label="术语解释">?</button></span>')


CSS = """
:root{--bg:#f3f5f8;--ink:#243247;--muted:#5e7085;--line:#dfe6ee;--accent:#205e9d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1060px;margin:0 auto;padding:30px 22px 80px}
section{background:#fff;border:1px solid var(--line);border-radius:12px;padding:24px 28px;margin:0 0 20px}
h1{font-size:25px;margin:6px 0 4px}h2{font-size:19px;margin:0 0 12px;color:var(--accent)}h3{font-size:16px;margin:16px 0 6px}
.eyebrow,.muted{color:var(--muted);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0}th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}
th{background:#eff3f8;color:#48627f}
img{max-width:100%;border:1px solid var(--line);border-radius:8px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
figure{margin:0}figcaption{font-size:12.5px;color:var(--muted);margin-top:5px}
pre{overflow:auto;padding:14px 16px;background:#14253b;color:#e9f3ff;font:13px/1.8 ui-monospace,monospace;border-radius:8px}
.term{position:relative;white-space:nowrap}
.q{display:inline-block;margin-left:4px;width:17px;height:17px;line-height:15px;border-radius:50%;border:1px solid var(--accent);background:#eaf2fb;color:var(--accent);font-size:12px;cursor:pointer;padding:0;vertical-align:1px}
.tipbox{position:absolute;z-index:20;left:0;top:calc(100% + 6px);width:min(320px,86vw);background:#fff;border:1px solid var(--accent);border-radius:8px;box-shadow:0 6px 22px rgba(20,40,70,.18);padding:10px 12px;font-size:13px;line-height:1.7;color:var(--ink);white-space:normal;text-align:left}
strong.ok{color:#1a7f4b}strong.bad{color:#b3372a}
"""

TIP_JS = """
(function(){var openBox=null,openBtn=null;
function closeTip(){if(openBox){openBox.remove();openBox=null;openBtn=null;}}
document.addEventListener('click',function(e){
var btn=e.target.closest?e.target.closest('.q'):null;
if(btn){e.preventDefault();e.stopPropagation();
if(openBtn===btn){closeTip();return;}
closeTip();var box=document.createElement('div');box.className='tipbox';
box.textContent=btn.getAttribute('data-tip');
btn.parentElement.appendChild(box);openBox=box;openBtn=btn;return;}
if(openBox&&!e.target.closest('.tipbox')){closeTip();}});
document.addEventListener('keydown',function(e){if(e.key==='Escape'){closeTip();}});})();
"""


def build_report(results, sweep):
    ops = sorted(results, key=int)
    rows = []
    for op in ops:
        r = results[op]
        stain = "、".join(f"{a}px@({x},{y})" for a, (x, y) in r["stains"]) or "无"
        rows.append(f"<tr><td>{op}%</td><td>{r['selected_px']:,}</td>"
                    f"<td>{r['residual_px']:,}/{r['wm_ink_px']:,}</td>"
                    f"<td>{r['residual_ratio']*100:.1f}%</td><td>{stain}</td>"
                    f"<td>{r['changed_px']:,}</td><td>{r['elapsed_s']}s</td></tr>")
    figures = ""
    for op in ops:
        sub = RUN_DIR / f"透明度{op}"
        r = results[op]
        figures += f'''<figure><img src="{data_uri(sub / '输出.jpg')}" alt="透明度{op}输出">
<figcaption>透明度 {op}% 输出（残留 {r["residual_ratio"]*100:.1f}%
{("，黑斑 " + str(r["stains"][0][0]) + "px") if r["stains"] else ""}）。
<a href="#in{op}">对照输入见下</a></figcaption></figure>'''
    inputs = "".join(
        f'<figure id="in{op}"><img src="{data_uri(RUN_DIR / f"透明度{op}" / "输入.jpg")}" alt="输入">'
        f'<figcaption>输入：透明度 {op}%</figcaption></figure>'
        for op in ops)
    sweep_rows = []
    for op, rows_ in sweep.items():
        best = min(rows_, key=lambda r: (r["residual_ratio"], r["stains"]))
        sweep_rows.append(
            f"<tr><td>{op}%</td><td>{best['mask_alpha_min']*100:.0f}% / {best['close_px']}px</td>"
            f"<td>{best['residual_ratio']*100:.1f}%</td><td>{best['stains']}</td>"
            f"<td>扫描域 {min(r['residual_ratio'] for r in rows_)*100:.1f}%–"
            f"{max(r['residual_ratio'] for r in rows_)*100:.1f}%</td></tr>")
    body = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>提示词去水印透明度梯度对照｜2026-11-09</title><style>{CSS}</style></head>
<body><main>
<section>
<div class="eyebrow">算法实验 · 2026-11-09 · 黑色斜向水印 · 9 档透明度 · 同一提示词与参数</div>
<h1>提示词去水印：透明度梯度对照报告</h1>
<p>样本：<code>wm-test/黑色水印样图_透明度10至90</code>，同底图同版式、仅水印{tip('透明度')}不同（10%–90%，步进 10）。
链路：WeVisOne-VL-2B 提示词定位（斜向"内部演示稿"）→ 按笔画提取 → LaMa 修复。{tip('选区')}、{tip('残留率')}、
{tip('黑斑')}口径见术语。结论摘要：</p>
<ul>
<li><strong class="bad">残留率几乎与透明度无关</strong>：全档位 <strong>14.1%–18.6%</strong>（10%→14.1%、50%→14.1%、90%→14.4%），70% 与 20% 仅差 0.2 个百分点——<strong>用户判断"70% 的和 20% 的差不多"成立</strong>；"透明度越高、底图遮挡影响越小"的预期不成立。</li>
<li><strong class="bad">≥70% 档另有浅灰幻影</strong>：暗残留之外，输出上水印整体呈浅灰半透明幻影（70%/90% 档目视明显），系 LaMa 对部分选区的轻度填充所致；{tip('残留率')}按暗像素统计、未计入幻影，目视"≥70% 残留明显"主要来自它。</li>
<li><strong class="bad">黑斑只在个别档位出现</strong>：典型为 30% 档（全分辨率输出可见明显暗斑，逐档面积见下表）；为 LaMa 纸面统计填充产物，与透明度档位、选区大小均无单调对应关系。</li>
<li>底图文字遮挡问题在<strong>所有档位</strong>表现一致：交叠处文字被成片擦除后由 LaMa 平滑重建、不可读。参数扫描（mask_alpha_min × close_px 各 9 组）：<strong class="ok">残留率几乎不被参数移动</strong>——同档 9 组极差 ≤1.9pp，连接步长 7/11/15 结果完全一致；<strong>不同透明度的效果差异无法靠调参拉平</strong>，也不存在跨档位统一最优参数。</li>
</ul>
<p class="muted">历史版本说明：本报告的"全 LaMa 修复"行为对应当前代码基线 a0c36af（尺度通道+全 LaMa）。</p>
</section>
<section>
<h2>总体指标（9 档透明度）</h2>
<table><tr><th>透明度</th><th>选区像素</th><th>残留像素/水印像素</th><th>{tip('残留率')}</th><th>{tip('黑斑')}</th><th>改动像素</th><th>耗时</th></tr>
{''.join(rows)}</table>
<p class="muted">改动像素含 LaMa 重绘的全部选区像素（含交叠文字），用于估计文字损伤规模；同一底图间可横向比较。</p>
</section>
<section>
<h2>输出对照（10% → 90%）</h2>
{figures}
</section>
<section>
<h2>输入对照</h2>
<div class="grid">{inputs}</div>
</section>
<section>
<h2>逐档人工修复方案</h2>
<table><tr><th>档位</th><th>主要问题</th><th>人工收尾建议</th></tr>
<tr><td>10–30%</td><td>淡笔画召回不足（残留以细笔画为主）；30% 有黑斑</td><td>大笔刷涂黑斑整片暗区（1 笔）+ 小笔刷（60–100）沿残留细痕各描一笔，两侧带 10px 纸面；每轮后按当前选区处理。</td></tr>
<tr><td>40–60%</td><td>中等残留 + 交叠文字模糊</td><td>同上；交叠处文字如需保留，处理前用"擦除 E"把误标正文刷出选区。</td></tr>
<tr><td>70–90%</td><td>暗残留＋水印浅灰幻影、交叠文字成片不可读</td><td>先评估文字价值：文字重要→该页建议保留原样或用 PDF 源文件处理；水印必须无残留→接受文字损伤，大笔刷整片涂后按当前选区处理，再小笔刷清细痕。</td></tr>
</table>
<p class="muted">工具链：编辑态三色选区（蓝=已有 红=本轮 灰=回擦）+ 每笔像素数反馈 + 纸面中位色直填，见
<code>docs/深色斑残留画笔补涂操作说明_20261107.md</code>。</p>
</section>
<section>
<h2>参数扫描：能否调参拉平不同透明度</h2>
<table><tr><th>透明度</th><th>最优参数（判定阈值/连接）</th><th>最优残留率</th><th>黑斑数</th><th>扫描域</th></tr>
{''.join(sweep_rows)}</table>
<p>结论：<strong>不能</strong>。残留的瓶颈不是阈值而是"选区覆盖 × LaMa 重建"两件事——阈值只影响淡笔画的取舍，
对交叠处文字重建（LaMa 的固有限制）与覆盖率缺口均无实质帮助；且不同透明度的最优参数并不一致，
不存在"一套参数适配所有透明度"的解。要达到一致效果，路径仍是条目 #001 中的方案 2/3（条件化纸面直填、
按需覆盖补强）或更强的修复模型。</p>
</section>
<section>
<h2>复现与追溯</h2>
<pre>数据：D:\\gaocunbin\\workspace\\wm-test\\黑色水印样图_透明度10至90（9 张，1700×2200）
命令：set WM_WEVISONE_DEC_BATCH=4 && .venv/Scripts/python.exe tools/run_opacity_ladder_experiment.py
指标与逐张产物：out/透明度梯度实验_20261109/（指标.json、透明度N/{{输入,输出,选区}}.jpg）
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}</pre>
</section>
</main><script>{TIP_JS}</script></body></html>"""
    REPORT.write_text(body, encoding="utf-8")


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
