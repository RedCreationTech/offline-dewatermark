"""生成 WeVisOne 提示词分割实验的独立 HTML 报告。

数据来源：out/WeVisOne提示词分割_斜向文字_20261106/（各提示词运行目录）
真值来源：data/watermark_regression_min_v1/references/core_009/alpha.png
报告为自包含单文件：所有图片以 base64 data URI 内嵌（照片 JPEG 最长边 800px，
二值掩码 PNG），发给他人无需附带目录。术语解释通过词旁 ❓ 浮层提供。
用法：.venv/Scripts/python.exe tools/build_wevisone_prompt_report.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out'
EXP = OUT / 'WeVisOne提示词分割_斜向文字_20261106'
SAMPLE_INPUT = ROOT / 'data/watermark_regression_min_v1/inputs/core/core_009.png'
GT_ALPHA = ROOT / 'data/watermark_regression_min_v1/references/core_009/alpha.png'
GT_MASK = ROOT / 'data/watermark_regression_min_v1/references/core_009/mask.png'
REPORT = OUT / 'WeVisOne提示词分割实验报告_20261106.html'

RUNS = [
    ('p1_原文案', '验证素材 3409'),
    ('p2_水印', '水印'),
    ('p3_斜向文字', '斜向文字'),
    ('p4_文字', '文字'),
    ('p5_英文watermark_text', 'watermark text'),
]

TERMS = {
    '斜向平铺文字': '同一行文字按固定角度重复排满整幅图的水印样式。本样本为绿色“验证素材 3409”，约 30° 角平铺，共 9 个实例，其中最下面一行 3 个被图像下边缘裁切。',
    '实例': '图中一次完整出现的水印文本。检测框和抠图都按实例给出；本图真值含 9 个实例，底行 3 个只剩上半部分。',
    '召回率（α>0）': '真值水印像素（alpha 通道大于 0 的全部像素）中被选中的比例。分母包含淡笔画和抗锯齿边缘，不能改分母来隐藏漏检。',
    '笔画召回率（α≥128）': '只统计真值 alpha 不低于 128 的明显笔画像素上的召回，阈值另行注明；用于观察淡边缘之外的实心笔画覆盖。',
    '精确率': '选中像素里属于真实水印的比例。精确率低说明把背景也选进来了，直接修复会损伤背景。',
    'alpha 通道': '合成水印图层时每个像素的不透明度（0–255）。本数据集真值由合成图层有效 8bit alpha 定义，mask.png 是 alpha>0 的支撑集。',
    '抠图（matte）': '模型为每个检出的实例输出的一张软透明度图（0–255）。本实验按 ≥128 二值化为“选中像素”后统计。',
    '检测框': '模型为每个实例给出的包围盒，只表示定位范围。按项目边界，检测框和粗抠图都不能直接当最终修复掩码。',
    'decoder_batch_size': '高分辨率抠图精修每批处理的实例个数。每批显存随 实例数×图幅 增长；本机 8GB 显卡默认 16 会溢出（OOM），设为 4 后跑通。',
    'OOM': 'Out Of Memory，显存溢出。本次首跑在 1152×864 原图 + bf16 下，精修阶段单次申请 6GiB 失败。',
    '开发集': '本项目里已被反复分析、调参用过的样本，不能当作独立泛化验证。core_009 属于固定最小回归集，是开发来源。',
}


def tip(term: str) -> str:
    text = TERMS[term]
    return (f'<span class="term">{escape(term)}'
            f'<button type="button" class="q" data-tip="{escape(text, quote=True)}" '
            f'aria-label="术语解释">?</button></span>')


def data_uri(path: Path, max_side: int = 800, photo: bool = True) -> str:
    img = Image.open(path)
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    if photo:
        img.convert('RGB').save(buf, 'JPEG', quality=82)
        mime = 'image/jpeg'
    else:
        img.save(buf, 'PNG', optimize=True)
        mime = 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def union_mask(run_dir: Path) -> np.ndarray:
    pred = None
    for mask in sorted((run_dir / 'masks').glob('mask_*.png')):
        arr = np.array(Image.open(mask).convert('L'))
        pred = arr if pred is None else np.maximum(pred, arr)
    if pred is None:
        raise SystemExit(f'{run_dir} 下没有 mask_*.png')
    return pred >= 128


def gt_instances(gt_all: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """把真值按实例分组（同一行文字内部先膨胀连通），返回包围盒与像素数。"""
    for it in range(12, 48, 4):
        lab, n = ndimage.label(ndimage.binary_dilation(gt_all, iterations=it))
        boxes = []
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            if len(xs) < 200:
                continue
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()),
                          int((gt_all & (lab == i)).sum())))
        if len(boxes) == 9:
            boxes.sort(key=lambda b: (round(b[1] / 200), b[0]))
            return boxes
    raise SystemExit('无法把真值划分为 9 个实例，请人工核对')


def main() -> None:
    alpha = np.array(Image.open(GT_ALPHA).convert('L'))
    gt_all = alpha > 0
    gt_strong = alpha >= 128
    instances = gt_instances(gt_all)

    rows = []
    per_instance = []
    for dirname, query in RUNS:
        run_dir = EXP / dirname
        sel = union_mask(run_dir)
        info = json.loads((run_dir / 'result.json').read_text(encoding='utf-8'))
        tp = int((sel & gt_all).sum())
        rows.append({
            'dirname': dirname, 'query': query,
            'mask_count': len(list((run_dir / 'masks').glob('mask_*.png'))),
            'selected_px': int(sel.sum()), 'frac': sel.mean(),
            'recall_all': tp / gt_all.sum(),
            'recall_strong': int((sel & gt_strong).sum()) / gt_strong.sum(),
            'precision': tp / max(1, int(sel.sum())),
            'missed_px': int((gt_all & ~sel).sum()), 'extra_px': int((sel & ~gt_all).sum()),
            'latency_s': info['latency_s'],
        })
        per_instance.append([float((sel[y0:y1 + 1, x0:x1 + 1] & gt_all[y0:y1 + 1, x0:x1 + 1]).sum()) / g
                             for (x0, y0, x1, y1, g) in instances])

    def pct(v: float) -> str:
        return f'{v * 100:.1f}%'

    metric_rows = ''.join(
        f"<tr><td>{escape(r['dirname'])}</td><td>“{escape(r['query'])}”</td><td>{r['mask_count']}</td>"
        f"<td>{r['selected_px']:,}</td><td>{pct(r['recall_all'])}</td><td>{pct(r['recall_strong'])}</td>"
        f"<td>{pct(r['precision'])}</td><td>{r['missed_px']:,}</td><td>{r['extra_px']:,}</td>"
        f"<td>{r['latency_s']:.1f}s</td></tr>"
        for r in rows)

    inst_rows = []
    for idx, (x0, y0, x1, y1, g) in enumerate(instances):
        cells = ''.join(
            f'<td class="{"miss" if v == 0 else ""}">{pct(v)}</td>' for v in (pi[idx] for pi in per_instance))
        inst_rows.append(f'<tr><td>实例{idx + 1}</td><td>({x0},{y0})–({x1},{y1})</td><td>{g:,}</td>{cells}</tr>')
    inst_table = ''.join(inst_rows)

    overlay_imgs = ''.join(
        f'<figure><img src="{data_uri(EXP / r["dirname"] / "overlay.png")}" alt="提示词 {escape(r["query"])} 的检测与抠图叠加">'
        f'<figcaption>提示词“{escape(r["query"])}”：{r["mask_count"]} 个实例，'
        f'α&gt;0 召回 {pct(r["recall_all"])}</figcaption></figure>'
        for r in rows)

    best = max(rows, key=lambda r: r['recall_all'])
    best_all = pct(best['recall_all'])
    best_strong = pct(best['recall_strong'])
    gt_all_n = int(gt_all.sum())
    gt_strong_n = int(gt_strong.sum())
    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WeVisOne 提示词分割实验｜斜向文字 core_009｜2026-11-06</title>
<style>
:root{{--bg:#f3f5f8;--ink:#243247;--muted:#5e7085;--line:#dfe6ee;--accent:#205e9d;--miss:#c0392b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:32px 22px 80px}}
section{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:26px 30px;margin:0 0 22px}}
h1{{font-size:27px;margin:6px 0 4px}}h2{{font-size:20px;margin:0 0 14px;color:var(--accent)}}h3{{font-size:16px;margin:22px 0 8px}}
.eyebrow,.muted{{color:var(--muted);font-size:13px}}.lead{{font-size:16px}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0}}th,td{{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}}
th{{background:#eff3f8;color:#48627f}}td.miss{{color:var(--miss);font-weight:700}}
figure{{margin:16px 0}}figcaption{{font-size:13px;color:var(--muted);margin-top:6px}}
img{{max-width:100%;border:1px solid var(--line);border-radius:8px}}
pre{{overflow:auto;padding:14px 16px;background:#14253b;color:#e9f3ff;font:13px/1.8 ui-monospace,monospace;border-radius:8px}}
.term{{position:relative;white-space:nowrap}}
.q{{display:inline-block;margin-left:4px;width:17px;height:17px;line-height:15px;border-radius:50%;border:1px solid var(--accent);background:#eaf2fb;color:var(--accent);font-size:12px;cursor:pointer;padding:0;vertical-align:1px}}
.q:focus-visible{{outline:2px solid var(--accent);outline-offset:1px}}
.tipbox{{position:absolute;z-index:20;left:0;top:calc(100% + 6px);width:min(320px,86vw);background:#fff;border:1px solid var(--accent);border-radius:8px;box-shadow:0 6px 22px rgba(20,40,70,.18);padding:10px 12px;font-size:13px;line-height:1.7;color:var(--ink);white-space:normal;text-align:left}}
ul{{padding-left:22px}}li{{margin:6px 0}}
code{{background:#eef2f7;border-radius:4px;padding:1px 5px}}
</style></head>
<body><main>
<section>
<div class="eyebrow">算法实验 · 2026-11-06 · 本机 RTX 3060 Ti 8GB · 入口 start.bat wevisone --gpu</div>
<h1>WeVisOne 提示词分割实验报告：斜向文字 core_009</h1>
<p class="lead">对固定回归样本 core_009（{tip('斜向平铺文字')}，1152×864）用 5 组提示词做分割对照。
最好的是“水印”（α&gt;0 召回 {best_all}、笔画召回 {best_strong}）；底行 3 个被下边缘裁切的实例里，
原文案和“水印”各检出 2 个，其余组更少，右下角实例 5 组全部漏检。
所有提示词{tip('精确率')}仅约 52%，{tip('抠图（matte）')}沿字形但明显偏粗，不能直接作为修复掩码。</p>
<p class="muted">这是单张{tip('开发集')}样本上的局部实验，不构成通用能力结论，未接入默认处理路径。</p>
</section>
<section>
<h2>实验设置</h2>
<table>
<tr><th>项目</th><th>内容</th></tr>
<tr><td>样本</td><td>core_009（data/watermark_regression_min_v1，inputs/core/core_009.png），{tip('斜向平铺文字')}，真值 {tip('alpha 通道')}&gt;0 共 {gt_all_n:,} px，α≥128 明显笔画 {gt_strong_n:,} px，共 9 个{tip('实例')}</td></tr>
<tr><td>模型 / 设备</td><td>models/WeVisOne-VL-2B（transformers 后端），cuda:0，bf16</td></tr>
<tr><td>推理入口</td><td><code>start.bat wevisone --gpu &lt;提示词&gt; &lt;图片&gt; &lt;输出目录&gt;</code>（内部跑 out/试跑_wevisone.py）</td></tr>
<tr><td>显存处理</td><td>原参数在 1152×864 + bf16 下{tip('OOM')}（精修单批申请 6GiB）。探针新增环境变量 <code>WM_WEVISONE_DEC_BATCH=4</code> 调小{tip('decoder_batch_size')}，入口用法不变；<code>PYTORCH_CUDA_ALLOC_CONF=expandable_segments</code> 在 Windows 版 torch 不支持，未生效也不需要</td></tr>
<tr><td>判定口径</td><td>{tip('抠图（matte）')}按 ≥128 二值化为选中像素后统计；{tip('召回率（α>0）')}分母含淡边缘，{tip('笔画召回率（α≥128）')}另注阈值；{tip('检测框')}只作定位范围</td></tr>
</table>
<h3>输入图与真值</h3>
<figure><img src="{data_uri(SAMPLE_INPUT)}" alt="core_009 输入图"><figcaption>输入图：绿色“验证素材 3409”约 30° 平铺，底行 3 个实例被下边缘裁切。</figcaption></figure>
<figure><img src="{data_uri(GT_MASK, photo=False)}" alt="真值掩码"><figcaption>真值掩码（mask.png，即 alpha&gt;0 的支撑集），二值图故用 PNG 内嵌。</figcaption></figure>
</section>
<section>
<h2>总体结果（5 组提示词）</h2>
<table>
<tr><th>运行目录</th><th>提示词</th><th>检出实例</th><th>选中像素</th><th>{tip('召回率（α>0）')}</th><th>{tip('笔画召回率（α≥128）')}</th><th>{tip('精确率')}</th><th>漏选 px</th><th>多选 px</th><th>耗时</th></tr>
{metric_rows}
</table>
<p class="muted">检出实例列：模型{tip('检测框')}数 / 真值 9 个。漏选 = 真值有而选中没有的像素；多选 = 选中而不属于真值的像素。</p>
</section>
<section>
<h2>逐实例召回（9 个实例 × 5 组提示词）</h2>
<table>
<tr><th>实例</th><th>包围盒 (x0,y0)–(x1,y1)</th><th>真值 px</th><th>p1 原文案</th><th>p2 水印</th><th>p3 斜向文字</th><th>p4 文字</th><th>p5 英文</th></tr>
{inst_table}
</table>
<p>实例 1–6 为完整显示的上两行，各组召回接近（0.84–0.94）；实例 7–9 是下边缘裁切行：<strong>原文案和“水印”各检出其中 2 个，“文字”检出 1 个，“斜向文字”和英文整行漏检；实例 9（右下角）5 组全部漏检</strong>（红字 0%）。</p>
</section>
<section>
<h2>目视证据（实际运行输出，非示意图）</h2>
<p>各组{tip('检测框')}与{tip('抠图（matte）')}叠加（模型保存的 overlay.png，预览降采样至最长边 800px）：</p>
{overlay_imgs}
<figure><img src="{data_uri(EXP / '局部对比_实例1.png')}" alt="实例1局部对比"><figcaption>实例 1 局部放大：黄 = 选中且真值命中，绿 = 真值漏选，品红 = 多选。可见 matte 沿字形但外扩偏粗，是{tip('精确率')}只有约 52% 的原因。</figcaption></figure>
</section>
<section>
<h2>发现与结论</h2>
<ul>
<li>提示词越指向水印本身效果越好：“验证素材 3409”（原文案）与“水印”总体最优；底行被边缘裁切的 3 个实例中，这两组各检出 2 个，描述形态（“斜向文字”）和英文（“watermark text”）整行漏检，右下角实例 5 组全漏。</li>
<li>完整实例上各组差异不大（0.84–0.94），差距主要来自候选{tip('实例')}检测，而非抠图质量。</li>
<li>所有提示词{tip('精确率')}约 52%：matte 沿字形但系统性偏粗（多选约 2.9 万 px）。按项目边界，该结果不能直接当修复掩码；若要使用需后处理（阈值/腐蚀类）并独立验证。</li>
<li>单图推理 16–22s（含检测 + 抠图），模型加载另计。</li>
</ul>
</section>
<section>
<h2>局限与未解决</h2>
<ul>
<li>仅 1 张样本、5 组提示词，且 core_009 属固定最小回归集（{tip('开发集')}），不能作为独立泛化验证。</li>
<li>本轮只评分割，未跑修复闭环；matte 偏粗的问题未做后处理对照。</li>
<li>未与其他水印样式（点阵、Logo、PDF 等）交叉验证，无退化结论。</li>
<li>候选规则未接入默认路径，保持实验状态。</li>
</ul>
</section>
<section>
<h2>复现方式</h2>
<pre>set WM_WEVISONE_DEC_BATCH=4
start.bat wevisone --gpu "验证素材 3409" data/watermark_regression_min_v1/inputs/core/core_009.png out/WeVisOne提示词分割_斜向文字_20261106/p1_原文案
:: 其余 4 组把提示词依次换成：水印 / 斜向文字 / 文字 / "watermark text"，输出目录同步更换
:: 本报告由 tools/build_wevisone_prompt_report.py 重新生成（指标在生成时对照真值重算）</pre>
<p class="muted">运行产物目录：out/WeVisOne提示词分割_斜向文字_20261106/（各组 masks/、overlay.png、result.json，评分汇总.json，局部对比_实例1.png）。</p>
</section>
</main>
<script>
(function () {{
  var openBox = null, openBtn = null;
  function closeTip() {{
    if (openBox) {{ openBox.remove(); openBox = null; openBtn = null; }}
  }}
  document.addEventListener('click', function (e) {{
    var btn = e.target.closest ? e.target.closest('.q') : null;
    if (btn) {{
      e.preventDefault(); e.stopPropagation();
      if (openBtn === btn) {{ closeTip(); return; }}
      closeTip();
      var box = document.createElement('div');
      box.className = 'tipbox';
      box.textContent = btn.getAttribute('data-tip');
      btn.parentElement.appendChild(box);
      openBox = box; openBtn = btn;
      return;
    }}
    if (openBox && !e.target.closest('.tipbox')) {{ closeTip(); }}
  }});
  document.addEventListener('keydown', function (e) {{
    if (e.key === 'Escape') {{ closeTip(); }}
  }});
}})();
</script>
</body></html>'''

    REPORT.write_text(html, encoding='utf-8')
    external = [part for part in html.split('src="')[1:] if not part.startswith('data:')]
    print(f'报告已生成: {REPORT}')
    print(f'外部图片引用数: {len(external)}（应为 0，报告自包含）')
    print(json.dumps({'sample': 'core_009', 'runs': len(rows),
                      'best_query': best['query'], 'recall_alpha_gt0': round(best['recall_all'], 4)},
                     ensure_ascii=False))


if __name__ == '__main__':
    sys.exit(main())
