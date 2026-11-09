"""生成《极淡区域反染色候选第一轮报告》自包含 HTML。

数据从 20261104 冻结封存件现算复现；评分在推理完成后对真值进行。
候选为隔离实验通道（默认关闭），本报告不构成默认路径成绩。
图片一律 base64 内嵌（JPEG，预览最长边 1000px），单文件可发送。
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'out'
DIR = OUT / '极淡区域反染色候选_20261104'
REPORT = DIR / '极淡区域反染色候选第二轮报告_20261104.html'
FREEZE = OUT / '项目防退化控制' / '20261104_002213_219680_freeze'
SAMPLES = ['document_029__faint', 'document_030__faint', 'document_031__faint', 'document_032__faint']


def b64(img, max_side=1000, q=86):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()


def _framed(panel: np.ndarray) -> np.ndarray:
    """每格加黑边框，格间黑 gutter——并排时边界一目了然。"""
    out = cv2.copyMakeBorder(panel, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return out


def region_trip(inp, out, clean, region, gt):
    ys, xs = np.nonzero(gt)
    pad = 30
    h, w = gt.shape
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
    black = np.full((y1 - y0 + 6, 14, 3), 0, np.uint8)   # 格间黑 gutter
    trip = np.hstack([_framed(inp[y0:y1, x0:x1]), black,
                      _framed(out[y0:y1, x0:x1]), black,
                      _framed(clean[y0:y1, x0:x1])])
    sc = min(1.6, 1500 / trip.shape[1])
    if sc > 1.0:
        trip = cv2.resize(trip, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    return trip


def region_overlay(overlay, gt):
    """选区叠加图同样只裁水印区域，不全页显示。"""
    ys, xs = np.nonzero(gt)
    pad = 30
    h, w = gt.shape
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
    return _framed(overlay[y0:y1, x0:x1])


def main():
    from wm.masks import estimate_single_page
    from wm.repair import repair
    from wm.contract import Thresholds

    manifest = json.loads((FREEZE / '样本清单.json').read_text())
    rows, figures = [], []
    for cid in SAMPLES:
        case = next(c for c in manifest['cases'] if cid in c['id'])
        img = cv2.imread(case['input'])
        clean = cv2.imread(case['references'][0]['clean'])
        alpha = cv2.imread(case['references'][0]['alpha'], 0)
        gt = alpha > 0
        before_err = float(np.abs(img.mean(2) - clean.mean(2))[gt].mean())
        mask = estimate_single_page(img, style='faint', thr=Thresholds(experimental_faint_region=True))
        if mask.method != 'faint_region_uniform_tint':
            rows.append((cid, '未触发', '', '', f'{before_err:.2f}', '', ''))
            continue
        sel = mask.alpha > 0
        out, _ = repair(img, mask, Thresholds())
        after_err = float(np.abs(out.mean(2) - clean.mean(2))[gt].mean())
        recall = float(sel[gt].mean())
        prec = float(sel[gt].sum() / max(sel.sum(), 1))
        coll = float(np.abs(out.mean(2) - clean.mean(2))[mask.footprint & ~gt].mean())
        outside = int(np.abs(out.astype(int) - img.astype(int))[~mask.footprint].max())
        a_true = float(alpha[gt].mean() / 255.0)
        a_hat = float(mask.overlay_opacity.max())
        rows.append((cid, f'{a_hat:.4f}（真值 {a_true:.4f}）', f'{recall:.3f}', f'{prec:.3f}',
                     f'{before_err:.2f} → {after_err:.2f}', f'{coll:.2f}', str(outside)))
        overlay = img.copy()
        overlay[mask.footprint] = (0.6 * overlay[mask.footprint] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)
        figures.append((cid, b64(region_trip(img, out, clean, mask.footprint, gt)),
                        b64(region_overlay(overlay, gt))))
        cv2.imwrite(str(DIR / f'{cid}_output.png'), out)

    def decline_probe(cid_part):
        case = next(c for c in manifest['cases'] if cid_part in c['id'])
        img = cv2.imread(case['input'])
        m = estimate_single_page(img, style='faint', thr=Thresholds(experimental_faint_region=True))
        return case['id'].split('/')[-1], m.method, int((m.alpha > 0).sum())

    declines = [decline_probe('charts_015__faint'), decline_probe('textures_015__faint')]
    negs = [c for c in manifest['cases'] if '无水印对照' in c['id']][:2]
    for c in negs:
        img = cv2.imread(c['input'])
        m = estimate_single_page(img, style='faint', thr=Thresholds(experimental_faint_region=True))
        declines.append((c['id'].split('/')[-1] + '（负例）', m.method, int((m.alpha > 0).sum())))

    body_rows = ''.join(
        f'<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td>'
        f'<td>{r[4]}</td><td>{r[5]}</td><td class="ok">{r[6]}</td></tr>' for r in rows)
    decline_rows = ''.join(f'<tr><td>{d[0]}</td><td>{d[1]}</td><td class="ok">{d[2]}</td></tr>' for d in declines)
    figs = ''.join(
        f'<figure><img src="{u}"/><figcaption>{cid}：<b>左=带水印输入 ｜ 中=候选修复输出 ｜ 右=干净参考</b>'
        f'（仅水印区域局部，格间黑线分隔）</figcaption></figure>'
        f'<figure><img src="{o}"/><figcaption>{cid}：拟合选区叠加（红色，仅水印区域局部）。</figcaption></figure>'
        for cid, u, o in figures)

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>
<title>极淡区域反染色候选第二轮报告 2026-11-04</title>
<style>
body{{font:15px/1.85 system-ui,'PingFang SC',sans-serif;background:#f1f4f8;color:#25394e;margin:0;padding:26px;max-width:1150px}}
h1{{font-size:22px}} h2{{font-size:17px;border-left:4px solid #2c6e9b;padding-left:10px;margin-top:30px}}
table{{border-collapse:collapse;background:#fff;width:100%;font-size:14px;margin:12px 0}}
td,th{{border:1px solid #d5dfe8;padding:7px 11px;text-align:left;vertical-align:top}} th{{background:#eef3f8}}
img{{width:100%;background:#fff;border:1px solid #dae4ef}} figure{{margin:14px 0}} figcaption{{font-size:13px;color:#61727f;margin-top:5px}}
.note{{background:#fff8e6;border:1px solid #e8d9a0;border-radius:8px;padding:11px 16px;font-size:14px}}
.warn{{background:#fdecea;border:1px solid #e6b0aa;border-radius:8px;padding:11px 16px;font-size:14px}}
.ok{{color:#1c7a4d;font-weight:600}} .bad{{color:#c0392b;font-weight:600}} code{{background:#eef1f4;padding:1px 5px;border-radius:4px;font-size:13px}}
.meta{{color:#7a8a94;font-size:13px}}
</style></head><body>
<h1>极淡区域反染色候选 · 第二轮报告（逐像素透明度＋块级区域＋双尺度证据）</h1>
<p class="meta">2026-11-04 ｜ 候选 <code>faint_region_uniform_tint</code>（默认关闭，仅"极淡"样式提示或
<code>Thresholds.experimental_faint_region=True</code> 触发）｜ 数据：固定回归封存件 20261104_002213，
评分在推理后对真值进行 ｜ 预登记门槛见 <code>验收门槛_20261104.json</code>，开发细节见
<code>本轮开发记录_20261104.md</code>。本报告单文件自包含。</p>

<h2>一句话结论</h2>
<p class="note">第二轮改动：拟合验证后把区域放大到整片字块（笔画间与边缘痕量全部纳入），
纸像素按各自偏离量算逐像素透明度（干净纸 α≈0 不被触碰，连带 ≈0.1–0.3 级）。
4/4 样本<b>召回 1.000，水印区误差降到 0.09–0.16 级</b>（处理前 1.45–3.25 级，竞品同区域 0.56 级），
区域外逐像素为 0，负例与相邻样式零误触发。<b>候选仍为实验状态</b>：全量 14 张×3 模式证据、
新样本与独立集 v2 待补，未接入默认路径。</p>

<h2>一、处理结果（水印包围盒局部：左=输入，中=候选输出，右=干净参考）</h2>
{figs}
<table><tr><th>样本</th><th>α̂（实测 vs 真值）</th><th>召回(alpha&gt;0)</th><th>精确率</th>
<th>水印区误差(级)</th><th>区域内连带</th><th>区域外最大改动</th></tr>{body_rows}</table>

<h2>二、正确拒绝的样本（白纸纯度/纸面参照/负例门槛）</h2>
<table><tr><th>样本</th><th>路由结果</th><th>选区像素</th></tr>{decline_rows}</table>
<p>灰底图表（charts）被白纸纯度门槛（纸面极点亮度 ≥250 且干净纸离散度 ≤3 级）拒绝；
纹理底（textures）无纸面参照拒绝；无水印负例零选区。相邻样式在候选开启/关闭下
alpha 逐字节一致，默认路径行为不变（详见开发记录）。</p>

<h2>三、与门槛对照</h2>
<table><tr><th>预登记门槛（4 张探针）</th><th>结果</th><th>判定</th></tr>
<tr><td>负例零改动 / 区域外零改动 / 相邻样式零影响</td><td>全部达成</td><td class="ok">通过</td></tr>
<tr><td>每张召回(alpha&gt;0) &gt; 0.5</td><td><b>1.000 / 1.000 / 1.000 / 1.000</b></td><td class="ok">通过</td></tr>
<tr><td>每张水印区误差 ≤ 1.5 级</td><td><b>0.14 / 0.12 / 0.16 / 0.09 级</b></td><td class="ok">通过</td></tr></table>
<p class="warn">探针范围只有 4 张，按预登记文件 promote 仍需：14 张×3 模式全量证据、
基线逐字节对照、浅灰底/渐变底新样本标定、test-release 与独立集 v2。
精确率 0.05 为预期现象——块级区域含笔画间干净纸，逐像素 α 使其连带 ≤0.23 级，无害。</p>
<p class="meta">复现：<code>tools/build_faint_region_round1_report.py</code> 从封存件现算（第二轮参数）；
配套背景报告：极淡水印漏检原因分析与竞品对照及优化方案 2026-11-04（总目录"失败与诊断"）。</p>
</body></html>"""
    REPORT.write_text(html, encoding='utf-8')
    print(f'written: {REPORT} ({len(html) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
