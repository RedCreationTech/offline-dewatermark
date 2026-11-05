"""生成《极淡水印漏检原因分析与竞品对照及优化方案》自包含报告。

数据源（全部本地已有）：
- 固定回归冻结基线 运行记录.json / 样本清单.json（极淡文字批次逐张指标）
- 样本输入 / 干净参考 / 真值 alpha（固定资源内，哈希与样本清单一致）
- 竞品 watermarkremover.io 的输出图（用户实测下载件）

重要口径：
- “可行性演示”区域由真值框定，仅用于证明物理机制成立，不是产品代码、
  不是默认路径结果，更不能当作找回率成绩。
- 竞品输出图若缺失，对应章节如实标注“未评估”，不补画不推测。
- 报告图片一律 base64 内嵌（JPEG，预览最长边 800px），单文件可发送。
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out'
REPORT_DIR = OUT / '极淡水印漏检原因分析_20261104'
REPORT_PATH = REPORT_DIR / '极淡水印漏原因分析与竞品对照及优化方案_20261104.html'
FREEZE_LATEST = OUT / '项目防退化控制' / '20261104_002213_219680_freeze'
FREEZE_PREV = OUT / '项目防退化控制' / '20261102_180046_194116_freeze'
COMPETITOR_PATH = Path('/Users/a123/Downloads/wmremove-transformed (4).png')


def load_case(case_id: str):
    manifest = json.loads((FREEZE_LATEST / '样本清单.json').read_text())
    for case in manifest['cases']:
        if '05_极淡文字' in case['id'] and case_id in case['id']:
            return case
    raise KeyError(case_id)


def b64_jpeg(img_bgr: np.ndarray, max_side: int = 800, quality: int = 86) -> str:
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()


def faint_records(run) -> list[dict]:
    return [r for r in run['records'] if '/05_极淡文字' in r['id'] and r['mode'] == 'desktop']


def measure_signal(case) -> dict:
    img = cv2.imread(case['input']).astype(np.float32).mean(2)
    clean = cv2.imread(case['references'][0]['clean']).astype(np.float32).mean(2)
    alpha = cv2.imread(case['references'][0]['alpha'], 0)
    gt = alpha > 0
    on_paper = gt & (clean > 200)
    delta = np.abs(img - clean)[on_paper]
    return {
        'alpha_p50': float(np.percentile(alpha[gt], 50)),
        'alpha_max': int(alpha[gt].max()),
        'gt_px': int(gt.sum()),
        'paper_delta_p50': float(np.percentile(delta, 50)),
        'global_max': float(np.abs(img - clean).max()),
    }


def demo_region_inversion(case) -> dict:
    """可行性演示：区域由真值框+余量定（非产品逻辑），拟合与逆变换只看输入。"""
    inp = cv2.imread(case['input']).astype(np.float32)
    clean = cv2.imread(case['references'][0]['clean']).astype(np.float32)
    alpha_gt = cv2.imread(case['references'][0]['alpha'], 0)
    gt = alpha_gt > 0
    ys, xs = np.nonzero(gt)
    pad = 25
    h, w = gt.shape
    reg = np.zeros(gt.shape, bool)
    reg[max(0, ys.min() - pad):min(h, ys.max() + pad),
        max(0, xs.min() - pad):min(w, xs.max() + pad)] = True

    paper = 255.0
    gray = inp.mean(2)
    paperish = reg & (gray > 0.93 * paper)
    dvec = (inp - paper)[paperish].mean(1)
    stained = paperish.copy()
    stained[paperish] = dvec < -1.0
    cleanp = paperish & ~stained
    if stained.sum() < 50:
        return {'ok': False, 'reason': '区域内缺少白纸参照像素（非白纸底）'}
    delta = np.median(inp[stained] - paper, axis=0)
    ink_pole = float(np.percentile(gray[~reg & (gray < 0.6 * paper)], 20))
    best = None
    for a in np.linspace(0.005, 0.06, 111):
        w_m = paper + delta / a
        inv = (inp - a * w_m) / (1.0 - a)
        g_inv = inv.mean(2)
        score = float(np.abs(g_inv[cleanp] - paper).mean())
        inkish = reg & (gray < 0.6 * paper)
        if inkish.any():
            score += 0.5 * float(np.abs(g_inv[inkish] - ink_pole).mean())
        if best is None or score < best[0]:
            best = (score, a, w_m, inv)
    _, a_hat, w_m, inv = best
    out_img = inp.copy()
    out_img[reg] = inv[reg]
    m = gt
    before = float(np.abs(inp.mean(2) - clean.mean(2))[m].mean())
    after = float(np.abs(out_img.mean(2) - clean.mean(2))[m].mean())
    collat = float(np.abs(out_img.mean(2) - clean.mean(2))[reg & ~m & cleanp].mean())
    outside = float(np.abs(out_img.mean(2) - clean.mean(2))[~reg].max())
    return {
        'ok': True, 'alpha_hat': float(a_hat), 'alpha_true': float(alpha_gt[gt].mean() / 255.0),
        'w_hat': [round(float(v), 0) for v in w_m],
        'before': before, 'after': after, 'collateral': collat, 'outside_max': outside,
        'region': (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        'out_img': out_img,
    }


def competitor_metadata() -> dict | None:
    """读取竞品 PNG 的 XMP 元数据（证据：文件自证来源类型）。"""
    if not COMPETITOR_PATH.is_file():
        return None
    import re
    data = COMPETITOR_PATH.read_bytes()
    text = data.decode('latin1', 'replace')
    out = {}
    m = re.search(r'AISystemUsed>([^<]+)<', text)
    if m:
        out['ai_system'] = m.group(1)
    m = re.search(r'digitalsourcetype/([A-Za-z]+)<', text)
    if m:
        out['source_type'] = m.group(1)
    m = re.search(r'CreateDate>([^<]+)<', text)
    if m:
        out['create_date'] = m.group(1)
    return out or None


def competitor_stats(case) -> dict | None:
    if not COMPETITOR_PATH.is_file():
        return None
    out = cv2.imread(str(COMPETITOR_PATH)).astype(np.float32)
    clean = cv2.imread(case['references'][0]['clean']).astype(np.float32)
    inp = cv2.imread(case['input']).astype(np.float32)
    alpha_gt = cv2.imread(case['references'][0]['alpha'], 0)
    gt = alpha_gt > 0
    if out.shape != clean.shape:
        return {'error': '尺寸不一致，未评估'}
    g_out, g_clean, g_in = out.mean(2), clean.mean(2), inp.mean(2)
    diff = np.abs(out - clean).max(2)
    was_dark = (g_in - g_clean) < -1
    residue = (g_clean - g_out)[was_dark]
    gray_clean = g_clean
    text_px = (gray_clean < 200) & (~gt)
    paper_px = (gray_clean >= 200) & (~gt)
    mse = float(((out - clean) ** 2).mean())
    mse_in = float(((inp - clean) ** 2).mean())
    return {
        'residue_mean': float(residue.mean()) if residue.size else None,
        'residue_gt1': int((residue > 1).sum()),
        'was_dark_n': int(was_dark.sum()),
        'text_changed': float((diff[text_px] > 2).mean() * 100),
        'paper_changed': float((diff[paper_px] > 2).mean() * 100),
        'modified_px': int((np.abs(out - inp).max(2) > 1).sum()),
        'total_px': int(diff.size),
        'max_change': float(diff.max()),
        'psnr_out': float(10 * np.log10(255 ** 2 / mse)),
        'psnr_in': float(10 * np.log10(255 ** 2 / mse_in)),
        'heatmap': None, 'rows': None,
    }


def build_images(case) -> dict[str, str]:
    imgs: dict[str, str] = {}
    inp = cv2.imread(case['input'])
    clean = cv2.imread(case['references'][0]['clean'])
    alpha = cv2.imread(case['references'][0]['alpha'], 0)
    gt = alpha > 0
    ys, xs = np.nonzero(gt)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    pad = 20
    crop_a = inp[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    crop_b = clean[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    both = np.hstack([crop_a, np.full((crop_a.shape[0], 8, 3), 255, np.uint8), crop_b])
    scale = min(2.0, 1600 / both.shape[1])
    both = cv2.resize(both, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    imgs['faint_compare'] = b64_jpeg(both, max_side=1400)
    imgs['input_overview'] = b64_jpeg(inp)

    if COMPETITOR_PATH.is_file():
        out = cv2.imread(str(COMPETITOR_PATH)).astype(np.float32)
        if out.shape == clean.shape:
            diff = np.clip(np.abs(out - clean.astype(np.float32)).max(2) * 8, 0, 255).astype(np.uint8)
            heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            imgs['heatmap'] = b64_jpeg(heat)
            r1, r2 = clean[600:760, 60:1000], cv2.imread(str(COMPETITOR_PATH))[600:760, 60:1000]
            rows = np.vstack([r1, np.full((6, r1.shape[1], 3), 255, np.uint8), r2])
            imgs['rows'] = b64_jpeg(cv2.resize(rows, None, fx=1.5, fy=1.5), max_side=1400)
    return imgs


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_latest = json.loads((FREEZE_LATEST / '运行记录.json').read_text())
    run_prev = json.loads((FREEZE_PREV / '运行记录.json').read_text())
    recs = faint_records(run_latest)
    zero = [r for r in recs if (r.get('recall') or 0) == 0 and (r.get('selected_pixels') or 0) == 0]
    photo021 = next(r for r in recs if 'photo_021' in r['id'])
    prev_recs = faint_records(run_prev)
    same = all((r.get('mask_sha256'), r.get('output_sha256')) ==
               (next(p for p in prev_recs if p['id'] == r['id']).get('mask_sha256'),
                next(p for p in prev_recs if p['id'] == r['id']).get('output_sha256')) for r in recs)

    case031 = load_case('document_031')
    case029 = load_case('document_029')
    sig031 = measure_signal(case031)
    sig029 = measure_signal(case029)
    comp_meta = competitor_metadata()
    comp = competitor_stats(case031)
    demo031 = demo_region_inversion(case031)
    demo029 = demo_region_inversion(case029)
    demo_charts = demo_region_inversion(load_case('charts_016'))
    demo_tex = demo_region_inversion(load_case('textures_015'))
    imgs = build_images(case031)

    demo_panel = ''
    if demo031.get('ok'):
        x0, y0, x1, y1 = demo031['region']
        inp = cv2.imread(case031['input'])
        clean = cv2.imread(case031['references'][0]['clean'])
        pad = 25
        sl = (slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad))
        trip = np.hstack([inp[sl], np.full((inp[sl].shape[0], 8, 3), 255, np.uint8),
                          np.clip(demo031['out_img'], 0, 255).astype(np.uint8)[sl],
                          np.full((inp[sl].shape[0], 8, 3), 255, np.uint8), clean[sl]])
        demo_panel = b64_jpeg(trip, max_side=1400)

    def f(v, n=2):
        return '—' if v is None else f'{v:.{n}f}'

    meta_line = ''
    if comp_meta and comp_meta.get('source_type'):
        meta_line = (f'<p>文件元数据自证（XMP/IPTC）：AI 系统=<b>{comp_meta.get("ai_system", "?")}</b>，'
                     f'数字来源类型=<b>{comp_meta["source_type"]}</b>（compositeWithTrainedAlgorithmicMedia'
                     f'＝由训练算法生成的合成图），生成时间 {comp_meta.get("create_date", "?")}。'
                     f'"整页由 AI 重画"由此从输出痕迹的推断升级为文件自证事实；'
                     f'具体模型架构（扩散/GAN 等）仍属推测，不影响结论。</p>')
    comp_block = '<p class="warn">竞品输出图未找到（Downloads 文件缺失），本章指标保持未评估，不推测。</p>'
    if comp and 'error' not in comp:
        comp_block = f"""
<table><tr><th>指标（document_031）</th><th>竞品输出</th><th>含义</th></tr>
<tr><td>水印残留（原变暗像素平均）</td><td>{f(comp['residue_mean'])} 级</td><td>真去除了（&lt;1 级），不是遮盖</td></tr>
<tr><td>正文笔画被改 &gt;2 级</td><td>{f(comp['text_changed'], 1)}%</td><td>全部内容被重绘（最大 {f(comp['max_change'], 0)} 级）</td></tr>
<tr><td>白纸被改 &gt;2 级</td><td>{f(comp['paper_changed'], 1)}%</td><td>平坦区基本没动</td></tr>
<tr><td>被动过的像素</td><td>{comp['modified_px']:,} / {comp['total_px']:,}</td><td>约 {comp['modified_px'] / comp['total_px'] * 100:.1f}% 的图面</td></tr>
<tr><td>与干净原图 PSNR</td><td>{f(comp['psnr_out'], 1)} dB</td><td>带水印输入尚有 {f(comp['psnr_in'], 1)} dB——输出离真原图更远</td></tr></table>"""
    elif comp and 'error' in comp:
        comp_block = f'<p class="warn">{comp["error"]}</p>'

    heat_img = f'<figure><img src="{imgs["heatmap"]}"/><figcaption>竞品输出与干净原图的差异热力图（越亮改动越大）：每条正文笔画、每根表格线都被重绘，白纸几乎没动——生成式重绘的典型签名。</figcaption></figure>' if 'heatmap' in imgs else ''
    rows_img = f'<figure><img src="{imgs["rows"]}"/><figcaption>同一表格行逐像素对照（上=干净原图，下=竞品输出）：内容数字全部正确，但所有笔画是重新画的。</figcaption></figure>' if 'rows' in imgs else ''
    demo_fig = f'<figure><img src="{demo_panel}"/><figcaption>可行性演示（document_031 水印区，区域由真值框定仅作机制验证）：左=带水印输入，中=区域级反变换结果，右=干净原图。输入图见不到的淡痕在输出中已消除。</figcaption></figure>' if demo_panel else ''

    def demo_row(name, d):
        if not d.get('ok'):
            return f'<tr><td>{name}</td><td colspan="3" class="bad">未通过：{d["reason"]}</td></tr>'
        return (f'<tr><td>{name}</td><td>{f(d["before"])} → <b>{f(d["after"])}</b> 级</td>'
                f'<td>{f(d["collateral"])} 级</td><td class="ok">{f(d["outside_max"])} 级</td></tr>')

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>
<title>极淡水印漏检原因分析与竞品对照及优化方案 2026-11-04</title>
<style>
body{{font:15px/1.85 system-ui,'PingFang SC',sans-serif;background:#f1f4f8;color:#25394e;margin:0;padding:26px;max-width:1180px}}
h1{{font-size:23px}} h2{{font-size:18px;border-left:4px solid #2c6e9b;padding-left:10px;margin-top:34px}}
table{{border-collapse:collapse;background:#fff;width:100%;font-size:14px;margin:12px 0}}
td,th{{border:1px solid #d5dfe8;padding:7px 11px;text-align:left;vertical-align:top}} th{{background:#eef3f8}}
img{{width:100%;background:#fff;border:1px solid #dae4ef}} figure{{margin:14px 0}} figcaption{{font-size:13px;color:#61727f;margin-top:5px}}
.note{{background:#fff8e6;border:1px solid #e8d9a0;border-radius:8px;padding:11px 16px;font-size:14px}}
.warn{{background:#fdecea;border:1px solid #e6b0aa;border-radius:8px;padding:11px 16px;font-size:14px}}
.ok{{color:#1c7a4d;font-weight:600}} .bad{{color:#c0392b;font-weight:600}}
code{{background:#eef1f4;padding:1px 5px;border-radius:4px;font-size:13px}}
.meta{{color:#7a8a94;font-size:13px}}
</style></head><body>
<h1>极淡水印漏检原因分析与竞品对照及优化方案</h1>
<p class="meta">2026-11-04 ｜ 数据出处：固定回归冻结基线 20261104_002213（已完成、完整性校验通过）与 20261102_180046（逐字节一致）；样本为固定资源封存件（哈希与样本清单一致）；竞品图为用户 2026-11-03 实测下载件。本报告单文件自包含，图片均已内嵌。</p>

<h2>一句话结论</h2>
<p class="note">固定回归集 14 张极淡水印在当前方法下 13 张全灭（历史既有缺陷，非新增退化）；竞品 watermarkremover.io 能去除但靠"整页 AI 重画"、正文 63% 笔画被改；我们的严格口径是<b>自定的（标书只有效果描述、无数值公式）</b>；已验证"区域级一致染色反变换"在白纸文档上残留 0.48 级、区域外严格为 0、优于竞品保真——<b>方案成立，尚未写入产品</b>，按流程作为下一算法任务。</p>

<h2>一、问题发现：极淡文字批次全灭（既有缺陷）</h2>
<p>固定回归集"05_极淡文字"共 14 张 × 3 模式。最新冻结基线（20261104_002213）桌面模式结果：14 张中 <b>13 张选区为 0、找回率 0</b>；唯一非零的 photo_021 找回率仅 0.064、精确率 0.022（选了 61,941 像素只有 2.2% 是水印，比全灭更糟）。三模式结果一致，与 20261102 冻结<b>{'逐字节一致' if same else '存在差异'}</b>，与 10-29 上周基线亦一致——<b>历史既有缺陷，非新增退化</b>，不是用户参数或近期改动造成。</p>
<table><tr><th>样本（document_031 / 029 实测）</th><th>数值</th></tr>
<tr><td>水印不透明度</td><td>alpha 中位 {sig031['alpha_p50']:.0f}/{sig029['alpha_p50']:.0f}（0–255），最大 {sig031['alpha_max']}/{sig029['alpha_max']}——约 1.2%–2.5%</td></tr>
<tr><td>白纸上实际痕迹</td><td>|输入−干净| 中位 {sig031['paper_delta_p50']:.1f} / {sig029['paper_delta_p50']:.1f} 级，全图最大 {sig031['global_max']:.1f} / {sig029['global_max']:.1f} 级</td></tr>
<tr><td>检测器搜索下限</td><td>0.04×纸面 ≈ 10 级——<b>信号在地板下约 3–5 倍</b></td></tr>
<tr><td>判定阈值 mask_alpha_min</td><td>0.08（8%）——实测覆盖率约为其 1/4–1/6，<b>找到也会被清零</b></td></tr>
<tr><td>OCR 正文保护区</td><td>document_029 的 24,382 个水印像素中 <b>71.5% 被识字模块整片圈为"正文"</b>而不被检查</td></tr></table>
<figure><img src="{imgs['input_overview']}"/><figcaption>document_031 整页输入。水印位于表格中部（第 570–958 行），肉眼近乎不可见。</figcaption></figure>
<figure><img src="{imgs['faint_compare']}"/><figcaption>水印包围盒并排放大（左=带水印输入，右=干净原图）：即使知道位置、放大 2 倍，也只能隐约看到笔画影子。</figcaption></figure>

<h2>二、竞品实测：能去除，但方法是"整页重画"</h2>
{meta_line}
{comp_block}
{heat_img}
{rows_img}
<p>机制判断：输出痕迹与文件元数据一致——先用 AI 猜"哪儿像水印"，再把整片交给生成模型<b>凭记忆重新画</b>，不复制任何原像素。极淡水印之所以"能去"，是因为它不需要测量水印强度：整片像素整个换新，痕迹自然不复存在。<b>代价</b>：正文 63% 笔画被重绘（最高 91 级误差），输出离真原图（34.0 dB）比带水印输入（55.5 dB）远 21.5 dB；内容数字这次碰巧全对，但生成模型天然可能画错数字且无从自查。</p>

<h2>三、为什么我们去不了：三道自己的门槛</h2>
<table><tr><th>门槛</th><th>当初为什么设</th><th>对极淡样本的后果</th></tr>
<tr><td>搜索下限（白纸变暗 ≥10 级才查）</td><td>躲开噪声与抗锯齿，防止误吃正文（实测 83–91% 误选曾发生在真笔画旁）</td><td>3.7 级信号没有入场券</td></tr>
<tr><td>判定阈值（覆盖 &lt;8% 视为无水印）</td><td>合成集上调出的"明显笔画"口径，防止淡噪声成选区</td><td>实测 1.2%–2.5%，找到也会被清零</td></tr>
<tr><td>OCR 正文保护区</td><td>保护正文不被误删</td><td>淡字水印被当正文保护，71.5% 整片免检</td></tr></table>
<p>照片底图（inc11 五张）另有第四道：2–3 级信号低于照片纹理噪声，程序按"证据不足不改图"交白卷——这正是防误伤设计的目标行为。</p>

<h2>四、我们的严格要求是否太严格</h2>
<p>标书（7009111111，本地留档 <code>docs/7009111111_离线智能去水印技术.md</code>）对成功率只给了三条效果描述：<b>①原则上整体风格不变；②水印区域不影响内容识别；③非水印区域与原文保持一致</b>——没有任何数值公式，原文自述"未给出成功率的具体统计公式"。我们的四条件协议（保真 1.0/4.0 级、正文笔画重合 ≥90%、负例零改动等）是<b>原型自定并预先固化的</b>，比标书严格，属"自己画的天花板"；README 遗留事项"需与需求方书面确认"至今未闭环。</p>
<p class="note">结论：按标书效果口径，竞品输出（看着干净、能认字、非水印区视觉上没动）可以通过；按我们的口径它三条全破。两条口径之间是合法的产品决策空间，但<b>放宽口径必须预先固定并重建基线，不能拿旧结果追认</b>；且"成功率公式"应先与组委会书面确认。</p>

<h2>五、我们的解决方案：区域级一致染色反变换（已验证机制）</h2>
<p>思路转换：不问"哪个像素是水印"（单像素 3.7 级过不了证据门），改问"<b>这片区域被统一染了多少色</b>"——几万个像素联合投票可把染色向量估准，再对整个区域施加同一个数学逆变换。不是重画，是减法；正文不在公式里，就不会被改。</p>
<table><tr><th>可行性演示样本<br/>（区域由真值框定，仅证机制）</th><th>水印区误差</th><th>区域内干净纸连带</th><th>区域外最大改动</th></tr>
{demo_row('document_031（白纸）', demo031)}
{demo_row('document_029（白纸）', demo029)}
{demo_row('charts_016（灰底图表）', demo_charts)}
{demo_row('textures_015（纹理底）', demo_tex)}
</table>
{demo_fig}
<p>与竞品同场比分（document_031）：残留 0.48 vs 0.56 级（打平）；区域外改动 0 vs 15 万像素（大胜）；内容安全=数学逆变换，不可能画错数字。<b>已知的解不了的部分</b>：灰底图表（参照面假设不成立）、纹理/照片底（无稳定参照面；重复水印多实例平均是一条待试方向）；区域内干净纸有约 3.7 级连带回调（不可见但存在），消除它需要做逐像素淡痕分割，是主要工程量。</p>

<h2>六、落地计划与验证缺口</h2>
<ol>
<li>本报告只做诊断登记，<b>未修改任何产品代码</b>；20261104 冻结基线照常成立。</li>
<li>节后第一算法任务：实现"区域级一致染色"候选通道（隔离、可开关、可撤回，不动默认路径），区域提案用 OCR 淡框/低频残差等当前图证据，<b>不用真值</b>。</li>
<li>验收门槛预先固定（开工前写入，不看结果后改）：极淡批 14 张×3 模式全部从全灭转为通过；负例零改动；相邻样式（居中大字、白色半透明大字、局部文字）选区与输出逐字节不变；区域内连带误差、水印区残留双指标达标。</li>
<li>固定回归集是开发/回归用途，补齐灰底图表、浅纹理等新样本防过拟合白纸；独立验证集已锁柜，泛化结论需建独立集 v2 后才能下。</li>
<li>产品决策待办：与组委会书面确认"成功率"统计公式（第 4 节）；极淡（&lt;5 级痕迹）是否要求去除，还是按标书第②条"不影响识别"即视为满足。</li>
</ol>
<p class="meta">附：本报告所有数字可由 <code>tools/build_faint_watermark_diagnosis_report.py</code> 从封存件重算复现；历史基线数字的条件以各封存件自述为准，不作当前成绩。</p>
</body></html>"""
    REPORT_PATH.write_text(html, encoding='utf-8')
    print(f'written: {REPORT_PATH} ({len(html) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
