"""把 run_dev_batch.py 的一轮产物整理成自包含 HTML 报告。

报告以「原图 / 结果图 / 水印图」三张一组平铺展示，全部图片 base64 内嵌，
单个 HTML 文件发出即可完整查看。若提供 dataset.json 与 *_gt.npz 真值，
只在推理产物生成完成后加载，用于事后评估（召回率／精确率等），不参与
任何推理过程。

用法：
  python tools/build_dev_batch_report.py --run-dir out/某批次/run1 \
      --dataset-json out/某批次/输入/dataset.json \
      --gt-dir out/某批次/输入 [--out 报告.html]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

PREVIEW_MAX = 800
BORDER_WIDTH = 36  # 等效 200dpi 下约 60px 的图边范围，用于拆分边框/内部残留

TERMS = {
    "recall": "召回率：真值水印像素中被选区覆盖的比例。分母是 alpha 大于 0 的全部水印像素，包含淡边缘，不能用提高阈值的方式缩小分母。",
    "stroke_recall": "明显笔画召回率：真值 alpha 不低于 0.5 的明显笔画像素中被选中的比例；0.5 阈值仅用于分档观察，不替代上一条全量口径。",
    "precision": "精确率：选区像素中真实属于水印的比例；其补集即多选（误选背景或正文）像素占比。",
    "coverage": "选区覆盖率：选区像素占整图像素的比例，只说明选区多大，不代表找得准。",
    "outside_change": "选区外改动：结果图在选区掩码之外仍发生变化的像素数；理想情况下应为 0，非 0 说明改动越过了选区边界。",
    "crosspage": "跨页先验：同一批次内同尺寸页面的共享定位证据，只作搜索范围，不直接成为修复选区。",
    "recall_priority": "笔画补全： always-on 能力，把与正文交叠的真实水印笔画补进选区，允许牺牲正文保护但不允许整片擦除背景。",
    "auto_mode": "自动识别模式：找水印方式为自动识别（非提示词定位），水印样式由程序自动路由。",
    "review": "待复核：识别质量或残留质量无法可靠测量时的状态，不等于失败，也不计入已验证成功。",
    "watermark_img": "水印图：程序从原图中按选区提取出的水印内容，用于人工核对定位是否准确。",
    "engine": "修复引擎：实际执行去水印的算法路径（如透明度反解、区域重绘等），由水印样式与证据自动决定。",
    "ref_error": "参考误差前→后：原图／结果图与干净参考在真实水印区域内的平均像素色差（前→后）。它混合了残留与重建差异，不是水印残留率；明显下降说明有效去除，上升说明修复反而加重了痕迹。",
    "resid": "可见残留像素：真实水印区域内、结果图与干净参考色差仍超过 15 的像素数。平均值会被大面积干净区域稀释，这个数抓小面积但肉眼可见的水印残留。",
}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def embed(path: Path, fmt: str = "JPEG", quality: int = 88) -> str:
    with Image.open(path) as im:
        if im.mode == "RGBA":
            # 提取的水印带 alpha 通道；直接转 RGB 会把整页填成 veil 色，
            # 先合成到白底再嵌入。
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im).convert("RGB")
        else:
            im = im.convert("RGB")
        if max(im.size) > PREVIEW_MAX:
            im.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.Resampling.LANCZOS)
        buf = BytesIO()
        if fmt == "JPEG":
            im.save(buf, "JPEG", quality=quality)
        else:
            im.save(buf, "PNG", optimize=True)
    return f"data:{ 'image/jpeg' if fmt == 'JPEG' else 'image/png' };base64," + base64.b64encode(buf.getvalue()).decode()


def term_html(key: str, label: str) -> str:
    return f'<span class="term" data-term="{key}">{esc(label)}<sup>❓</sup></span>'


def fmt_pct(value) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def load_gt(gt_dir: Path, doc_id: str):
    for cand in (gt_dir / f"{doc_id}_gt.npz", gt_dir / f"{doc_id}.npz"):
        if cand.is_file():
            data = np.load(cand)
            if "alpha" in data:
                clean = np.asarray(data["clean"]) if "clean" in data else None
                return np.asarray(data["alpha"], dtype=np.float32), clean
    return None, None


def enclosed_hole_mask(truth: np.ndarray) -> np.ndarray:
    """真值笔画完全包围的背景（字洞／字间空白）。"""
    import cv2
    bg = (~truth).astype(np.uint8)
    h, w = bg.shape
    # 外圈补一圈背景使所有触边背景连通，从角泛洪标记"外部"；
    # 剩余背景即不触边的包围空洞。O(像素)，避免逐连通域全图扫描。
    pad = np.ones((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = bg
    cv2.floodFill(pad, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)
    return pad[1:-1, 1:-1] == 1


def evaluate(mask_png: Path, alpha: np.ndarray, clean: np.ndarray | None,
             output: np.ndarray | None, source: np.ndarray | None) -> dict:
    selected = np.asarray(Image.open(mask_png).convert("L")) > 0
    if selected.shape != alpha.shape:
        # 输入降采样运行（如提示词模式的 WeVisOne 内存适配）时，把选区对齐到
        # 真值分辨率再评估；对齐只影响评估口径，不影响运行产物本身。
        selected = np.asarray(Image.fromarray(selected.astype(np.uint8) * 255).resize(
            (alpha.shape[1], alpha.shape[0]), Image.Resampling.NEAREST)) > 0
    if output is not None and clean is not None and output.shape != clean.shape:
        output = np.asarray(Image.fromarray(output.astype(np.uint8)).resize(
            (clean.shape[1], clean.shape[0]), Image.Resampling.LANCZOS)).astype(float)
    if source is not None and clean is not None and source.shape != clean.shape:
        source = np.asarray(Image.fromarray(source.astype(np.uint8)).resize(
            (clean.shape[1], clean.shape[0]), Image.Resampling.LANCZOS)).astype(float)
    truth = alpha > 0
    stroke = alpha >= 0.5
    inter = int((selected & truth).sum())
    truth_n = int(truth.sum())
    sel_n = int(selected.sum())
    holes = enclosed_hole_mask(truth)
    result = {
        "truth_pixels": truth_n,
        "selected_pixels": sel_n,
        "coverage": sel_n / selected.size,
        "recall": inter / truth_n if truth_n else None,
        "stroke_recall": (selected & stroke).sum() / stroke.sum() if stroke.sum() else None,
        "precision": inter / sel_n if sel_n else None,
        "missed_pixels": int((truth & ~selected).sum()),
        "extra_pixels": int((selected & ~truth).sum()),
        "hole_extra_pixels": int((selected & holes).sum()),
    }
    if clean is not None and output is not None and source is not None:
        err_before = np.abs(source.astype(float) - clean.astype(float)).mean(2)
        err_after = np.abs(output.astype(float) - clean.astype(float)).mean(2)
        result["ref_error_before_in_truth"] = float(err_before[truth].mean())
        result["ref_error_after_in_truth"] = float(err_after[truth].mean())
        # 平均误差会被大面积干净区域稀释，小面积高对比残留照样肉眼可见；
        # 用真值区内误差仍超阈值的像素数抓这类残留。
        resid = truth & (err_after > 15)
        result["visible_resid_pixels"] = int(resid.sum())
        h, w = truth.shape
        border = np.zeros_like(truth)
        b = BORDER_WIDTH
        border[:b, :] = 1; border[-b:, :] = 1; border[:, :b] = 1; border[:, -b:] = 1
        result["border_resid_pixels"] = int((resid & border).sum())
        result["interior_resid_pixels"] = int((resid & ~border).sum())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="生成批次自动识别测试的自包含 HTML 报告")
    parser.add_argument("--run-dir", required=True, type=Path, help="run_dev_batch.py 产物目录")
    parser.add_argument("--dataset-json", type=Path, default=None, help="dataset.json（分组与格元标签）")
    parser.add_argument("--gt-dir", type=Path, default=None, help="含 *_gt.npz 的事后评估目录")
    parser.add_argument("--out", type=Path, default=None, help="输出 HTML 路径")
    parser.add_argument("--note", default=None, help="在报告口径区注入的说明段落（HTML 转义后插入）")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    stamp_raw = manifest.get("generated_at")
    run_time = (datetime.fromisoformat(stamp_raw) if stamp_raw
                else datetime.fromtimestamp((run_dir / "run.json").stat().st_mtime))
    stamp = run_time.strftime("%Y-%m-%d %H:%M")

    samples_meta = {}
    if args.dataset_json and args.dataset_json.is_file():
        for s in json.loads(args.dataset_json.read_text(encoding="utf-8")).get("samples", []):
            samples_meta[s["sample_id"]] = s

    code_now = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT / "wm").glob("*.py"))}
    code_frozen = manifest.get("code_snapshot", {})
    code_match = code_now == code_frozen

    pages_dir = run_dir / "pages"
    cards = []
    eval_rows = []
    for rec in manifest["records"]:
        index = rec["index"]
        folder = next(pages_dir.glob(f"{index + 1:02d}_*"), None)
        if folder is None:
            continue
        meta = samples_meta.get(rec["doc_id"], {})
        group = meta.get("group", "未分组")
        cell = meta.get("cell", "")
        imgs = {}
        for name in ("input", "output", "watermark", "mask"):
            p = folder / f"{name}.png"
            if p.is_file():
                imgs[name] = p
        verdict = rec.get("verdict", {})
        failures = "、".join(verdict.get("failures", [])) or "无"
        status = "运行失败" if "error" in rec else ("待复核" if verdict.get("review_needed") else ("通过" if not verdict.get("failures") else "未通过"))
        stage_ms = verdict.get("stage_ms", {}) or {}

        eval_data = None
        if args.gt_dir and (folder / "mask.png").is_file():
            alpha, clean = load_gt(args.gt_dir.resolve(), rec["doc_id"])
            if alpha is not None:
                output = (np.asarray(Image.open(folder / "output.png").convert("RGB"))
                          if (folder / "output.png").is_file() else None)
                source = (np.asarray(Image.open(folder / "input.png").convert("RGB"))
                          if (folder / "input.png").is_file() else None)
                eval_data = evaluate(folder / "mask.png", alpha, clean, output, source)

        cards.append(dict(rec=rec, group=group, cell=cell, imgs=imgs, status=status,
                          failures=failures, stage_ms=stage_ms, eval=eval_data,
                          folder=folder))
        if eval_data:
            eval_rows.append((rec, meta, eval_data))

    def img_figure(path: Path, label: str) -> str:
        fmt = "PNG" if path.stem == "mask" else "JPEG"
        return (f'<figure><img src="{embed(path, fmt)}" alt="{esc(label)}">'
                f'<figcaption>{esc(label)}</figcaption></figure>')

    params = manifest["gui_defaults"]
    thr = params["thresholds"]
    outcome_counts: dict[str, int] = {}
    for o in manifest.get("outcomes", []):
        outcome_counts[o] = outcome_counts.get(o, 0) + 1

    # 真值自动分档（有 gt 时无需肉眼即可区分好坏）：
    #   去除干净 = 修复后参考误差低、真值区可见残留像素少；
    #   修复加重 = 修复后误差上升；部分去除 = 有下降但残留仍明显；无真值 = 未分档。
    CLEAN_ERR_MAX = 8.0
    CLEAN_RESID_MAX = 3000

    def tier_of(ev) -> str:
        if not ev or ev.get("ref_error_after_in_truth") is None or ev.get("ref_error_before_in_truth") is None:
            return "未分档"
        if ev["ref_error_after_in_truth"] > ev["ref_error_before_in_truth"]:
            return "修复加重"
        if ev["ref_error_after_in_truth"] <= CLEAN_ERR_MAX and ev.get("visible_resid_pixels", 0) <= CLEAN_RESID_MAX:
            return "去除干净"
        return "部分去除"

    for c in cards:
        c["tier"] = tier_of(c["eval"])

    tiers: dict[str, list] = {}
    for c in cards:
        tiers.setdefault(c["tier"], []).append(c)
    tier_order = ["去除干净", "部分去除", "修复加重", "未分档"]
    good_n = sum(len(tiers.get(t, [])) for t in ("去除干净",))
    bad_n = sum(len(tiers.get(t, [])) for t in ("部分去除", "修复加重"))

    def sample_card(c) -> str:
        rec, ev = c["rec"], c["eval"]
        elapsed = rec.get("verdict", {}).get("elapsed_ms", 0)
        return f'''<article class="sample">
<h3>{esc(rec["doc_id"])} <small>{esc(c["group"])} · {esc(c["cell"])}</small> <span class="badge tier">{esc(c["tier"])}</span> <span class="badge { "fail" if c["status"] in ("运行失败", "未通过") else "warn" if c["status"] == "待复核" else "pass" }">{esc(c["status"])}（程序判定）</span></h3>
<div class="trio">
{img_figure(c["imgs"]["input"], "原图")}
{img_figure(c["imgs"]["output"], "结果图") if "output" in c["imgs"] else '<figure><div class="missing">无结果图</div><figcaption>结果图</figcaption></figure>'}
{img_figure(c["imgs"]["watermark"], "水印图（提取）") if "watermark" in c["imgs"] else '<figure><div class="missing">无水印图</div><figcaption>水印图（提取）</figcaption></figure>'}
</div>
<div class="evidence">{img_figure(c["imgs"]["mask"], "选区掩码（证据）") if "mask" in c["imgs"] else ""}</div>
<table class="metrics"><tbody>
<tr><th>处理方法</th><td>{esc(rec.get("method", "—"))}／{esc(rec.get("repair_mode", "—"))}</td><th>耗时</th><td>{elapsed / 1000:.2f} 秒</td></tr>
<tr><th>选中像素</th><td>{rec.get("mask_pixels", 0):,}</td><th>{term_html("coverage", "选区覆盖率")}</th><td>{fmt_pct(rec.get("mask_coverage"))}</td></tr>
<tr><th>结果改动像素</th><td>{rec.get("changed_pixels", 0):,}</td><th>{term_html("outside_change", "选区外改动")}</th><td>{rec.get("changed_outside_mask_pixels", 0):,}</td></tr>
<tr><th>判定未过项</th><td colspan="3">{esc(c["failures"])}</td></tr>
</tbody></table>
</article>'''

    def eval_table(members) -> str:
        rows = []
        for c in members:
            ev = c["eval"]
            if not ev:
                continue
            rec = c["rec"]
            err_b = ev.get("ref_error_before_in_truth")
            err_a = ev.get("ref_error_after_in_truth")
            err_txt = ("—" if err_b is None or err_a is None else f"{err_b:.1f}→{err_a:.1f}")
            resid = ev.get("visible_resid_pixels")
            rows.append(f'''<tr><td>{esc(rec["doc_id"])}</td><td>{esc(c["group"])} · {esc(c["cell"])}</td>
<td>{fmt_pct(ev["recall"])}</td><td>{fmt_pct(ev["stroke_recall"])}</td><td>{fmt_pct(ev["precision"])}</td>
<td>{ev["missed_pixels"]:,}</td><td>{ev["extra_pixels"]:,}</td><td>{ev["hole_extra_pixels"]:,}</td>
<td>{err_txt}</td><td>{resid:,}</td><td>{ev.get("border_resid_pixels", 0):,}</td><td>{ev.get("interior_resid_pixels", 0):,}</td></tr>''')
        return f'''<table class="eval"><thead><tr><th>样本</th><th>组 · 格元</th><th>{term_html("recall", "召回率 α>0")}</th><th>{term_html("stroke_recall", "明显笔画召回率")}</th><th>{term_html("precision", "精确率")}</th><th>漏选像素</th><th>多选像素</th><th>字洞误选</th><th>{term_html("ref_error", "参考误差前→后")}</th><th>{term_html("resid", "可见残留像素")}</th><th>边框残留</th><th>内部残留</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'''

    tier_titles = {
        "去除干净": "效果较好：去除干净",
        "部分去除": "问题较多：部分去除仍有残留",
        "修复加重": "问题较多：修复后水印加重",
        "未分档": "未分档（无真值评估）",
    }
    group_sections = []
    for tier in tier_order:
        members = tiers.get(tier)
        if not members:
            continue
        group_sections.append(f'''<section><h2>{esc(tier_titles[tier])}（{len(members)} 张）</h2>{"".join(sample_card(c) for c in members)}{eval_table(members)}</section>''')

    eval_note = ""
    if eval_rows:
        means = {k: float(np.mean([r[2][k] for r in eval_rows if r[2][k] is not None])) for k in ("recall", "stroke_recall", "precision")}
        eval_note = f'''<p>真值评估自动分档（有真值即可机器分档，无需肉眼；仅作事后分档，不代表独立泛化）：按「{term_html("ref_error", "参考误差前→后")}」变化分档——修复后下降且绝对值 ≤ {CLEAN_ERR_MAX:.0f} 记「去除干净」，上升记「修复加重」，下降但残留明显记「部分去除」。本批：去除干净 {good_n} 张、问题较多 {bad_n} 张。均值：{term_html("recall", "召回率")} {fmt_pct(means["recall"])}；{term_html("stroke_recall", "明显笔画召回率")} {fmt_pct(means["stroke_recall"])}；{term_html("precision", "精确率")} {fmt_pct(means["precision"])}。「字洞误选」为选区落在真值笔画完全包围的背景内的像素数。</p>'''

    problems = [c for c in cards if c["status"] in ("运行失败", "未通过") or c["rec"].get("changed_outside_mask_pixels", 0) > 0]
    problem_list = "".join(f'<li>{esc(c["rec"]["doc_id"])}：{esc(c["status"])}；选区外改动 {c["rec"].get("changed_outside_mask_pixels", 0):,} 像素；{esc(c["failures"])}</li>' for c in problems)
    eval_problems = []
    for c in cards:
        if c["tier"] in ("修复加重", "部分去除"):
            ev = c["eval"]
            detail = []
            if ev.get("ref_error_after_in_truth") is not None and ev.get("ref_error_before_in_truth") is not None:
                detail.append(f"参考误差 {ev['ref_error_before_in_truth']:.1f}→{ev['ref_error_after_in_truth']:.1f}")
            if ev.get("precision") is not None:
                detail.append(f"精确率 {ev['precision'] * 100:.1f}%")
            if ev.get("recall") is not None:
                detail.append(f"召回率 {ev['recall'] * 100:.1f}%")
            eval_problems.append(f"<li>{esc(c['rec']['doc_id'])}：{'；'.join(detail)}。</li>")
    if eval_problems:
        problem_list += ("<li>真值评估分档为「修复加重／部分去除」的样本：</li>" + "".join(eval_problems))
    if not problem_list:
        problem_list = "<li>本轮没有运行失败、判定未过、选区外改动非零或真值评估异常的样本。</li>"

    condition_note = ""
    try:
        first = manifest["records"][0]
        fdir = next(pages_dir.glob(f"{first['index'] + 1:02d}_*"))
        fin = Image.open(fdir / "input.png")
        g0 = np.load(args.gt_dir.resolve() / f"{first['doc_id']}_gt.npz") if args.gt_dir else None
        if g0 is not None and fin.size != (g0["alpha"].shape[1], g0["alpha"].shape[0]):
            condition_note = (f'<p class="notice"><strong>运行条件：</strong>本轮输入为降采样副本'
                         f'（{fin.size[0]}×{fin.size[1]}，原图 {g0["alpha"].shape[1]}×{g0["alpha"].shape[0]}），'
                         f'用于适配提示词模型服务的内存上限；真值评估已对齐到原图分辨率，绝对值与全分辨率轮次'
                         f'比较时需注意口径。</p>')
    except Exception:
        eval_note = ""
    if args.note:
        condition_note += f'<p class="notice"><strong>说明：</strong>{esc(args.note)}</p>'

    mode_label = "提示词定位" if manifest.get("detection_mode") == "prompt" else "自动识别"
    if manifest.get("prompt_query"):
        mode_label += f"（{manifest['prompt_query']}）"
    mode_tag = "提示词定位" if manifest.get("detection_mode") == "prompt" else "自动识别"
    terms_js = json.dumps(TERMS, ensure_ascii=False)
    title = f"批次{mode_tag}测试报告 {run_time.strftime('%Y-%m-%d %H:%M')}"
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>
body{{font-family:"Microsoft YaHei","PingFang SC",sans-serif;margin:0;background:#f5f6f8;color:#222}}main{{max-width:1280px;margin:0 auto;padding:24px}}h1{{font-size:24px}}h2{{font-size:20px;border-left:4px solid #3a6ea5;padding-left:8px;margin-top:36px}}h3{{font-size:16px}}.lead{{color:#555}}.badge{{font-size:12px;padding:2px 8px;border-radius:10px;margin-left:8px}}.badge.pass{{background:#dcf5e3;color:#16713a}}.badge.warn{{background:#fff3cd;color:#8a6d00}}.badge.fail{{background:#f8d7da;color:#a12622}}.trio{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.evidence{{margin-top:10px;max-width:320px}}figure{{margin:0;background:#fff;border:1px solid #ddd;border-radius:6px;padding:6px}}figure img{{width:100%;display:block}}figcaption{{font-size:12px;color:#666;text-align:center;margin-top:4px}}.missing{{height:180px;display:flex;align-items:center;justify-content:center;color:#999}}table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}}th,td{{border:1px solid #ddd;padding:5px 8px;text-align:left}}th{{background:#eef1f5}}.metrics th{{width:120px}}.eval th{{background:#e8eef7}}.notice{{background:#fff8e6;border:1px solid #f0dfae;padding:10px 14px;border-radius:6px;margin:14px 0}}.term{{border-bottom:1px dashed #3a6ea5;cursor:help}}#term-pop{{position:fixed;display:none;max-width:360px;background:#222;color:#fff;font-size:13px;padding:10px 12px;border-radius:8px;z-index:99;line-height:1.6}}.stats{{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}}.stats div{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 18px}}.stats b{{font-size:20px;display:block}}.badge.tier{{background:#e3ecfb;color:#1d4f91;font-weight:600}}small{{color:#777}}section{{margin-bottom:12px}}</style></head><body><main>
<h1>{esc(title)}</h1>
<p class="lead">批次「{esc(manifest["batch_name"])}」共 {manifest["total_pages"]} 张，{term_html("auto_mode", "自动识别模式")}，采用桌面 GUI 默认参数一轮自动化测试。每张按「原图 / 结果图 / 水印图」一组展示，掩码为定位证据。</p>
<div class="stats">
<div><b>{manifest["total_pages"]}</b><span>输入图片</span></div>
<div><b>{good_n}</b><span>去除干净（真值分档）</span></div>
<div><b>{bad_n}</b><span>问题较多（真值分档）</span></div>
<div><b>{outcome_counts.get("待复核", 0)}</b><span>{term_html("review", "待复核")}（程序无参考判定）</span></div>
<div><b>{outcome_counts.get("未通过", 0) + outcome_counts.get("运行失败", 0)}</b><span>未通过／失败</span></div>
<div><b>{manifest["elapsed_seconds"]:.1f}s</b><span>整批耗时</span></div>
</div>
<div class="notice"><strong>口径与边界：</strong>推理只读取输入图片，不读取干净参考、alpha 真值或合成参数；{term_html("crosspage", "跨页先验")}只作搜索范围；{term_html("recall_priority", "笔画补全")}始终开启。下列评估指标（如有）在全部推理产物保存后才加载真值计算，仅用于效果分档与逐轮回归，不构成独立泛化验证。结果可追溯到输入哈希、阈值、模式与代码快照（当前源码与运行快照{"一致" if code_match else "不一致"}）。</div>
<h2>运行参数（GUI 默认值）</h2>
<table><tbody>
<tr><th>水印判定阈值</th><td>{thr["mask_alpha_min"]}</td><th>笔画连接</th><td>{params["close_px"]} px</td></tr>
<tr><th>修复参考范围</th><td>{thr["lama_context_margin"]} px</td><th>分块重叠</th><td>{thr["lama_tile_overlap"]} px</td></tr>
<tr><th>选区外扩</th><td>{thr["lama_mask_dilate"]} px</td><th>文字识别</th><td>开启</td></tr>
<tr><th>找水印方式</th><td>{mode_label}</td><th>样式路由</th><td>自动</td></tr>
</tbody></table>
{eval_note}
{condition_note}
{"".join(group_sections)}
<h2>需要关注的问题</h2><ul>{problem_list}</ul>
<p class="lead" style="font-size:13px">生成：tools/{Path(__file__).name}；运行清单 run.json；代码快照 {len(code_frozen)} 个文件；LaMa 模型 {manifest.get("lama_model_sha256", "")[:16]}…。同名批次的后续轮次使用新目录，不复用本目录。</p>
</main><div id="term-pop"></div><script>
const TERMS={terms_js};const pop=document.getElementById("term-pop");
document.querySelectorAll(".term").forEach(el=>{{
el.addEventListener("click",e=>{{e.stopPropagation();
if(pop.style.display==="block"&&pop.dataset.term===el.dataset.term){{pop.style.display="none";return;}}
pop.textContent=TERMS[el.dataset.term]||"";pop.dataset.term=el.dataset.term;
pop.style.display="block";pop.style.left=Math.min(e.clientX,innerWidth-380)+"px";pop.style.top=(e.clientY+14)+"px";}});}});
document.addEventListener("click",()=>pop.style.display="none");
document.addEventListener("keydown",e=>{{if(e.key==="Escape")pop.style.display="none";}});
</script></body></html>'''

    mode_file_tag = "提示词定位" if manifest.get("detection_mode") == "prompt" else "自动识别"
    out = args.out or (run_dir / f"批次{mode_file_tag}测试报告_{run_time.strftime('%Y%m%d_%H%M')}.html")
    out.write_text(page, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
