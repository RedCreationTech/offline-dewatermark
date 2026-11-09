"""Synthetic document-with-watermark dataset with ground truth.

The client supplied no samples, so this is the only way to get measurable
numbers this week. Every page ships with its clean original, the per-pixel
alpha map and the veil parameters, so mask quality and restoration quality are
both directly computable.
"""
from __future__ import annotations

import io
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# The corpus is only as faithful to a real tender document as the fonts it can
# find on this machine, so these are looked up rather than hard-coded.
_FONT_ROOTS = [p for p in [
    os.environ.get("WM_FONTS_DIR"),
    "C:/Windows/Fonts",
    "/usr/share/fonts",
    "/Library/Fonts",
    "/System/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    str(Path(__file__).with_name("fonts")),
] if p]

_FONT_FILES = {
    "body": ["simsun.ttc", "NotoSerifCJKsc-Regular.otf", "SourceHanSerifSC-Regular.otf",
             "wqy-zenhei.ttc", "Songti.ttc"],
    "head": ["msyhbd.ttc", "NotoSansCJKsc-Bold.otf", "SourceHanSansSC-Bold.otf",
             "wqy-microhei.ttc", "PingFang.ttc"],
    "hei": ["simhei.ttf", "NotoSansCJKsc-Regular.otf", "SourceHanSansSC-Regular.otf",
            "wqy-zenhei.ttc", "Hiragino Sans GB.ttc"],
}

_font_cache: dict[str, str] = {}


def _resolve_font(kind: str) -> str:
    if kind not in _font_cache:
        for root in _FONT_ROOTS:
            base = Path(root)
            for name in _FONT_FILES[kind]:
                # Linux ships fonts nested by vendor, Windows keeps them flat.
                found = [base / name] if (base / name).exists() else \
                    list(base.rglob(name)) if os.name == "posix" else []
                if found:
                    _font_cache[kind] = str(found[0])
                    break
            if kind in _font_cache:
                break
        if kind not in _font_cache:
            # Falling back to PIL's default bitmap font would draw every Han
            # glyph as an empty box, and the mask metrics would then measure a
            # corpus that contains no text. Refuse loudly instead.
            raise FileNotFoundError(
                f"未找到中文字体（kind={kind}），已尝试目录：{_FONT_ROOTS}。"
                f"请安装黑体/宋体，或设置环境变量 WM_FONTS_DIR 指向字体目录。")
    return _font_cache[kind]

_SENTENCES = [
    "本项目围绕离线智能去水印技术开展研究，重点解决复杂样式水印的定位与还原问题。",
    "系统在工作站本地部署模型算法，全流程不依赖外部网络环境，保证数据不出机房。",
    "测试样本覆盖字母、数字、点状图案以及上述元素的多种组合形式，并按样式分层统计。",
    "文档版面包含正文、表格虚线、印章区域与照片纹理区域，用于评估误删风险。",
    "处理结果需要同时满足整体风格不变、水印区域不影响内容识别、非水印区域保持一致。",
    "批处理采用流式调度，不对单批次的页数与张数设置上限，支持断点续跑。",
    "对识别置信度不足的页面自动路由至人工复核环节，避免生成式修复引入错字。",
    "指标验证以逐页审计日志为依据，记录所选路径、耗时与各项判定结果。",
]

_TABLE_ROWS = [
    ("WM-201", "跨页统计聚合掩码", "P0", "0.92"),
    ("WM-209", "内容保护白名单", "P0", "0.88"),
    ("WM-403", "字形引导重建", "P0", "0.75"),
    ("WM-503", "非水印区保真校验", "P0", "0.99"),
    ("WM-705", "显存与并发调度", "P0", "0.61"),
]

_AMOUNTS = [
    "合同金额：人民币 1,286,750.00 元",
    "编号：ZG-2026-7009111111 ／ 版本：V3.14",
    "日期：2026-10-19  校验码：8F3A-71C0",
    "数量 60 套，单价 21,445.83 元，折扣率 12.5%",
]

WATERMARK_STYLES = (
    "tiled_text",
    "big_center",
    "dot_grid",
    "corner_id",
    "heavy_dark",
)


def _font(kind: str, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_resolve_font(kind), px)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int, max_lines: int):
    lines: list[str] = []
    current = ""
    for ch in text:
        if draw.textlength(current + ch, font=font) > width:
            lines.append(current)
            current = ch
            if len(lines) >= max_lines:
                break
        else:
            current += ch
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines


def render_clean_page(rng: random.Random, dpi: int, page_index: int,
                      bg_type: str = "document") -> np.ndarray:
    """A4 page rendered at dpi.

    bg_type:
      - "white":    plain white page (minimal content)
      - "photo":    photo-like texture (gradient + noise, no text)
      - "document": full document with text, tables, seal, photo block (default)
    """
    scale = dpi / 72.0
    w = int(round(8.27 * scale * 72))
    h = int(round(11.69 * scale * 72))

    if bg_type == "white":
        return np.full((h, w, 3), 255, dtype=np.uint8)

    if bg_type == "photo":
        grad = np.tile(np.linspace(80, 200, w, dtype=np.float32), (h, 1))
        noise = np.array([rng.randint(0, 50) for _ in range(h * w)], dtype=np.float32).reshape(h, w)
        grad += noise
        photo = np.repeat(grad[..., None], 3, axis=2)
        photo = np.clip(photo, 0, 255).astype(np.uint8)
        img = Image.fromarray(photo)
        img = img.filter(ImageFilter.GaussianBlur(2.0))
        return np.asarray(img)

    # bg_type == "document" (full document with text, tables, seal, photo block)
    scale = dpi / 72.0
    w = int(round(8.27 * scale * 72))
    h = int(round(11.69 * scale * 72))
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin = int(0.9 * scale * 72)
    content_w = w - 2 * margin
    y = margin

    title = f"离线智能去水印技术研究报告 · 第 {page_index + 1} 页"
    f_title = _font("head", int(18 * scale))
    draw.text((margin, y), title, fill=(20, 20, 20), font=f_title)
    y += int(34 * scale)

    f_body = _font("body", int(11.5 * scale))
    for sentence in _SENTENCES[: 3 + (page_index % 3)]:
        for line in _wrap(draw, sentence, f_body, content_w, 3):
            draw.text((margin, y), line, fill=(25, 25, 25), font=f_body)
            y += int(19 * scale)
        y += int(8 * scale)

    f_mono = _font("hei", int(11 * scale))
    for amount in rng.sample(_AMOUNTS, 2):
        draw.text((margin, y), amount, fill=(15, 15, 15), font=f_mono)
        y += int(20 * scale)
    y += int(10 * scale)

    col_x = [margin + (content_w * i) // 4 for i in range(4)]
    row_h = int(22 * scale)
    f_tab = _font("body", int(10.5 * scale))
    headers = ("编号", "功能项", "优先级", "权重")
    draw.text((col_x[0] + 6, y), headers[0], fill=(30, 30, 30), font=f_tab)
    draw.text((col_x[1] + 6, y), headers[1], fill=(30, 30, 30), font=f_tab)
    draw.text((col_x[2] + 6, y), headers[2], fill=(30, 30, 30), font=f_tab)
    draw.text((col_x[3] + 6, y), headers[3], fill=(30, 30, 30), font=f_tab)
    y += row_h
    for row in _TABLE_ROWS:
        for i, cell in enumerate(row):
            draw.text((col_x[i] + 6, y), cell, fill=(35, 35, 35), font=f_tab)
        # dotted rules are legitimate content and must never be removed
        for dx in range(col_x[0], w - margin, 8):
            draw.point((dx, y + row_h - 2), fill=(150, 150, 150))
        y += row_h
    y += int(16 * scale)

    for sentence in _SENTENCES[3:6]:
        for line in _wrap(draw, sentence, f_body, content_w, 3):
            draw.text((margin, y), line, fill=(25, 25, 25), font=f_body)
            y += int(19 * scale)
        y += int(8 * scale)

    ph = int(90 * scale)
    grad = np.tile(np.linspace(60, 210, content_w, dtype=np.float32), (ph, 1))
    grad += np.array([rng.randint(0, 40) for _ in range(ph)], dtype=np.float32)[:, None]
    block = Image.fromarray(np.repeat(grad[..., None], 3, axis=2).astype(np.uint8))
    block = block.filter(ImageFilter.GaussianBlur(1.2))
    img.paste(block, (margin, min(y, h - ph - margin)))
    y += ph + int(14 * scale)

    seal_cx, seal_cy = w - margin - int(60 * scale), min(y + int(40 * scale), h - int(90 * scale))
    r = int(48 * scale)
    draw.ellipse(
        [seal_cx - r, seal_cy - r, seal_cx + r, seal_cy + r],
        outline=(178, 34, 34), width=max(2, int(3 * scale)),
    )
    f_seal = _font("hei", int(12 * scale))
    draw.text((seal_cx - int(30 * scale), seal_cy - int(8 * scale)), "专用章",
              fill=(178, 34, 34), font=f_seal)

    draw.text((margin, h - int(40 * scale)), f"— {page_index + 1} —",
              fill=(90, 90, 90), font=f_body)
    return np.asarray(img)


OPACITY_FACTORS = {"low": 0.5, "medium": 1.0, "high": 1.5}
BG_TYPES = ("white", "photo", "document")


def _text_alpha(shape, text: str, font_px: int, spacing_x: int, spacing_y: int,
                angle: float, rng: random.Random,
                anchor: tuple[int, int] | None = None) -> np.ndarray:
    """Coverage map of one text run — tiled grid, page-centred, or anchored."""
    layer = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(layer)
    font = _font("hei", font_px)
    tw = draw.textlength(text, font=font)
    if spacing_x <= 0 or spacing_y <= 0:
        cx, cy = anchor or (shape[1] // 2, shape[0] // 2)
        draw.text((cx - tw / 2, cy - font_px / 2), text, fill=255, font=font)
    else:
        for ty in range(-spacing_y, shape[0] + spacing_y, spacing_y):
            for tx in range(-int(tw) - spacing_x, shape[1] + spacing_x, spacing_x):
                draw.text((tx, ty), text, fill=255, font=font)
    if angle:
        layer = layer.rotate(angle, resample=Image.BICUBIC, expand=False)
    return np.asarray(layer, dtype=np.float32) / 255.0


def _dot_alpha(shape, period: int, radius: float, rng: random.Random,
               jitter: bool = True) -> np.ndarray:
    layer = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(layer)
    for gy in range(0, shape[0] + period, period):
        for gx in range(0, shape[1] + period, period):
            ox = rng.randint(-1, 1) if jitter else 0
            oy = rng.randint(-1, 1) if jitter else 0
            cx, cy = gx + ox, gy + oy
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)
    return np.asarray(layer, dtype=np.float32) / 255.0


def make_watermark(shape, style: str, dpi: int, page_index: int,
                   rng: random.Random,
                   opacity_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (alpha_map, watermark_rgb_color, strength).

    opacity_factor scales the base strength: low=0.5, medium=1.0, high=1.5.
    """
    scale = dpi / 200.0
    if style == "tiled_text":
        alpha = _text_alpha(shape, "内部资料 CONF-2026", int(20 * scale), int(340 * scale),
                            int(230 * scale), -30, rng)
        color = np.array([120.0, 120.0, 120.0])
        strength = 0.45
    elif style == "big_center":
        alpha = _text_alpha(shape, "机密 T7009", int(120 * scale), 0, 0, -35, rng)
        color = np.array([90.0, 90.0, 110.0])
        strength = 0.55
    elif style == "dot_grid":
        alpha = _dot_alpha(shape, max(10, int(18 * scale)), max(1.0, 2.2 * scale), rng,
                           jitter=False)
        color = np.array([140.0, 140.0, 150.0])
        strength = 0.55
    elif style == "corner_id":
        alpha = _text_alpha(shape, f"ZG-{page_index + 1:03d}-8F3A", int(16 * scale), 0, 0, 0,
                            rng, anchor=(shape[1] - int(160 * scale), shape[0] - int(60 * scale)))
        color = np.array([70.0, 70.0, 70.0])
        strength = 0.6
    elif style == "heavy_dark":
        alpha = _text_alpha(shape, "禁止复制 NO-COPY", int(24 * scale), int(380 * scale),
                            int(250 * scale), -28, rng)
        color = np.array([40.0, 40.0, 40.0])
        strength = 0.8
    else:
        raise ValueError(f"unknown style {style}")
    adjusted_strength = float(np.clip(strength * opacity_factor, 0.05, 0.95))
    return alpha.astype(np.float32), color.astype(np.float32), adjusted_strength


def apply_watermark(clean: np.ndarray, alpha: np.ndarray, color: np.ndarray,
                   strength: float) -> np.ndarray:
    a = np.clip(alpha * strength, 0.0, 1.0)[..., None]
    out = a * color[None, None, :] + (1.0 - a) * clean.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade(img: np.ndarray, rng: random.Random, jpeg_q: int = 0,
            noise: int = 0) -> np.ndarray:
    out = img
    if jpeg_q:
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, "JPEG", quality=jpeg_q)
        out = np.asarray(Image.open(buf).convert("RGB"))
    if noise:
        gen = np.random.default_rng(rng.randrange(2 ** 32))
        out = np.clip(
            out.astype(np.int16) + gen.integers(-noise, noise + 1, out.shape),
            0, 255,
        ).astype(np.uint8)
    return out


def generate_dataset(root: str | Path, docs: int = 3, pages_per_doc: int = 6,
                     dpi: int = 200, styles: tuple[str, ...] = WATERMARK_STYLES,
                     jpeg_q: int = 0, noise: int = 0, seed: int = 20261019) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    for d in range(docs):
        rng = random.Random(seed + d)
        style = styles[d % len(styles)]
        doc_id = f"doc{d:02d}_{style}"
        doc_dir = root / doc_id
        doc_dir.mkdir(exist_ok=True)
        for p in range(pages_per_doc):
            clean = render_clean_page(rng, dpi, p)
            alpha, color, strength = make_watermark(clean.shape, style, dpi, p, rng)
            watermarked = apply_watermark(clean, alpha, color, strength)
            if jpeg_q or noise:
                watermarked = degrade(watermarked, rng, jpeg_q, noise)
            np.savez_compressed(
                doc_dir / f"page{p:02d}_gt.npz",
                clean=clean, alpha=alpha, color=color, strength=np.float32(strength),
            )
            Image.fromarray(watermarked).save(doc_dir / f"page{p:02d}.png")
            index.append({
                "doc_id": doc_id, "page_index": p, "style": style, "dpi": dpi,
                "degraded": bool(jpeg_q or noise),
            })
    meta = {
        "seed": seed, "dpi": dpi, "docs": docs, "pages_per_doc": pages_per_doc,
        "styles": list(styles), "jpeg_q": jpeg_q, "noise": noise,
        "note": "同一文档内水印样式/颜色/强度恒定，正文逐页不同（跨页统计先验的前提）",
        "pages": index,
    }
    (root / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return root


def generate_stratified_dataset(root: str | Path, dpi: int = 200,
                                styles: tuple[str, ...] = WATERMARK_STYLES,
                                opacity_levels: tuple[str, ...] = ("low", "medium", "high"),
                                bg_types: tuple[str, ...] = BG_TYPES,
                                jpeg_q: int = 0, noise: int = 0,
                                seed: int = 20261019) -> Path:
    """Generate a stratified test set: styles × opacity × bg_type.

    Default: 5 styles × 3 opacity × 3 bg_types = 45 samples.
    Each sample has a unique combination, with ground truth (clean, alpha, color, strength).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    rng = random.Random(seed)
    sample_id = 0

    for style in styles:
        for opacity_name in opacity_levels:
            opacity_factor = OPACITY_FACTORS[opacity_name]
            for bg_type in bg_types:
                clean = render_clean_page(rng, dpi, sample_id, bg_type=bg_type)
                alpha, color, strength = make_watermark(
                    clean.shape, style, dpi, sample_id, rng, opacity_factor=opacity_factor
                )
                watermarked = apply_watermark(clean, alpha, color, strength)
                if jpeg_q or noise:
                    watermarked = degrade(watermarked, rng, jpeg_q, noise)

                sid = f"s{sample_id:03d}_{style}_{opacity_name}_{bg_type}"
                np.savez_compressed(
                    root / f"{sid}_gt.npz",
                    clean=clean, alpha=alpha, color=color,
                    strength=np.float32(strength),
                )
                Image.fromarray(watermarked).save(root / f"{sid}.png")
                index.append({
                    "sample_id": sid, "style": style,
                    "opacity_level": opacity_name, "opacity_factor": opacity_factor,
                    "bg_type": bg_type, "strength": round(strength, 3),
                    "dpi": dpi, "degraded": bool(jpeg_q or noise),
                })
                sample_id += 1

    meta = {
        "type": "stratified",
        "seed": seed, "dpi": dpi,
        "styles": list(styles),
        "opacity_levels": {name: OPACITY_FACTORS[name] for name in opacity_levels},
        "bg_types": list(bg_types),
        "jpeg_q": jpeg_q, "noise": noise,
        "total_samples": sample_id,
        "note": "分层测试集：样式 × 透明度 × 底图类型全组合，用于能力边界评估",
        "samples": index,
    }
    (root / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return root


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="合成水印测试集生成器",
        epilog=f"可用样式：{', '.join(WATERMARK_STYLES)}；底图类型：{', '.join(BG_TYPES)}",
    )
    ap.add_argument("--out", type=Path, default=Path("out/合成测试集"),
                    help="输出目录（默认：out/合成测试集）")
    ap.add_argument("--stratified", action="store_true",
                    help="生成分层测试集（样式 × 透明度 × 底图类型全组合）")
    ap.add_argument("--docs", type=int, default=3,
                    help="文档数量，每种样式至少一个文档（默认：3，仅非分层模式）")
    ap.add_argument("--pages", type=int, default=6,
                    help="每文档页数（默认：6，仅非分层模式）")
    ap.add_argument("--dpi", type=int, default=200,
                    help="渲染 DPI（默认：200）")
    ap.add_argument("--styles", nargs="+", default=list(WATERMARK_STYLES),
                    choices=WATERMARK_STYLES,
                    help="使用的水印样式列表（默认：全部 5 种）")
    ap.add_argument("--opacity-levels", nargs="+", default=["low", "medium", "high"],
                    choices=list(OPACITY_FACTORS.keys()),
                    help="透明度档位（默认：low medium high，仅分层模式）")
    ap.add_argument("--bg-types", nargs="+", default=list(BG_TYPES),
                    choices=BG_TYPES,
                    help="底图类型（默认：white photo document，仅分层模式）")
    ap.add_argument("--jpeg-q", type=int, default=0,
                    help="JPEG 压缩质量，0 = 不压缩（默认：0）")
    ap.add_argument("--noise", type=int, default=0,
                    help="高斯噪声幅度，0 = 不加噪（默认：0）")
    ap.add_argument("--seed", type=int, default=20261019,
                    help="随机种子（默认：20261019）")
    args = ap.parse_args()

    if args.stratified:
        n_styles = len(args.styles)
        n_opacity = len(args.opacity_levels)
        n_bg = len(args.bg_types)
        total = n_styles * n_opacity * n_bg
        result = generate_stratified_dataset(
            root=args.out,
            dpi=args.dpi,
            styles=tuple(args.styles),
            opacity_levels=tuple(args.opacity_levels),
            bg_types=tuple(args.bg_types),
            jpeg_q=args.jpeg_q,
            noise=args.noise,
            seed=args.seed,
        )
        print(f"已生成分层测试集：{n_styles} 样式 × {n_opacity} 透明度 × {n_bg} 底图 = {total} 个样本")
    else:
        result = generate_dataset(
            root=args.out,
            docs=args.docs,
            pages_per_doc=args.pages,
            dpi=args.dpi,
            styles=tuple(args.styles),
            jpeg_q=args.jpeg_q,
            noise=args.noise,
            seed=args.seed,
        )
        print(f"已生成 {args.docs} 个文档 × {args.pages} 页 = {args.docs * args.pages} 个样本")
    print(f"输出目录：{result}")
    print(f"元数据：{result / 'dataset.json'}")
