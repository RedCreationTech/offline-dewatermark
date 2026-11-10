"""生成斜向平铺分层合成样本，并通过固定控制模板生成逐页报告。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.regression_guard import canonical, digest
from wm.synth import BG_TYPES, _resolve_font, apply_watermark, degrade, render_clean_page

STYLE = "tiled_text"
TEXT = "内部资料 08812 2026-11-10"
ANGLE_DEG = -35.0
COL_MULT = 1.35
ROW_MULT = 4.5
STAGGER = 0.5
BASE_PX = 56
BASE_STRENGTH = 0.30

COLOR_RGB = {"gray": (120.0, 120.0, 120.0), "dark": (40.0, 40.0, 40.0),
             "red": (190.0, 60.0, 60.0), "blue": (60.0, 110.0, 190.0),
             "yellow": (230.0, 200.0, 80.0)}
OPACITY_LEVELS = [("faint", 0.12, "极淡12%", "极淡12"), ("base", 0.30, "基准30%", "透明度基准"),
                  ("mid", 0.50, "中50%", "中等50"), ("heavy", 0.70, "浓70%", "浓70")]
COLOR_LEVELS = [("base", "gray", "基准灰字", "颜色基准"), ("dark", "dark", "深黑字", "深黑字"),
                ("red", "red", "红字", "红字"), ("blue", "blue", "蓝字", "蓝字"),
                ("yellow", "yellow", "浅黄字", "浅黄字")]
SIZE_LEVELS = [("tiny", 0.50, "小字28px·10pt", "小字28"),
               ("small", 0.72, "小字40px·14pt", "小字40"),
               ("base", 1.0, "基准56px·20pt", "尺寸基准"),
               ("large", 1.43, "大字80px·29pt", "大字80")]
GROUP_LABEL = {"opacity": "透明度组", "color": "颜色组", "size": "尺寸组", "bg": "底图组"}
BG_LABEL = {"white": "纯白纸", "photo": "照片", "document": "文档"}
BASE = {"bg": "white", "strength": BASE_STRENGTH, "color": "gray", "size": 1.0}


def oblique_alpha(shape, px: int, dpi: int = 200) -> tuple[np.ndarray, dict]:
    """按真实平铺写法生成覆盖图：铺满对角平面 → 旋转 → 裁回页面。"""
    h, w = shape[:2]
    scale = dpi / 200.0
    px = int(round(px * scale))
    font = ImageFont.truetype(_resolve_font("hei"), px)
    probe = ImageDraw.Draw(Image.new("L", (10, 10)))
    tw = probe.textlength(TEXT, font=font)
    sx = int(round(tw * COL_MULT))
    sy = int(round(px * 1.15 * ROW_MULT))
    diag = int(math.hypot(w, h)) + 2 * px
    layer = Image.new("L", (diag, diag), 0)
    draw = ImageDraw.Draw(layer)
    row = 0
    for ty in range(-sy, 2 * diag, sy):
        off = int(sx * STAGGER) if row % 2 else 0
        row += 1
        for tx in range(-int(tw) - sx + off, 2 * diag, sx):
            draw.text((tx, ty), TEXT, fill=255, font=font)
    layer = layer.rotate(ANGLE_DEG, resample=Image.BICUBIC, expand=False)
    x0, y0 = (diag - w) // 2, (diag - h) // 2
    alpha = np.asarray(layer, dtype=np.uint8)[y0:y0 + h, x0:x0 + w]
    params = dict(font_px=px, text_width_px=round(tw), spacing_x_px=sx,
                  spacing_y_px=sy, angle_deg=ANGLE_DEG, stagger=STAGGER)
    return alpha.astype(np.float32) / 255.0, params


def load_bases(bg_levels: list[str], base_dir: Path, out_dir: Path, dpi: int,
               seed: int) -> dict[str, tuple[Path, np.ndarray]]:
    bases: dict[str, tuple[Path, np.ndarray]] = {}
    for bg in bg_levels:
        reused = sorted(base_dir.glob(f"*_{bg}_clean.png")) if base_dir.is_dir() else []
        if reused:
            path = reused[0]
            bases[bg] = (path, np.asarray(Image.open(path).convert("RGB")))
            continue
        own = out_dir / "_底图"
        own.mkdir(parents=True, exist_ok=True)
        path = own / f"base_{bg}_clean.png"
        if not path.is_file():
            clean = render_clean_page(random.Random(seed + BG_TYPES.index(bg)),
                                      dpi, 0, bg_type=bg)
            Image.fromarray(clean).save(path)
        bases[bg] = (path, np.asarray(Image.open(path).convert("RGB")))
    return bases


def plan(only: list[str], bgs: list[str], levels: list[str] | None = None) -> list[dict]:
    cells = []

    def push(group: str, key: str, name: str, cell: str, **over) -> None:
        if levels and key not in levels:
            return
        cells.append(dict(group=group, key=key, name=name, cell=cell,
                          level={**BASE, **over}))

    if "opacity" in only:
        for key, strength, cell, name in OPACITY_LEVELS:
            push("opacity", key, name, cell, strength=strength)
    if "color" in only:
        for key, color, cell, name in COLOR_LEVELS:
            push("color", key, name, cell, color=color)
    if "size" in only:
        for key, mult, cell, name in SIZE_LEVELS:
            push("size", key, name, cell, size=mult)
    if "bg" in only and len(bgs) > 1:
        for bg in bgs:
            push("bg", bg, bg, BG_LABEL.get(bg, bg), bg=bg)
    if not cells:
        raise SystemExit(f"--only 里没有可生成的组：{only}")
    return cells


def prepare_dataset(args: argparse.Namespace) -> None:
    cells = plan(args.only, args.bgs, args.levels)
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    bgs = list(dict.fromkeys(cell["level"]["bg"] for cell in cells))
    bases = load_bases(bgs, args.base_dir, out_dir, args.dpi, args.seed)

    samples = []
    for idx, c in enumerate(cells):
        level = c["level"]
        base_path, base = bases[level["bg"]]
        alpha, params = oblique_alpha(base.shape, BASE_PX * level["size"], args.dpi)
        rgb = np.array(COLOR_RGB[level["color"]], dtype=np.float64)
        strength = float(np.clip(level["strength"], 0.02, 0.95))
        watermarked = apply_watermark(base, alpha, rgb, strength)
        if args.jpeg_q or args.noise:
            watermarked = degrade(watermarked, random.Random(args.seed + idx),
                                  args.jpeg_q, args.noise)
        sid = f"s{idx:03d}_{GROUP_LABEL[c['group']]}_{c['name']}"
        np.savez_compressed(out_dir / f"{sid}_gt.npz",
                            clean=base, alpha=alpha.astype(np.float32),
                            color=rgb.astype(np.float32), strength=np.float32(strength))
        Image.fromarray(watermarked).save(out_dir / f"{sid}.png")
        n = 220
        corner = float(min((alpha[:n, :n] > 0.5).mean(), (alpha[:n, -n:] > 0.5).mean(),
                           (alpha[-n:, :n] > 0.5).mean(), (alpha[-n:, -n:] > 0.5).mean()))
        samples.append({
            "sample_id": sid, "style": STYLE, "group": GROUP_LABEL[c["group"]],
            "group_key": c["group"], "cell_key": c["key"], "cell": c["cell"],
            "bg_type": level["bg"],
            "input_sha256": digest(out_dir / f"{sid}.png"),
            "opacity_level": f"{strength:.2f}", "strength": round(strength, 3),
            "color_name": level["color"], "color_rgb": [float(v) for v in rgb],
            "size_factor": level["size"], "font_px": params["font_px"],
            "spacing_px": [params["spacing_x_px"], params["spacing_y_px"]],
            "angle_deg": params["angle_deg"], "stagger": params["stagger"],
            "coverage": round(float((alpha > 0.5).mean()), 4),
            "corner_coverage_min": round(corner, 4),
            "dpi": args.dpi, "degraded": bool(args.jpeg_q or args.noise),
            "base_file": str(base_path), "base_sha256": digest(base_path),
        })

    meta = {
        "type": "stratified", "grid": "oblique_tiled_ofat_v2",
        "seed": args.seed, "dpi": args.dpi, "styles": [STYLE],
        "handwriting": {"text": TEXT, "angle_deg": ANGLE_DEG, "col_mult": COL_MULT,
                        "row_mult": ROW_MULT, "stagger": STAGGER, "base_px": BASE_PX,
                        "base_strength": BASE_STRENGTH,
                        "note": "铺满对角平面后旋转裁回页面，四角与边缘含裁断残笔；"
                                "列距行距随字号等比，故尺寸组只变尺寸不变密度"},
        "groups": {GROUP_LABEL[g]: sum(1 for s in samples if s["group_key"] == g)
                   for g in dict.fromkeys(c["group"] for c in cells)},
        "bases": {bg: {"file": str(p), "sha256": digest(p)} for bg, (p, _) in bases.items()},
        "total_samples": len(samples),
        "note": "斜向平铺 OFAT 网格：每组自带基准，组间不交叉，底图逐格固定。"
                "带 alpha 真值，仅用于效果分档与逐轮回归，不能充当独立泛化验证；"
                "用于调参的格子即转为开发集。",
        "samples": samples,
    }
    (out_dir / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")

    print(f"生成 {len(samples)} 格：{out_dir}")
    base_cells = [s for s in samples if s["cell_key"] == "base"]
    print(f"底图唯一哈希 {len({s['base_sha256'] for s in samples})} 个 | "
          + (f"{len(base_cells)} 张基准格 png 哈希 {len({s['input_sha256'] for s in base_cells})} 个（应为 1）"
             if base_cells else "本批未含基准格（--levels 只出指定档），跳过基准一致性自检"))
    print(f"{'sample_id':34} {'组':7} {'格':14} {'strength':>8} {'px':>4} {'覆盖率':>7} {'四角最小':>8}")
    for s in samples:
        print(f"{s['sample_id']:34} {s['group']:7} {s['cell']:14} "
              f"{s['strength']:>8.2f} {s['font_px']:>4} {s['coverage'] * 100:>6.2f}% "
              f"{s['corner_coverage_min'] * 100:>7.2f}%")



STYLE_LABEL = {
    'tiled_text': '平铺文字水印',
    'big_center': '居中大号文字',
    'dot_grid': '点阵纹理',
    'corner_id': '小角标编号',
    'heavy_dark': '深色重文字',
}
LEGACY_BG_LABEL = {'white': '白纸', 'photo': '照片', 'document': '文档'}
OPACITY_LABEL = {'low': '低透明度', 'medium': '中透明度', 'high': '高透明度'}
COLOR_LABEL = {'gray': '灰字', 'dark': '深黑字', 'red': '红字', 'blue': '蓝字', 'yellow': '浅黄字'}


def case_label(sample: dict) -> str:
    """新网格按组＋格命名；v1 老数据集无这两个字段时保持原三段（必要时补颜色/字号）。"""
    style = STYLE_LABEL.get(sample['style'], sample['style'])
    if sample.get('cell') and sample.get('group'):
        return f'{style}·{sample["group"]}·{sample["cell"]}'
    parts = [style,
             OPACITY_LABEL.get(sample['opacity_level'], sample['opacity_level']),
             LEGACY_BG_LABEL.get(sample['bg_type'], sample['bg_type'])]
    if sample.get('color_name'):
        parts.append(COLOR_LABEL.get(sample['color_name'], sample['color_name']))
    if sample.get('font_px'):
        parts.append(f"{sample['font_px']}px")
    return '·'.join(parts)


def extract_references(dataset_dir: Path, samples: list[dict]) -> Path:
    """把每个样本的 _gt.npz 拆成控制管线要求的 clean/alpha 文件。"""
    ref_dir = dataset_dir / '参考'
    ref_dir.mkdir(exist_ok=True)
    for sample in samples:
        sid = sample['sample_id']
        clean_path = ref_dir / f'{sid}_clean.png'
        alpha_path = ref_dir / f'{sid}_alpha.npz'
        with np.load(dataset_dir / f'{sid}_gt.npz', allow_pickle=False) as gt:
            clean, alpha, strength = gt['clean'], gt['alpha'], gt['strength']
        if clean_path.is_file():
            with Image.open(clean_path) as image:
                if not np.array_equal(np.asarray(image.convert('RGB')), clean):
                    raise ValueError(f'已有底图参考与标注不一致：{clean_path}')
        else:
            Image.fromarray(clean).save(clean_path)
        if alpha_path.is_file():
            with np.load(alpha_path, allow_pickle=False) as reference:
                if not (np.array_equal(reference['alpha'], alpha)
                        and np.array_equal(reference['strength'], strength)):
                    raise ValueError(f'已有透明度参考与标注不一致：{alpha_path}')
        else:
            # 评分时将覆盖形状与强度相乘，沿用合成时的口径。
            np.savez_compressed(alpha_path, alpha=alpha, strength=strength)
    return ref_dir


def dataset_scope(meta: dict) -> dict:
    """从实际样本汇总格数、分组、底图和样式，并引用清单的设计说明。"""
    samples = meta['samples']
    groups: Counter = Counter(s.get('group') or '分层测试集' for s in samples)
    bases = {s.get('base_sha256') or s.get('bg_type') for s in samples}
    bg_names = '、'.join(sorted({LEGACY_BG_LABEL.get(s['bg_type'], s['bg_type'])
                                 for s in samples if s.get('bg_type')}))
    styles = '、'.join(sorted({STYLE_LABEL.get(s['style'], s['style']) for s in samples}))
    facts = f'本批 {len(samples)} 格：' + '／'.join(f'{name}{count}格' for name, count in groups.items())
    facts += f'；底图 {len(bases)} 张' + (f'（{bg_names}）' if bg_names else '')
    facts += f'；样式 {styles}'
    # 数据集自带的 note 会说"每组自带基准"，但 --levels 只补跑个别格时批里真没有基准格，得据实纠正
    if any('cell_key' in s for s in samples):
        n_base = sum(1 for s in samples if s.get('cell_key') == 'base')
        facts += (f'；基准格 {n_base} 个' if n_base else '；本批未含基准格（属子集补跑，"每组自带基准"对本批不适用）')
    caveat = meta.get('note') or ('合成能力边界样本；带 alpha 真值，仅用于效果分档与逐轮回归，'
                                  '不能充当独立泛化验证，也不参与防退化门槛。')
    return dict(groups=dict(groups), note=f'{facts}。{caveat}')


def build_inventory(dataset_dir: Path) -> dict:
    meta = json.loads((dataset_dir / 'dataset.json').read_text(encoding='utf-8'))
    samples = meta['samples']
    if not samples or len({s['sample_id'] for s in samples}) != len(samples):
        raise ValueError('分层清单为空或含重复样本 ID')
    ref_dir = extract_references(dataset_dir, samples)
    cases = []
    for sample in samples:
        sid = sample['sample_id']
        input_path = (dataset_dir / f'{sid}.png').resolve()
        clean_path = (ref_dir / f'{sid}_clean.png').resolve()
        alpha_path = (ref_dir / f'{sid}_alpha.npz').resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f'样本输入缺失：{input_path}')
        input_sha = digest(input_path)
        if sample.get('input_sha256') and sample['input_sha256'] != input_sha:
            raise ValueError(f'样本输入与清单哈希不符：{input_path}')
        clean_sha = digest(clean_path)
        refs = [dict(clean=str(clean_path), clean_sha256=clean_sha,
                     alpha=str(alpha_path), alpha_sha256=digest(alpha_path))]
        style = sample['style']
        group_name = sample.get('group') or '分层测试集'
        row = dict(
            id=sid, input=str(input_path), input_sha256=input_sha,
            batch=f"{group_name}/{style}", group=group_name,
            style=style,
            label=case_label(sample),
            negative=False, page_count=1, dpi=sample.get('dpi', 200),
            source_group=f'stratified_base_{clean_sha}',
            split_group=f'stratified_base_{clean_sha}',
            base_id=clean_sha, original_split='stratified_synthetic',
            references=refs, role='stratified_synthetic_capability',
        )
        row['reference_identity'] = canonical(refs)
        cases.append(row)
    return dict(
        schema_version=1, input_root=str(dataset_dir.resolve()), cases=cases,
        **dataset_scope(meta),
        total_inputs=len(cases), total_pages=sum(r['page_count'] for r in cases),
        unique_input_hashes=len({r['input_sha256'] for r in cases}),
        independent_validation=False,
    )


def run_control(args: argparse.Namespace) -> int:
    from tools import build_report_index
    from tools.regression_control import execute, prepare, stamp, verified_run

    inventory = build_inventory(args.dataset_dir.resolve())
    out = args.out_root.resolve() / f'{stamp()}_分层测试集_freeze'
    prepare(out, inventory, scope='stratified')
    print(f'已冻结：{out}', flush=True)
    execute(out)
    run = verified_run(out)
    reports = sorted(out.glob('项目防退化控制明细报告_*.html'))
    for report in reports:
        print(f'明细报告：{report}')
    if run['complete'] and reports and out.is_relative_to(build_report_index.OUT):
        build_report_index.main()
    failed = sum(r['status'] != 'completed' for r in run['records'])
    print(f'已记录 {len(run["records"])}/{len(run["expected_keys"])} 页，运行失败 {failed} 页')
    return 0 if run['complete'] and reports and not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    generator = commands.add_parser('prepare', help='生成单因素分层网格')
    generator.add_argument('--out', type=Path, required=True, help='本轮的新目录')
    generator.add_argument('--base-dir', type=Path, default=ROOT / 'out/分层测试集_v1/参考',
                           help='复用干净底图的目录，找不到时在本轮渲染')
    generator.add_argument('--only', nargs='+', default=['opacity', 'color', 'size'],
                           choices=['opacity', 'color', 'size', 'bg'])
    generator.add_argument('--levels', nargs='+', help='只生成指定档位键')
    generator.add_argument('--bgs', nargs='+', default=['white'], choices=BG_TYPES)
    generator.add_argument('--jpeg-q', type=int, default=0, help='JPEG 质量，0 表示不压缩')
    generator.add_argument('--noise', type=int, default=0)
    generator.add_argument('--dpi', type=int, default=200)
    generator.add_argument('--seed', type=int, default=20261110)
    control = commands.add_parser('run', help='冻结并运行已有分层样本')
    control.add_argument('--dataset-dir', type=Path, required=True)
    control.add_argument('--out-root', type=Path, default=ROOT / 'out/项目防退化控制')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_dataset(args)
        return 0
    return run_control(args)


if __name__ == '__main__':
    sys.exit(main())
