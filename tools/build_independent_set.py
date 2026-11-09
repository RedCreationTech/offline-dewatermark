"""Build the LOCKED independent validation set v1 (data/watermark_independent_v1/).

Purpose: a never-tuned-on evaluation set for watermark removal quality.
Built 2026-11-02, one-look baseline, then frozen.

Independence rules (AGENTS.md section 5):
- brand-new bases only: 12 newly downloaded CC0 photos (fetch_independent_bases.py,
  deduped against v1 by title/sha256/perceptual hash) + 6 new documents + 4 new
  charts + 4 new procedural textures, all generated with seed 20261102;
- watermark text/templates differ from v1 ('独立核验样本' tokens, new seed chain);
- cross-set checks vs v1 bases AND the 14 dev images are asserted before writing;
- exact 8-bit effective alpha + binary mask for every positive; negatives all-zero;
- split field is 'independent' everywhere; never to be merged into training splits.

This script is an experiment-grade builder: it does not touch the application,
the default pipeline, or any existing dataset.
"""
from __future__ import annotations
import argparse, collections, hashlib, io, json, math, random, shutil
from pathlib import Path

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/watermark_independent_v1'
V1 = ROOT / 'data/watermark_training_v1'
REG = ROOT / 'data/watermark_regression_min_v1'
FONT = DEST / 'assets/fonts/NotoSansCJKsc-Regular.otf'
SEED = 20261102
LABEL = '独立核验'

# 12 scenarios; styles intentionally mirror the painful real styles (tiled text,
# dot grid, corner marks, cropped edges, heavy dark text) with v1-compatible names.
SCENARIOS = [
    ('local_text', 'C01', '局部文字'),
    ('timestamp', 'C01', '日期时间戳'),
    ('logo_transparent', 'C02', '透明几何 Logo'),
    ('center_text', 'C03', '居中大字'),
    ('center_white', 'C03', '白色半透明大字'),
    ('tiled_diagonal', 'C04', '斜向平铺文字'),
    ('tiled_irregular', 'C04', '非等距密集文字'),
    ('dot_grid', 'C05', '点阵'),
    ('line_net', 'C05', '细线网格'),
    ('tiny_corner', 'C06', '高清小角标'),
    ('cropped_edge', 'C06', '边缘裁切文字'),
    ('heavy_dark', 'C07', '深色大字'),
]
VARIANTS_PER_SCENARIO = 5
NEGATIVES = 6


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dhash(im: Image.Image) -> int:
    a = list(im.convert('L').resize((9, 8)).getdata())
    return sum((a[y * 9 + x] > a[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8))


def savejson(p: Path, x) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2))


def font(n) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT), max(8, int(n)))


def seed_for(s: str) -> int:
    return int(hashlib.sha256((str(SEED) + ':' + s).encode()).hexdigest()[:12], 16)


def filepart(path: Path) -> str:
    return str(path.relative_to(DEST))


# ----------------------------------------------------------------------------- bases
def clean_pdf(idx: int, path: Path) -> None:
    """New fictitious document bases; layout deliberately diverges from v1
    (different page size, margins, grid geometry, column structure) so the
    pages are not near-duplicate re-renders of v1 document templates."""
    rng = random.Random(SEED + 7000 + idx)
    doc = fitz.open()
    topics = ['库区环境监测记录', '样例账单与凭证', '会议决议与签到', '培训考核登记']
    for pno in range(2):
        p = doc.new_page(width=612, height=792)  # US Letter, not v1's 595x842
        p.draw_rect((0, 0, 612, 792), color=None, fill=(1, 1, 1))
        p.draw_rect((48, 48, 564, 58), fill=(.34, .22, .12), color=None)  # different band color/position
        p.insert_text((52, 92), topics[idx % 4] + f' · 独验 {idx + 1:03d}', fontname='china-s', fontsize=22, color=(.2, .12, .08))
        p.insert_text((52, 116), f'INDEPENDENT VALIDATION MATERIAL | SET {idx + 1:03d} | PAGE {pno + 1}', fontsize=8, color=(.45, .42, .38))
        p.insert_text((52, 140), '所有名称、编号与数值均为本项目构造的虚拟内容。', fontname='china-s', fontsize=11)
        if idx % 4 in [0, 1]:
            # two-column body, different from v1's single full-width lines
            for col in range(2):
                x0 = 52 + col * 258
                for j in range(26):
                    y = 168 + j * 19
                    if idx % 4 == 0:
                        line = f'{j + 1:02d} 监测点位{chr(65 + (j + col * 3) % 8)}-{col + 1}：读数{rng.randrange(10, 980)}，已复核。'
                        p.insert_text((x0, y), line, fontname='china-s', fontsize=9)
                    else:
                        line = f'{j + 1:02d} Entry {col + 1}-{rng.randrange(100, 9999)}: {rng.random() * 100:.3f} recorded.'
                        p.insert_text((x0, y), line, fontsize=9)
        elif idx % 4 == 2:
            for j in range(21):
                y = 164 + j * 24
                p.draw_line((52, y), (568, y), color=(.74, .7, .66), width=.5)
                if j < 20:
                    p.insert_text((60, y + 16), f'Row {j + 1:02d}', fontsize=9)
                    p.insert_text((180, y + 16), str(rng.randrange(1000, 9000)), fontsize=9)
                    p.insert_text((330, y + 16), f'{rng.random() * 100:.2f} %', fontsize=9)
                    p.insert_text((455, y + 16), 'Checked', fontsize=8)
            for x in [52, 170, 320, 445, 568]:
                p.draw_line((x, 164), (x, 644), color=(.74, .7, .66), width=.5)
            p.insert_text((52, 684), '数值仅用于构造表格，不对应任何真实个人或组织。', fontname='china-s', fontsize=10)
        else:
            for j in range(10):
                x = 60 + j * 50
                ht = rng.randint(50, 180)
                p.draw_rect((x, 430 - ht, x + 26, 430), fill=(.5, .28 + j * .03, .16), color=None)
                p.insert_text((x, 448), str(j + 1), fontsize=8)
            for j in range(12):
                p.insert_text((52, 480 + j * 19), f'条目{j + 1}：领用物料{rng.randrange(10, 90)}件；经办人签字留空。', fontname='china-s', fontsize=9)
        p.draw_line((52, 726), (568, 726), color=(.6, .55, .5))
        p.insert_text((52, 748), f'INDEPENDENT SAMPLE {idx + 1:03d} / {pno + 1:02d}', fontsize=8)
        p.insert_text((428, 748), '独验编号 / 非水印', fontname='china-s', fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.set_metadata({'title': f'Independent fictitious source {idx + 1:03d}', 'author': 'Project-created', 'subject': 'No real personal data'})
    doc.save(path, garbage=4, deflate=True)
    doc.close()


def make_chart(i: int) -> Image.Image:
    rng = np.random.default_rng(SEED + 8000 + i)
    w, h = 1152, 864
    im = Image.new('RGB', (w, h), (245, 248, 251))
    d = ImageDraw.Draw(im)
    d.text((65, 40), f'独立核验图表 / 构造底图 {i + 1:02d}', font=font(32), fill=(30, 50, 80))
    d.line((70, 700, 1090, 700), fill=(60, 80, 100), width=2)
    d.line((70, 150, 70, 700), fill=(60, 80, 100), width=2)
    for j in range(8):
        x = 110 + j * 115
        v = int(rng.integers(120, 500))
        d.rectangle((x, 700 - v, x + 65, 699), fill=(35 + j * 12, 90 + j * 6, 180 - j * 8))
        d.text((x, 715), f'M{j + 1}', font=font(20), fill=(40, 50, 65))
        d.text((x, 660 - v), str(v), font=font(18), fill=(45, 60, 80))
    pts = [(110 + j * 115, int(rng.integers(180, 450))) for j in range(8)]
    d.line(pts, fill=(90, 160, 75), width=4)
    d.text((75, 790), '构造图例及坐标轴属于正文内容。', font=font(22), fill=(75, 85, 100))
    return im


def make_texture(i: int) -> Image.Image:
    rng = np.random.default_rng(SEED + 8100 + i)
    w, h = 1152, 864
    yy, xx = np.mgrid[:h, :w]
    idx = i % 4
    if idx == 0:
        v = (np.sin(xx / (9 + i)) + np.cos(yy / (13 + i))) * 25 + 125
    elif idx == 1:
        v = (np.sin(xx * .025 + np.cos(yy * .02) * 2) + np.sin((xx + yy) * .045)) * 35 + 120
    elif idx == 2:
        v = ((xx // (22 + i) + yy // (22 + i)) % 2) * 100 + 60
    else:
        v = rng.normal(128, 35, (h, w))
    channels = [np.clip(v + rng.normal(0, 8, (h, w)) + k * 18, 0, 255) for k in range(3)]
    return Image.fromarray(np.stack(channels, -1).astype('uint8'))


def collect_bases() -> list[dict]:
    photos = json.loads((DEST / 'sources/commons_cc0_independent.json').read_text())
    if len(photos) != 12:
        raise ValueError(f'Expected exactly 12 independent CC0 photos, got {len(photos)}')
    bases = []
    for r in photos:
        creator = 'independent-set'
        bases.append(dict(id=r['id'], path=r['normalized_path'], source_group=r['source_group'],
                          split_group='independent:' + creator, split='independent',
                          background='photo_' + r['background'], provenance='downloaded_cc0_photo',
                          license='CC0-1.0', source_evidence=r['evidence_path']))
    for i in range(6):
        bid = f'xdoc_{i + 1:03d}'
        pdf = DEST / f'bases/documents/{bid}.pdf'
        clean_pdf(i, pdf)
        with fitz.open(pdf) as doc:
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
        path = pdf.with_suffix('.png')
        pix.save(path)
        bases.append(dict(id=bid, path=filepart(path), pdf=filepart(pdf), source_group='original:' + bid,
                          split_group='independent:' + bid, split='independent',
                          background='document_zh', provenance='project_original_document',
                          license='LicenseRef-Project-Original'))
    for i in range(4):
        bid = f'xchart_{i + 1:03d}'
        path = DEST / f'bases/charts/{bid}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        make_chart(i).save(path)
        bases.append(dict(id=bid, path=filepart(path), source_group='original:' + bid,
                          split_group='independent:' + bid, split='independent', background='chart',
                          provenance='project_original_procedural', license='LicenseRef-Project-Original'))
    for i in range(4):
        bid = f'xtex_{i + 1:03d}'
        path = DEST / f'bases/textures/{bid}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        make_texture(i).save(path)
        bases.append(dict(id=bid, path=filepart(path), source_group='original:' + bid,
                          split_group='independent:' + bid, split='independent', background='procedural_texture',
                          provenance='project_original_procedural', license='LicenseRef-Project-Original'))
    for b in bases:
        b['sha256'] = digest(DEST / b['path'])
        with Image.open(DEST / b['path']) as im:
            b['size'] = list(im.size)
    savejson(DEST / 'bases/manifest.json', bases)
    return bases


# ------------------------------------------------------------------------ watermarks
def watermark_layer(size, style: str, key: str):
    """Self-contained copy of the v1 generator with a new seed chain and tokens."""
    w, h = size
    s = min(w, h)
    rng = random.Random(seed_for(key))
    layer = Image.new('RGBA', size)
    color = (rng.randrange(20, 160), rng.randrange(20, 160), rng.randrange(20, 160))
    opacity = rng.choice([.15, .3, .5, .7])
    token = f'{LABEL}样本 {seed_for(key) % 10000:04d}'
    params = dict(style=style, seed=seed_for(key), opacity=opacity, color=list(color), text=token,
                  asset_partition='independent', asset_id='independent:' + key,
                  watermark_origin='project_synthetic')

    def text_tile(text, fs, fill, angle=0, stroke=0):
        f = font(fs)
        box = f.getbbox(text, stroke_width=stroke)
        pad = max(8, stroke + 4)
        t = Image.new('RGBA', (max(1, box[2] - box[0] + pad * 2), max(1, box[3] - box[1] + pad * 2)))
        ImageDraw.Draw(t).text((pad - box[0], pad - box[1]), text, font=f, fill=fill,
                               stroke_width=stroke, stroke_fill=fill)
        return t.rotate(angle, Image.Resampling.BICUBIC, expand=True) if angle else t

    def paste(t, x, y):
        layer.alpha_composite(t, (int(x), int(y)))

    def centered(text=token, fs=None, fill=None, angle=0):
        t = text_tile(text, fs or s * .09, fill or (*color, round(opacity * 255)), angle)
        paste(t, (w - t.width) / 2, (h - t.height) / 2)

    def logo(x, y, side, alpha):
        d = ImageDraw.Draw(layer)
        cx, cy = x + side / 2, y + side / 2
        n = rng.randrange(5, 10)
        points = []
        for j in range(n * 2):
            a = j * math.pi / n
            r = side * (.48 if j % 2 == 0 else rng.uniform(.18, .30))
            points.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
        d.polygon(points, fill=(*color, alpha))
        d.ellipse((cx - side * .12, cy - side * .12, cx + side * .12, cy + side * .12), fill=(240, 245, 250, alpha))
        d.text((x + side * .05, y + side * .74), f'X{seed_for(key) % 1000:03}', font=font(side * .14), fill=(255, 255, 255, alpha))

    a = round(opacity * 255)
    if style in ['local_text', 'timestamp', 'website']:
        text = token if style == 'local_text' else f'2026-11-{1 + seed_for(key) % 28:02d} {rng.randrange(24):02d}:{rng.randrange(60):02d}' if style == 'timestamp' else f'https://ind-{seed_for(key) % 1000}.example.test'
        t = text_tile(text, s * .035, (*color, a))
        placement = rng.choice(['bottom_right', 'top_left', 'body_overlap'])
        x, y = (w - t.width - s * .03, h - t.height - s * .03) if placement == 'bottom_right' else (s * .03, s * .03) if placement == 'top_left' else ((w - t.width) / 2, h * .45)
        paste(t, x, y)
        params.update(text=text, placement=placement)
    elif style == 'logo_transparent':
        logo(w - s * .25, h - s * .25, s * .21, a)
    elif style in ['center_text', 'center_white', 'heavy_dark']:
        if style == 'center_white':
            color = (255, 255, 255)
        if style == 'heavy_dark':
            color = (10, 12, 18)
            opacity = .85
        params.update(color=list(color), opacity=opacity)
        centered(fs=s * .11, fill=(*color, round(opacity * 255)), angle=25)
    elif style in ['tiled_diagonal', 'tiled_irregular']:
        irregular = style == 'tiled_irregular'
        fs = s * (.045 if irregular else .055)
        t = text_tile(token, fs, (*color, a), 32)
        step_x = max(30, int(t.width * (.78 if irregular else 1.25)))
        step_y = max(30, int(t.height * (.7 if irregular else 1.5)))
        for y in range(-t.height, h + t.height, step_y):
            for x in range(-t.width, w + t.width, step_x):
                paste(t, x + (rng.randrange(-12, 13) if irregular else 0), y + (rng.randrange(-10, 11) if irregular else 0))
        params['rotation_degrees'] = 32
        params['irregular'] = irregular
    elif style == 'dot_grid':
        d = ImageDraw.Draw(layer)
        step = max(18, int(s * .05))
        rad = max(1, int(s * .0025))
        family = rng.choice(['square', 'rectangular', 'staggered'])
        step_y = int(step * 1.4) if family == 'rectangular' else step
        for j, y in enumerate(range(step_y // 2, h, step_y)):
            shift = step // 2 if family == 'staggered' and j % 2 else 0
            for x in range(step // 2 + shift, w, step):
                d.ellipse((x - rad, y - rad, x + rad, y + rad), fill=(*color, a))
        params.update(spacing_px=[step, step_y], radius_px=rad, grid_family=family)
    elif style == 'line_net':
        d = ImageDraw.Draw(layer)
        step = max(24, int(s * .08))
        lw = max(1, int(s * .0012))
        for x in range(-h, w + h, step):
            d.line((x, 0, x + h, h), fill=(*color, a), width=lw)
            d.line((x, 0, x - h, h), fill=(*color, a), width=lw)
        params.update(spacing_px=step, line_width_px=lw)
    elif style == 'tiny_corner':
        fs = max(9, int(s * .012))
        t = text_tile(f'IND-{seed_for(key) % 10000:04}', fs, (*color, 170))
        paste(t, w - t.width - 4, h - t.height - 4)
        params.update(font_size_px=fs, opacity=170 / 255)
    elif style == 'cropped_edge':
        t = text_tile(token, s * .09, (*color, a), 12)
        paste(t, w - t.width * .55, h - t.height * .6)
        params['edge_crop'] = True
    else:
        raise ValueError(style)
    alpha = np.asarray(layer.getchannel('A'))
    if not np.any(alpha):
        raise ValueError('Empty watermark: ' + style)
    params.update(alpha_min_nonzero=int(alpha[alpha > 0].min()), alpha_max=int(alpha.max()),
                   support_fraction=float(np.count_nonzero(alpha) / alpha.size),
                   alpha_definition='effective_8bit_alpha_of_composited_synthetic_layer')
    return layer, params


# ----------------------------------------------------------------------------- images
def build_images(bases: list[dict]) -> list[dict]:
    rows = []
    pool_photos = [b for b in bases if b['background'].startswith('photo_')]
    pool_docs = [b for b in bases if b['provenance'] == 'project_original_document']
    pool_charts = [b for b in bases if b['background'] == 'chart']
    pool_tex = [b for b in bases if b['background'] == 'procedural_texture']
    for si, (style, scenario, label) in enumerate(SCENARIOS):
        chosen = [pool_photos[(si * 2) % len(pool_photos)], pool_photos[(si * 2 + 5) % len(pool_photos)],
                  pool_docs[si % len(pool_docs)], pool_charts[si % len(pool_charts)], pool_tex[si % len(pool_tex)]]
        for vi, b in enumerate(chosen):
            sid = f'ind_{si + 1:02d}_{style}__{b["id"]}'
            folder = DEST / f'images/independent/{sid}'
            folder.mkdir(parents=True, exist_ok=True)
            clean = Image.open(DEST / b['path']).convert('RGB')
            overlay, params = watermark_layer(clean.size, style, sid)
            watermarked = Image.alpha_composite(clean.convert('RGBA'), overlay).convert('RGB')
            paths = dict(clean=b['path'])
            for role, im in [('watermarked', watermarked), ('overlay_rgba', overlay),
                             ('alpha', overlay.getchannel('A')),
                             ('mask', overlay.getchannel('A').point(lambda v: 255 if v else 0))]:
                p = folder / (role + '.png')
                im.save(p, compress_level=3)
                paths[role] = filepart(p)
            rows.append(dict(id=sid, kind='image', scenario=scenario, subtype=style, label=label,
                             split='independent', source_group=b['source_group'], split_group=b['split_group'],
                             base_id=b['id'], background=b['background'],
                             provenance='project_synthetic_watermark_on_' + b['provenance'], license=b['license'],
                             size=list(clean.size), paths=paths,
                             sha256={k: digest(DEST / v) for k, v in paths.items()},
                             parameters=params, watermark_origin='project_synthetic', is_negative=False,
                             mask_definition='support_of_known_composited_layer_alpha_gt_zero',
                             partition='independent',
                             lock='one_look_baseline_then_frozen'))
    neg_bases = [pool_photos[1], pool_photos[8], pool_docs[1], pool_docs[4], pool_charts[0], pool_tex[0]]
    for b in neg_bases:
        sid = b['id'] + '__negative'
        folder = DEST / f'images/independent/{sid}'
        folder.mkdir(parents=True, exist_ok=True)
        clean = Image.open(DEST / b['path']).convert('RGB')
        paths = dict(clean=b['path'], watermarked=b['path'])
        for role in ['mask', 'alpha']:
            p = folder / (role + '.png')
            Image.new('L', clean.size).save(p)
            paths[role] = filepart(p)
        rows.append(dict(id=sid, kind='image', scenario='N00', subtype='no_watermark', label='无水印负例',
                         split='independent', source_group=b['source_group'], split_group=b['split_group'],
                         base_id=b['id'], background=b['background'], provenance=b['provenance'], license=b['license'],
                         size=list(clean.size), paths=paths,
                         sha256={k: digest(DEST / v) for k, v in paths.items()},
                         parameters={}, watermark_origin='none', is_negative=True,
                         mask_definition='all_zero', partition='independent',
                         lock='one_look_baseline_then_frozen'))
    return rows


# ---------------------------------------------------------------------------------- qa
def cross_set_checks(rows: list[dict]) -> dict:
    v1_bases = json.loads((V1 / 'bases/manifest.json').read_text())
    v1_sha = {b['sha256'] for b in v1_bases}
    my_sha = {r['paths']['clean'] and digest(DEST / r['paths']['clean']) for r in rows}
    my_sha |= {b['sha256'] for b in json.loads((DEST / 'bases/manifest.json').read_text())}
    sha_overlap = sorted(v1_sha & my_sha)

    def load_gray(paths):
        out = []
        for p in paths:
            with Image.open(p) as im:
                out.append(np.asarray(im.convert('L').resize((256, 256), Image.Resampling.BICUBIC), np.float32))
        return out

    # Near-white pages collide on 8x8 gradient hash; use 64x64 gray statistics.
    v1_photo_gray = load_gray(sorted((V1 / 'bases/photos').glob('*.png')))
    v1_other_gray = load_gray(sorted((V1 / 'bases/charts').glob('*.png')) +
                                sorted((V1 / 'bases/textures').glob('*.png')) +
                                sorted((V1 / 'bases/documents').glob('*.png')))
    reg_manifest = [json.loads(x) for x in (REG / 'manifest.jsonl').read_text().splitlines() if x.strip()]
    dev_gray = []
    for case in reg_manifest:
        if case.get('suite') != 'real':
            continue
        src = REG / case['paths']['input']
        try:
            dev_gray.append(np.asarray(Image.open(src).convert('L').resize((256, 256), Image.Resampling.BICUBIC), np.float32))
        except Exception:
            try:
                with fitz.open(src) as fdev:
                    pix = fdev[0].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                dev_gray.append(np.asarray(Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
                                           .convert('L').resize((256, 256), Image.Resampling.BICUBIC), np.float32))
            except Exception as exc:
                print('WARN dev hash skipped', src.name, exc)

    def nearest(gray, refs):
        best = None
        for ref in refs:
            d = float(np.abs(gray - ref).mean())
            if best is None or d < best:
                best = d
        return best

    base_files = sorted((DEST / 'bases').rglob('*.png'))
    min_photo, min_other, min_dev = 1e9, 1e9, 1e9
    for p in base_files:
        with Image.open(p) as im:
            gray = np.asarray(im.convert('L').resize((256, 256), Image.Resampling.BICUBIC), np.float32)
        min_photo = min(min_photo, nearest(gray, v1_photo_gray))
        min_other = min(min_other, nearest(gray, v1_other_gray))
        min_dev = min(min_dev, nearest(gray, dev_gray))
    return dict(sha256_overlap_with_v1=sha_overlap,
                min_gray_mad_vs_v1_photos=round(min_photo, 2),
                min_gray_mad_vs_v1_other_bases=round(min_other, 2),
                min_gray_mad_vs_dev14=round(min_dev, 2),
                pass_sha=not sha_overlap, pass_photo=min_photo > 8.0, pass_other=min_other > 4.0,
                pass_dev=min_dev > 4.0)


def integrity_checks(rows: list[dict]) -> dict:
    problems = []
    for r in rows:
        clean = Image.open(DEST / r['paths']['clean']).convert('RGB')
        mask = np.asarray(Image.open(DEST / r['paths']['mask']).convert('L'))
        if r['is_negative']:
            if mask.max() != 0:
                problems.append(r['id'] + ': negative mask not zero')
            continue
        overlay = Image.open(DEST / r['paths']['overlay_rgba'])
        recomposed = Image.alpha_composite(clean.convert('RGBA'), overlay).convert('RGB')
        watermarked = Image.open(DEST / r['paths']['watermarked']).convert('RGB')
        if not np.array_equal(np.asarray(recomposed), np.asarray(watermarked)):
            problems.append(r['id'] + ': watermarked != composite(clean, overlay)')
        alpha = np.asarray(overlay.getchannel('A'))
        if not np.array_equal(mask, (alpha > 0).astype('uint8') * 255):
            problems.append(r['id'] + ': mask != alpha>0')
        diff = np.abs(np.asarray(watermarked).astype(int) - np.asarray(clean).astype(int)).max(2)
        if diff[alpha > 0].mean() < 0.5:
            problems.append(r['id'] + ': watermark pixels nearly identical to clean')
    return dict(problems=problems, pass_integrity=not problems)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-fetch-check', action='store_true')
    args = ap.parse_args()
    FONT.parent.mkdir(parents=True, exist_ok=True)
    if not FONT.exists():
        src = V1 / 'assets/fonts/NotoSansCJKsc-Regular.otf'
        shutil.copy2(src, FONT)
    lic_src = V1 / 'licenses/NotoSansCJK-OFL.txt'
    lic_dst = DEST / 'licenses/NotoSansCJK-OFL.txt'
    lic_dst.parent.mkdir(parents=True, exist_ok=True)
    if not lic_dst.exists():
        shutil.copy2(lic_src, lic_dst)
    if not (DEST / 'sources/commons_cc0_independent.json').exists():
        raise SystemExit('Run tools/fetch_independent_bases.py first')
    bases = collect_bases()
    rows = build_images(bases)
    (DEST / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (DEST / 'splits').mkdir(exist_ok=True)
    (DEST / 'splits/independent.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    integrity = integrity_checks(rows)
    cross = cross_set_checks(rows)
    qa = dict(created_on='2026-11-02', seed=SEED, checks=dict(integrity=integrity, cross_set=cross),
              pass_all=integrity['pass_integrity'] and all([cross['pass_sha'], cross['pass_photo'],
                                                            cross['pass_other'], cross['pass_dev']]))
    savejson(DEST / 'qa/validation.json', qa)
    savejson(DEST / 'build_summary.json', dict(
        seed=SEED, created_on='2026-11-02', total_records=len(rows),
        positives=sum(not r['is_negative'] for r in rows), negatives=sum(r['is_negative'] for r in rows),
        scenarios=dict(collections.Counter(r['scenario'] for r in rows)),
        scenario_styles=[s[0] for s in SCENARIOS], variants_per_scenario=VARIANTS_PER_SCENARIO,
        bases=len(bases), partition_rule='entire set locked; one-look baseline then frozen',
        independence='new bases / new seed chain / new tokens; cross-set overlap asserted in qa/validation.json',
        note='project-synthetic only; not derived from v1 or dev-14 assets; never merge into training splits'))
    savejson(DEST / 'dataset_card.json', dict(
        name='独立验证集 v1（一次性查看后锁柜）', version='1.0.0', created_on='2026-11-02',
        intended_use='独立泛化评估：基线一次性查看，此后任何据其调参的行为使其降级为开发/回归用途',
        counts=dict(total=len(rows), positive=sum(not r['is_negative'] for r in rows),
                    negative=sum(r['is_negative'] for r in rows), bases=len(bases), scenarios=len(SCENARIOS)),
        splits=dict(independent=len(rows)), partition='independent', lock='one_look_baseline_then_frozen',
        watermark_origin='100% project-synthetic', alpha='exact 8-bit effective alpha',
        overlap_policy='excludes v1 bases and 14 dev images by sha256 and perceptual hash (qa/validation.json)',
        rights_basis='CC0-1.0 photos with evidence; project-original documents/charts/textures'))
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    if not qa['pass_all']:
        raise SystemExit('QA FAILED — see qa/validation.json')
    print('COMPLETE', len(rows), 'records ->', DEST)


if __name__ == '__main__':
    main()
