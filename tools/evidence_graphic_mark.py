#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图形块候选通道证据脚本（不动应用代码）。

提案规则：5x5 局部方差<120（平色）且与 25px 环均值差>18 的种子像素，
按连通域逐个过门槛：面积 800–150000、宽高比 0.35–2.8、内部σ≤15、
颜色距离≥45、凸度 0.40–0.95、OCR 文字区排除。
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wm import ocr  # noqa: E402


def components(img, keepout=None):
    f = img.astype(np.float32)
    mean = ndimage.uniform_filter(f, size=(5, 5, 1))
    var = ndimage.uniform_filter(f * f, size=(5, 5, 1)) - mean * mean
    flat = var.max(2) < 120
    diff = np.abs(f.mean(2) - ndimage.uniform_filter(f.mean(2), size=(25, 25)))
    seed = flat & (diff > 18)
    if keepout is not None:
        seed &= ~keepout
    lab, n = ndimage.label(seed)
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        a = int(comp.sum())
        if not 800 <= a <= 150000:
            continue
        ys, xs = np.nonzero(comp)
        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        if not (0.35 <= w / h <= 2.8):
            continue
        inner = f[comp]
        if inner.std(0).mean() > 15:
            continue
        dm = ndimage.binary_dilation(comp, iterations=13) & ~comp
        if not dm.any():
            continue
        cdist = float(np.linalg.norm(inner.mean(0) - f[dm].mean(0)))
        if cdist < 45:
            continue
        pts = np.column_stack([xs[::max(1, len(xs) // 1200)],
                               ys[::max(1, len(ys) // 1200)]])
        try:
            sol = a / ConvexHull(pts).volume
        except Exception:
            continue
        if not (0.40 <= sol <= 0.95):
            continue
        out.append(dict(mask=comp, area=a, wh=(int(w), int(h)),
                        cdist=round(cdist, 1), sig=round(float(inner.std(0).mean()), 1),
                        sol=round(float(sol), 2)))
    return out


def main():
    app = ROOT / 'data' / '应用测试输入'
    truth = {}
    for ds in ('watermark_training_v1', 'watermark_training_increment_v1_1'):
        for line in (ROOT / 'data' / ds / 'manifest.jsonl').read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get('kind') == 'image' and not r.get('is_negative'):
                    truth[r['id']] = (ds, r)

    print('== logo/签名 正例 ==')
    for style in ('logo_opaque', 'logo_transparent', 'signature'):
        rows = []
        for cid, (ds, rec) in sorted(truth.items()):
            if not cid.endswith('__' + style):
                continue
            hits = list(app.rglob(cid + '.png'))
            if not hits:
                continue
            img = np.asarray(Image.open(hits[0]).convert('RGB'))
            alpha = np.asarray(Image.open(str(ROOT / 'data' / ds / rec['paths']['alpha']))
                               .convert('L').resize(img.shape[1::-1], Image.Resampling.LANCZOS))
            t = alpha > 20
            keep = ocr.text_zones(img)
            comps = components(img, keepout=keep)
            sel = np.zeros(t.shape, bool)
            for c in comps:
                sel |= c['mask']
            rec_ = float((t & sel).sum() / max(t.sum(), 1))
            fp = int((sel & ~t).sum())
            rows.append((cid, rec_, fp, len(comps)))
        good = sum(1 for _, r_, _, _ in rows if r_ > 0.5)
        print(f'{style}: 提案召回>50% 的页 {good}/{len(rows)}')
        for cid, r_, fp, n in rows:
            flag = 'OK ' if r_ > 0.5 else '低 '
            print(f'  {flag}{cid:<44} 召回{r_*100:5.1f}% 多选{fp:6d}px 候选{n}')

    print('== 负例 ==')
    nfire = 0
    for p in sorted(app.rglob('*__negative*.png')) + sorted(app.rglob('*normal_logo_signature_negative*')):
        img = np.asarray(Image.open(p).convert('RGB'))
        comps = components(img, keepout=ocr.text_zones(img))
        if comps:
            nfire += 1
            print(' ', p.relative_to(app), [(c['area'], c['wh'], c['cdist'], c['sol']) for c in comps[:3]])
    print(f'负例触发候选: {nfire} 张')

    print('== 其他常见样式抽查（应全部无候选）==')
    for style in ('center_text', 'center_white', 'center_gray', 'timestamp', 'website',
                  'local_text', 'tiny_corner', 'tiled_diagonal'):
        n = 0
        for p in sorted(app.rglob(f'*__{style}.png'))[:6]:
            img = np.asarray(Image.open(p).convert('RGB'))
            if components(img, keepout=ocr.text_zones(img)):
                n += 1
                print('  触发!', p.relative_to(app))
        print(f'{style}: 抽查触发 {n} 张')


if __name__ == '__main__':
    main()
