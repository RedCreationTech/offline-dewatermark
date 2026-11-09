"""浅灰底/渐变底新开发样本生成与白纸纯度门槛边界标定（2026-11-05）。

目的（预登记门槛 promotion_requires）：标定 faint_region 白纸纯度门槛
（纸面极点亮度 P≥250、干净纸亮度 std≤3）在浅灰底/渐变底上的实际边界，
产出后续转正可用的新开发样本。

方法：合成"文档页"（底 + 正文行 + 一块极淡一致染色区，模拟时间戳染色），
底部分两系列：
- 浅灰底：平坦底色 L ∈ {254,252,250,248,245,240,235,230,220}；
- 渐变底：白→L 垂直渐变，L ∈ {252,248,242,235,228,220}。
每页跑 faint_region.estimate，记录触发/拒绝与第一道拒绝的门槛，
找出"接受→拒绝"的边界所在区间。

合成参数（α、W、染色区位置）全系列固定，只动底色——边界差异唯一来自
参照面纯度。这些是开发/标定样本，不是独立验证样本。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

OUT = ROOT / 'out/极淡区域反染色验收_20261105/新开发样本_灰底标定'
ALPHA = 0.015          # 与 document_029/032 实测同量级
W_COLOR = 80           # 水印颜色（深灰）
REGION_FRAC = 0.10     # 染色区占比（与合成测试同一量级）


def make_page(base_kind: str, level: int, size=(600, 800)):
    h, w = size
    if base_kind == 'flat':
        img = np.full((h, w, 3), level, np.uint8)
    else:  # gradient: 顶部白 → 底部 level
        col = np.linspace(255, level, h, dtype=np.float32)[:, None, None]
        img = np.repeat(np.repeat(col, w, axis=1), 3, axis=2).astype(np.uint8)
    for y in (60, 120, 540):
        darker = tuple(max(0, c - 180) for c in (level,) * 3) if level > 128 else (60, 60, 60)
        cv2.putText(img, 'RECORD 086 CONFIRMED', (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, darker, 2, cv2.LINE_AA)
    rw, rh = int(w * 0.45), int(h * 0.22)
    x0, y0 = (w - rw) // 2, (h - rh) // 2
    tint = (1 - ALPHA) * img[y0:y0 + rh, x0:x0 + rw] + ALPHA * W_COLOR
    img[y0:y0 + rh, x0:x0 + rw] = np.round(tint).astype(np.uint8)
    gt = np.zeros((h, w), bool)
    gt[y0:y0 + rh, x0:x0 + rw] = True
    return img, gt


def probe(img):
    """返回 (触发?, 详情)。细节含第一道门槛，便于边界归因。"""
    from wm import faint_region as fr
    gray = img.astype(np.float32).mean(2)
    ref = fr._paper_reference(img.astype(np.float32), gray)
    gates = {}
    if ref is None:
        return False, {'gate': '纸面参照不足（纸占比<40%）'}
    P, _paper = ref
    gates['P'] = round(P, 2)
    paperish = gray > 0.93 * P
    dev = P - gray
    evidence = fr._evidence_mask(dev, paperish)
    if evidence is None:
        return False, {'gate': '多数决染色证据不足', **gates}
    core = fr._dilate(evidence, fr._CORE_DILATE)
    if not (fr._REGION_MIN_PX <= core.sum() <= fr._REGION_MAX_FRAC * core.size):
        return False, {'gate': '区域尺寸门（核心区超界）', **gates,
                       'core_px': int(core.sum())}
    fit = fr._fit_uniform_tint(img.astype(np.float32), gray, core, paperish, P)
    if fit is None:
        # 细分是白纸纯度还是拟合优度
        in_region = core & paperish
        dvec = (img.astype(np.float32) - P)[in_region].mean(1)
        stained = in_region.copy()
        stained[in_region] = dvec < -1.0
        cleanp = in_region & ~stained
        gate = '拟合优度/颜色合理区间'
        if P < fr._PAPER_LEVEL_MIN:
            gate = f'白纸纯度：P={P:.1f} < {fr._PAPER_LEVEL_MIN}'
        elif cleanp.sum() >= 200 and float(gray[cleanp].std()) > fr._PAPER_CLEAN_STD_MAX:
            gate = f'干净纸离散度：std={float(gray[cleanp].std()):.2f} > {fr._PAPER_CLEAN_STD_MAX}'
        return False, {'gate': gate, **gates}
    a_hat, _w, stats = fit
    return True, {'gate': '通过（触发）', 'alpha_hat': round(a_hat, 4), **gates,
                  **{k: round(float(v), 3) for k, v in stats.items()}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, levels in (('flat', [254, 252, 250, 248, 245, 240, 235, 230, 220]),
                         ('gradient', [252, 248, 242, 235, 228, 220])):
        for level in levels:
            img, gt = make_page(kind, level)
            name = f'{kind}_L{level}'
            cv2.imwrite(str(OUT / f'{name}.png'), img)
            triggered, detail = probe(img)
            rows.append({'id': name, 'base_kind': kind, 'base_level': level,
                         'alpha': ALPHA, 'w_color': W_COLOR,
                         'triggered': triggered, **detail})
            print(name, '->', '触发' if triggered else '拒绝', detail, flush=True)
    (OUT / '标定结果.json').write_text(
        json.dumps({'schema_version': 1,
                    'purpose': '白纸纯度门槛（P≥250、干净纸std≤3）边界标定；开发样本，非独立验证',
                    'fixed_synthesis': {'alpha': ALPHA, 'w_color': W_COLOR,
                                        'region_frac': REGION_FRAC, 'size': [600, 800]},
                    'rows': rows}, ensure_ascii=False, indent=2))
    print(f'标定结果：{OUT / "标定结果.json"}')


if __name__ == '__main__':
    main()
