"""日期时间戳 OCR 增强二次识别 · 只读诊断探针（2026-11-05）。

背景：14 张日期时间戳样本中 11 张 0% 召回，归因集中在第一环——OCR 没读出
那行小字半透明时间戳，后续通道连搜索范围都没有。本探针验证：对比度增强
+ 上采样的二次 OCR 能否在这些页面上把入口补出来。

只做诊断，不改 wm 任何代码：
- 第一轮：现成 wm.ocr.read_lines（原图，桌面默认同款）；
- 第二轮变体 A：全图 灰度→CLAHE→2x 上采样后 read_lines（坐标映射回原图）；
- 第二轮变体 B：只增强第一轮未覆盖区域的外接带（时间戳高频位角落 +
  未覆盖区域整体），避免全图误读扩散；
- 命中定义：读出文本匹配时间戳正则（日期型或时间型），与样本真值文案
  无关；推理完成后才用参考核对命中行是否落在真实水印区（评分分离）。

输出：探针结果 JSON + 每样本读出明细，供是否进入实现轮的决策。
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

OUT = ROOT / 'out/日期时间戳OCR增强探针_20261105'
FREEZE = ROOT / 'out/项目防退化控制/20261104_002213_219680_freeze'

DATE_RE = re.compile(r'(19|20)\d{2}\s*[-/年.]\s*\d{1,2}\s*[-/月.]\s*\d{1,2}')
TIME_RE = re.compile(r'\d{1,2}\s*[:：]\s*\d{2}')


def is_timestamp_like(text: str) -> bool:
    t = re.sub(r'\s+', ' ', text)
    return bool(DATE_RE.search(t) or TIME_RE.search(t))


def enhance(gray: np.ndarray) -> np.ndarray:
    """CLAHE + 2x 上采样（检测端小字的瓶颈是像素数，不是识别模型）。"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    return cv2.resize(eq, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def read_on(work_rgb: np.ndarray):
    """在（可能被增强/缩放过的）图上跑 read_lines，盒子映射回原尺寸。"""
    from wm import ocr
    h, w = work_rgb.shape[:2]
    lines = ocr.read_lines(work_rgb)
    return lines or []


def coverage_mask(shape, lines, dilate=9):
    mask = np.zeros(shape, np.uint8)
    for box, _t, _c in lines:
        cv2.fillPoly(mask, [np.asarray(box, np.int32)], 1)
    if lines:
        k = 2 * dilate + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask > 0


def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((FREEZE / '样本清单.json').read_text())
    ts_cases = [c for c in manifest['cases'] if c['style'] == 'timestamp']
    negs = [c for c in manifest['cases'] if c.get('negative')]
    # 负例取不同底质的 4 张
    picked_neg, seen = [], set()
    for c in negs:
        sg = c.get('source_group') or c['id']
        if sg not in seen:
            picked_neg.append(c)
            seen.add(sg)
        if len(picked_neg) >= 4:
            break

    results = {}
    for case in ts_cases + picked_neg:
        cid = case['id']
        img = cv2.imread(case['input'])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row = {'negative': bool(case.get('negative'))}

        t0 = time.time()
        first = read_on(rgb)
        row['first_pass'] = {
            'lines': len(first),
            'timestamp_hits': [t for _b, t, _c in first if is_timestamp_like(t)],
            'texts': [t for _b, t, _c in first],
            'elapsed_s': round(time.time() - t0, 2),
        }

        covered = coverage_mask(gray.shape, first)

        # 变体 A：全图增强 2x
        t0 = time.time()
        enh = enhance(gray)
        lines_a = read_on(cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB))
        scale = 2.0
        new_a = []
        for box, text, conf in lines_a:
            box_orig = np.asarray(box, np.float32) / scale
            box_orig[:, 0] = np.clip(box_orig[:, 0], 0, gray.shape[1] - 1)
            box_orig[:, 1] = np.clip(box_orig[:, 1], 0, gray.shape[0] - 1)
            inside = covered[box_orig[:, 1].astype(int) % gray.shape[0],
                             box_orig[:, 0].astype(int) % gray.shape[1]].mean()
            if inside < 0.5 and is_timestamp_like(text):
                new_a.append({'text': text, 'conf': round(float(conf), 3),
                              'box': np.round(box_orig, 1).tolist()})
        row['variant_A_full_enhance'] = {
            'new_timestamp_hits': new_a, 'lines': len(lines_a),
            'elapsed_s': round(time.time() - t0, 2),
        }

        # 变体 B：未覆盖区域贴回原尺寸增强版（覆盖区域保持原图）
        t0 = time.time()
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enh_1x = clahe.apply(gray)
        uncov = ~covered
        comp_img = rgb.copy().astype(np.float32)
        comp_img[uncov] = cv2.cvtColor(enh_1x, cv2.COLOR_GRAY2RGB).astype(np.float32)[uncov]
        comp_img = comp_img.astype(np.uint8)
        lines_b = read_on(comp_img)
        new_b = []
        for box, text, conf in lines_b:
            box_orig = np.asarray(box, np.float32)
            inside = covered[box_orig[:, 1].astype(int) % gray.shape[0],
                             box_orig[:, 0].astype(int) % gray.shape[1]].mean()
            if inside < 0.5 and is_timestamp_like(text):
                new_b.append({'text': text, 'conf': round(float(conf), 3),
                              'box': np.round(box_orig, 1).tolist()})
        row['variant_B_uncovered_enhance'] = {
            'new_timestamp_hits': new_b, 'lines': len(lines_b),
            'elapsed_s': round(time.time() - t0, 2),
        }

        # 评分分离：推理完成后才用真值核对命中行是否落在真实水印区
        if case.get('references') and not row['negative']:
            alpha = cv2.imread(case['references'][0]['alpha'], 0)
            gt = alpha > 0
            for var in ('variant_A_full_enhance', 'variant_B_uncovered_enhance'):
                for hit in row[var]['new_timestamp_hits']:
                    bx = np.asarray(hit['box'], np.int32)
                    xs, ys = bx[:, 0], bx[:, 1]
                    x0, x1 = max(0, xs.min()), min(gt.shape[1], xs.max() + 1)
                    y0, y1 = max(0, ys.min()), min(gt.shape[0], ys.max() + 1)
                    region = np.zeros_like(gt)
                    region[y0:y1, x0:x1] = True
                    hit['in_true_watermark'] = bool((region & gt).any())
                    ys_g, xs_g = np.nonzero(gt)
                    hit['gt_center_dist_px'] = int(np.hypot(
                        (x0 + x1) / 2 - xs_g.mean(), (y0 + y1) / 2 - ys_g.mean())) if len(xs_g) else None
        results[cid] = row
        print(f"{cid}: 一轮{row['first_pass']['lines']}行 "
              f"命中{len(row['first_pass']['timestamp_hits'])} | "
              f"A新增{len(row['variant_A_full_enhance']['new_timestamp_hits'])} | "
              f"B新增{len(row['variant_B_uncovered_enhance']['new_timestamp_hits'])}", flush=True)

    (OUT / '探针结果_20261105.json').write_text(
        json.dumps({'schema_version': 1, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    'note': '只读诊断探针；时间戳命中由正则定义，与样本真值文案无关；'
                            'in_true_watermark 为推理后核对。',
                    'date_re': DATE_RE.pattern, 'time_re': TIME_RE.pattern,
                    'results': results}, ensure_ascii=False, indent=2))
    print('written:', OUT / '探针结果_20261105.json')


if __name__ == '__main__':
    main()
