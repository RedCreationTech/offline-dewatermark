#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""常见水印基线报告（模版版）适配器。

把 out/常见水印基线_20261104/（run_common_baseline.py 的产出）装配成
防退化控制轮目录的格式（运行记录.json / 原始清单.json / 参考_*/逐图效果初筛.json），
然后直接调用 tools/build_control_reports.py 的 build_both() ——
摘要/明细报告即为 11-04 定稿模版本体（结论置顶、三档 tab、筛选、
较上次退化/优化对比、蓝漏选/橙多选核对图、放大查看）。

对比参考取 20261104_002213 全量封存基线的同页子集（desktop 模式）。
生成后把文件名与标题从“项目防退化控制”改为“常见水印基线”，内容其余部分不动。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'out' / (sys.argv[1] if len(sys.argv) > 1 else '常见水印基线_20261104')
APP = ROOT / 'data' / '应用测试输入'
FREEZE = ROOT / 'out' / '项目防退化控制' / '20261104_002213_219680_freeze'

sys.path.insert(0, str(ROOT))
from tools.build_control_reports import build_both  # noqa: E402

STYLE_LABEL = {
    'timestamp': '日期时间戳', 'website': '网址文字', 'logo_transparent': '透明几何 Logo',
    'logo_opaque': '不透明几何 Logo', 'local_text': '局部文字', 'tiny_corner': '高清小角标',
    'signature': '模拟手写签名', 'center_text': '居中大字', 'center_white': '白色半透明大字',
    'center_gray': '灰黑半透明大字', 'tiled_diagonal': '斜向平铺文字',
}


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def truth_index():
    idx = {}
    for name in ('watermark_training_v1', 'watermark_training_increment_v1_1'):
        ds = ROOT / 'data' / name
        for line in (ds / 'manifest.jsonl').read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('kind') == 'image' and not r.get('is_negative'):
                idx[r['id']] = (ds, r)
    return idx


def input_map():
    """样本 id -> 应用测试输入相对路径"""
    m = {}
    for p in sorted(APP.rglob('*.png')):
        m[p.stem] = p.relative_to(APP).as_posix()
    return m


def overlay_mask(img: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[pred] = (0.5 * out[pred] + 0.5 * np.array([255, 64, 64])).astype(np.uint8)
    return out


def check_image(img: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """蓝=漏选（真水印没被选），橙=多选（选了背景）"""
    out = img.copy()
    miss = truth & ~pred
    fp = pred & ~truth
    out[miss] = (0.45 * out[miss] + 0.55 * np.array([64, 120, 255])).astype(np.uint8)
    out[fp] = (0.45 * out[fp] + 0.55 * np.array([255, 140, 40])).astype(np.uint8)
    return out


def main():
    s = json.loads((RUN / 'summary.json').read_text())
    idx = truth_index()
    imap = input_map()

    # 逐页派生指标与模版需要的图像文件
    records = []
    cases = []
    for r in s['results']:
        cid = r['case_id']
        rel = imap.get(cid)
        if rel is None:
            print('跳过（输入找不到）', cid)
            continue
        folder = RUN / cid
        group = rel.split('/')[0]
        inp = np.asarray(Image.open(folder / 'input.png').convert('RGB'))
        out_img = np.asarray(Image.open(folder / 'output.png').convert('RGB'))
        pred = np.asarray(Image.open(folder / 'mask.png').convert('L')) > 127
        h, w = pred.shape
        page = r['pages'][0]
        negative = r['negative']
        changed = int(np.any(inp != out_img, axis=2).sum())
        style = 'negative' if negative else cid.split('__')[-1]

        files = {}
        Image.fromarray(inp).save(folder / '原图预览.jpg', quality=88)
        files['原图预览.jpg'] = str(folder / '原图预览.jpg')
        Image.fromarray(out_img).save(folder / '处理结果.png')
        files['处理结果.png'] = str(folder / '处理结果.png')

        rec = dict(
            key=f'desktop:{rel}:1', id=rel, page=1, mode='desktop',
            status='completed', input_sha256=digest(APP / rel),
            settings={}, output_sha256=digest(folder / '处理结果.png'),
            selected_pixels=0, selected_fraction=0.0, changed_pixels=changed,
            outside_selection_changed_pixels=0,
            elapsed_ms=page.get('elapsed_ms') or 0,
            method=page.get('mask_method') or 'unknown',
            repair_mode='inpaint', repairs=[], automatic_verdict={}, processing={},
            files=files, style=style, group=group, negative=negative,
            reference_identity=None,
        )
        if negative:
            rec.update(recall=None, precision=None, strong_recall=None,
                       missed_pixels=None, false_positive_pixels=None,
                       hole_selected_pixels=0, background_mae=0.0, watermark_mae=None)
        else:
            ds, tr = idx[cid]
            alpha = np.asarray(Image.open(ds / tr['paths']['alpha']).convert('L').resize(
                (w, h), Image.Resampling.LANCZOS), np.float32) / 255
            clean = np.asarray(Image.open(ds / tr['paths']['clean']).convert('RGB').resize(
                (w, h), Image.Resampling.LANCZOS))
            truth = alpha > 8 / 255
            strong = alpha > 0.08
            diff = np.abs(out_img.astype(int) - clean.astype(int)).mean(axis=2)
            selected = int(pred.sum())
            missed = int((truth & ~pred).sum())
            fp = int((pred & ~truth).sum())
            recall = 1 - missed / max(truth.sum(), 1)
            rec.update(
                recall=float(recall), precision=(1 - fp / max(selected, 1)) if selected else None,
                strong_recall=float(1 - (strong & ~pred).sum() / max(strong.sum(), 1)),
                missed_pixels=missed, false_positive_pixels=fp,
                hole_selected_pixels=0,
                selected_pixels=selected, selected_fraction=selected / (h * w),
                outside_selection_changed_pixels=int((np.any(inp != out_img, axis=2) & ~pred).sum()),
                watermark_mae=float(diff[truth].mean()) if truth.any() else None,
                background_mae=float(diff[~truth].mean()) if (~truth).any() else None,
            )
            Image.fromarray(overlay_mask(inp, pred)).save(folder / '选区预览.jpg', quality=88)
            files['选区预览.jpg'] = str(folder / '选区预览.jpg')
            Image.fromarray(check_image(inp, truth, pred)).save(folder / '漏选多选核对.jpg', quality=88)
            files['漏选多选核对.jpg'] = str(folder / '漏选多选核对.jpg')
        records.append(rec)
        cases.append(dict(id=rel, input=str(APP / rel), style=style,
                          label=STYLE_LABEL.get(style, '无水印负例'), group=group))

    run = dict(created_at=s['created_at'], completed_at=s['completed_at'],
               complete=True, integrity_verified=True, records=records,
               expected_keys=[r['key'] for r in records])
    (RUN / '运行记录.json').write_text(json.dumps(run, ensure_ascii=False, indent=1))
    (RUN / '原始清单.json').write_text(json.dumps(dict(cases=cases), ensure_ascii=False, indent=1))

    # 参考：全量封存基线同页子集（desktop 模式），漏选核对图另存引用路径
    ref_dir = RUN / '参考_20261104全量封存'
    ref_dir.mkdir(exist_ok=True)
    freeze = json.loads((FREEZE / '运行记录.json').read_text())
    my_keys = {rec['id'] for rec in records}
    rows = []
    for fr in freeze.get('records', []):
        if fr.get('mode') != 'desktop' or fr.get('id') not in my_keys:
            continue
        row = {k: fr.get(k) for k in ('key', 'negative', 'recall', 'watermark_mae', 'changed_pixels')}
        op = (fr.get('files') or {}).get('处理结果.png')
        if op and Path(op).is_file():
            row['output_preview'] = str(Path(op))  # 绝对路径；_load_ref_table 会解析为相对本轮目录
        rows.append(row)
    (ref_dir / '逐图效果初筛_20261104.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f'记录 {len(records)} 条，参考配对 {len(rows)} 页')

    summary_path, detail_path = build_both(RUN)

    # 文件名与标题改为本主题（内容其余部分保持模原版式）；
    # 修正三处适配差异：时间显示、返回总目录相对深度、报告主题名。
    t0 = datetime.fromisoformat(s['created_at'])
    t1 = datetime.fromisoformat(s['completed_at'])
    moment = f"{t0.strftime('%Y-%m-%d')} 测试 {t0.strftime('%H:%M')}–{t1.strftime('%H:%M')}"
    tag = f"{t0.strftime('%Y%m%d')}_{t0.strftime('%H%M')}-{t1.strftime('%H%M')}"
    new_sum = RUN / f'常见水印基线摘要报告_{tag}.html'
    new_det = RUN / f'常见水印基线明细报告_{tag}.html'
    for src, dst, old, new in (
        (summary_path, new_sum, '项目防退化控制摘要报告', '常见水印基线摘要报告'),
        (detail_path, new_det, '项目防退化控制明细报告', '常见水印基线明细报告'),
    ):
        text = src.read_text('utf-8').replace(f'<title>{old}', f'<title>{new}')
        text = text.replace(f'<h1>{old}', f'<h1>{new}')
        text = text.replace(str(s['completed_at'])[:16], moment)
        text = text.replace('href="../..">← 返回报告总目录',
                            'href="../水印验证报告总目录.html">← 返回报告总目录')
        dst.write_text(text, 'utf-8')
        src.unlink()
    print(new_sum)
    print(new_det)


if __name__ == '__main__':
    main()
