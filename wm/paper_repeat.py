"""Remove repeated neutral oblique overlays on bright paper from one image.

Whole-word visual matches confirm OCR candidates; all matching copies are used.
Only confirmed strokes on paper are filled. Dark print, coloured ink and table
rules are preserved rather than handed to a generative repair model.
"""
from __future__ import annotations
import difflib
import warnings
import cv2
import numpy as np
from . import ocr, repeated
from .contract import MaskLayer, MaskSource


def _fit_columns(line_peaks, step_hint):
    """全局晶格拟合：列位置 = a_行 + b·j + d_行·(j 奇偶)，最小二乘一次拟合。

    水印是程序按固定网格排版的，位置是公式不是预测。逐行「行首 + 中位步长×序号」
    的算术外推会把量化噪声累积成行尾漂移；全局拟合用所有证据峰平摊，外推误差
    降到亚像素级。斜率 b 全行共享（同一晶格），相位 a 与奇偶项 d 每行独立
    （错行排版、各行 j 锚点奇偶不一致，共享奇偶项会互相抵消）。

    line_peaks: 每行的证据峰 x 列表（可为空列表）。
    返回 (params, b)：params[i] = (a, d) 或 None（证据不足的行）；
    拟合不成立返回 None（调用方回退旧逻辑）。
    """
    rows = []
    for li, xs in enumerate(line_peaks):
        if len(xs) < 2:
            continue
        xs = np.asarray(sorted(xs), float)
        rows.append((li, xs))
    if not rows:
        return None
    # 步长单位取「各行差值中位数的最小值」：晶格严格周期，差值是真步长的整数倍，
    # 只要有任一行检出了相邻实例就是真步长；疏排页 step_hint（验证峰中位步长）
    # 可能本身是 2 倍步长，直接用它编序号会把 b 拟成 2 倍（s014 实测）。
    unit = min(float(np.median(np.diff(xs))) for _, xs in rows)
    if not .5 * step_hint < unit < 2.0 * step_hint:
        return None
    rows = [(li, xs, np.concatenate([[0], np.cumsum(
        np.maximum(1, np.round(np.diff(xs) / unit)))]).astype(float)) for li, xs in rows]
    R = len(rows)
    n_obs = sum(len(r[1]) for r in rows)
    # 奇偶项每行多一个未知数：平均每行不足 3 个观测时 b 会不可辨识
    # （2 峰行被 a、d 精确吸收），宁可不要奇偶项。
    use_parity = n_obs >= 3 * R + 2
    n_unk = (2 * R if use_parity else R) + 1
    if n_obs < n_unk + 1:
        return None
    A = np.zeros((n_obs, n_unk))
    yv = np.zeros(n_obs)
    k = 0
    for rk, (li, xs, jj) in enumerate(rows):
        for x, j in zip(xs, jj):
            A[k, rk] = 1.0
            if use_parity:
                A[k, R + rk] = j % 2
            A[k, n_unk - 1] = j
            yv[k] = x
            k += 1
    sol, *_ = np.linalg.lstsq(A, yv, rcond=None)
    b = float(sol[n_unk - 1])
    if not .7 * unit < b < 1.4 * unit:
        return None
    # 晶格是严格周期排版，拟合好则残差仅 NCC 峰位量化噪声（±2px）；
    # 残差大说明 j 序号认错（误差是步长量级），宁可回退旧逻辑。
    if float(np.abs(A @ sol - yv).max()) > min(unit * .35, 6.0):
        return None
    params = [None] * len(line_peaks)
    for rk, (li, xs, jj) in enumerate(rows):
        params[li] = (float(sol[rk]), float(sol[R + rk]) if use_parity else 0.0)
    return params, b


def _fit_row_slope(line_peaks_xy):
    """全局行斜率拟合：去旋转角有残差时，行在去旋转系里仍带统一斜率（实测
    ~0.011 ≈ 0.63°，全行 y 落差 11-23px），常数行心盖章沿行越盖越歪（页边
    漏选真凶）。斜率全页共享（同一角度残差），截距每行独立。

    line_peaks_xy: 每行 [(x, y), ...]（可为空）。返回 (s, {行号: 截距}) 或 None。
    """
    rows = [(li, np.asarray(pts, float)) for li, pts in enumerate(line_peaks_xy)
            if len(pts) >= 2]
    if len(rows) < 2:
        return None
    n_obs = sum(len(p) for _, p in rows)
    if n_obs < len(rows) + 2:
        return None
    A = np.zeros((n_obs, len(rows) + 1))
    yv = np.zeros(n_obs)
    k = 0
    for rk, (li, pts) in enumerate(rows):
        for x, y in pts:
            A[k, rk] = 1.0
            A[k, len(rows)] = x
            yv[k] = y
            k += 1
    sol, *_ = np.linalg.lstsq(A, yv, rcond=None)
    s = float(sol[len(rows)])
    if abs(s) > .05:
        return None
    if float(np.abs(A @ sol - yv).max()) > 4.0:
        return None
    return s, {li: float(sol[rk]) for rk, (li, _) in enumerate(rows)}


def repeated_line_evidence(observations, minimum: int = 4) -> bool:
    """≥minimum 条近乎相同的整行长文本（≥5 字符）= 平铺水印签名。

    供路由在日期染色小行等早抢路径前判断（角度组 s002：水平水印行含日期，
    被 ocr_seed_stain 截胡）。纯文本判断，不做图像计算，代价可忽略。
    """
    groups = []
    for box, text, confidence in observations or []:
        height, length, _tilt = ocr._geometry(box)
        if confidence >= .65 and len(text) >= 5 and length > 3 * height:
            norm = ''.join(text.split())
            for grp in groups:
                if difflib.SequenceMatcher(None, norm, grp[0]).ratio() >= .8:
                    grp[1] += 1
                    break
            else:
                groups.append([norm, 1])
    return any(g[1] >= minimum for g in groups)


def table_rules(img):
    gray = img.astype(np.float32).mean(2)
    length = max(25, min(100, min(gray.shape)//12))
    ink = (gray < .65*float(np.percentile(gray, 97))).astype(np.uint8)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    # A dark watermark block is not a table rule. Keep only thin runs;
    # otherwise an invertible dark overlay would be copied back into the result.
    horizontal &= 1-cv2.morphologyEx(horizontal,cv2.MORPH_OPEN,np.ones((5,1),np.uint8))
    vertical &= 1-cv2.morphologyEx(vertical,cv2.MORPH_OPEN,np.ones((1,5),np.uint8))
    return cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8)) > 0


def estimate(img):
    gray = img.astype(np.float32).mean(2)
    if np.mean(gray > 235) < .70:
        return None
    observations = ocr.read_lines(img)
    img_lines = observations or []  # 原图 OCR 行留底，后面直立正文判别复用，不重复调用
    angles = []
    horiz = []
    wm_boxes = []   # 原图坐标下的水印行框：后面从 keepout 里减掉
    upright_body = False
    for box, text, confidence in observations or []:
        height, length, tilt = ocr._geometry(box)
        upright_body |= confidence>=.65 and len(text)>=4 and tilt<10
        v = box[1]-box[0]
        angle = float(np.degrees(np.arctan2(v[1], v[0])))
        if confidence >= .65 and len(text) >= 5 and length > 3*height:
            # 斜窗口下限 10°（原 15°，角度组 s000 实测 -15.0° 死在边界）：
            # 直立正文判别用 ≤2.5°、扫描微斜 <8°，10° 保持安全距离。
            if 10 < abs(angle) < 60:
                angles.append(angle)
                wm_boxes.append((angle, box))
            elif abs(angle) <= 10:
                horiz.append((angle, text, box))
    if not angles:
        # 水平/近水平重复水印（2026-11-27 角度组 s002）：与直立正文同角度，
        # 判别靠「同一字符串整行重复 ≥4 次」——正文行内容各不相同，表格短
        # 标签过不了 len≥5，页眉页脚页码逐页变化。下游晶格拟合、逐像素验证、
        # 占比闸门与斜向路径完全一致，不降低任何保护。
        groups = []
        for ang, text, box in horiz:
            norm = ''.join(text.split())
            for grp in groups:
                if difflib.SequenceMatcher(None, norm, grp[0]).ratio() >= .8:
                    grp[1].append(ang)
                    grp[2].append((ang, box))
                    break
            else:
                groups.append([norm, [ang], [(ang, box)]])
        best = max(groups, key=lambda g: len(g[1]), default=None)
        if best is None or len(best[1]) < 4:
            return None
        angles = best[1]
        wm_boxes = best[2]
    angle = float(np.median(angles))
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rw = int(abs(matrix[0, 0])*w+abs(matrix[0, 1])*h)
    rh = int(abs(matrix[0, 0])*h+abs(matrix[0, 1])*w)
    matrix[:, 2] += [(rw-w)/2, (rh-h)/2]
    rotated = cv2.warpAffine(img, matrix, (rw, rh), borderValue=(255, 255, 255))
    valid = cv2.warpAffine(np.ones((h, w), np.uint8), matrix, (rw, rh), flags=cv2.INTER_NEAREST)>0
    # Bring faint oblique lettering forward while suppressing dark body ink.
    g = rotated.astype(np.float32).mean(2)
    enhanced = 255-np.clip((255-g)*4,0,255)
    enhanced[g < 140] = 255
    observations = list(ocr.read_lines(rotated) or [])
    intact = sum(conf>=.65 and len(text)>=5 and ocr._geometry(box)[2]<6
                 and ocr._geometry(box)[1]>4*ocr._geometry(box)[0]
                 for box,text,conf in observations)
    sparse = intact < 4
    observations.extend(ocr.read_lines(np.repeat(enhanced.astype(np.uint8)[:,:,None],3,axis=2)) or [])
    seeds = []
    seed_texts = []
    for box, text, confidence in observations:
        height, length, tilt = ocr._geometry(box)
        if confidence >= (.45 if sparse else .65) and len(text) >= (3 if sparse else 5) and tilt < 6 and length > (2.0 if sparse else 4)*height:
            seeds.append((box, height, length))
            seed_texts.append(text)
    if not seeds:
        return None
    g = rotated.astype(np.float32).mean(2)
    size = max(7, int(np.median([s[1] for s in seeds])*.6)) | 1
    # Dark body type used to dominate the match score on dense paragraphs.
    # Estimate the repeated letter tone from OCR candidates and suppress darker
    # foreground for localization only; source pixels are never preprocessed.
    keepout = ocr.text_zones(img)
    if keepout is not None and wm_boxes and keepout.any():
        # keepout 本应只装正文，但 ≤15° 的水印行也会被 text_zones 当正文
        # （角度组 s000/s002 实测：keepout 墨 26-32% 全是水印自己，正文墨池
        # 为 0，双极拟合必死）。只剔 ≤15° 的水印行（更斜的本来就不在
        # text_zones 里，剔了会扰动文档底小字页的拟合，实测 s009/s010 召回
        # -0.1pp）；拷贝后再改不污染 OCR 缓存。
        keepout = keepout.copy()
        _kh, _kw = keepout.shape
        for _ang, _box in wm_boxes:
            if abs(_ang) > 15:
                continue
            _lo = np.maximum(np.floor(_box.min(0)).astype(int) - 4, 0)
            _hi = np.minimum(np.ceil(_box.max(0)).astype(int) + 4, [_kw, _kh])
            if _hi[0] > _lo[0] and _hi[1] > _lo[1]:
                keepout[_lo[1]:_hi[1], _lo[0]:_hi[0]] = False
    rotated_keepout = np.zeros(g.shape,bool) if keepout is None else cv2.warpAffine(
        keepout.astype(np.uint8),matrix,(rw,rh),flags=cv2.INTER_NEAREST)>0
    tones=[]
    neutral=rotated.max(2).astype(float)-rotated.min(2)<12
    for box,_,_ in seeds:
        lo=np.maximum(np.floor(box.min(0)).astype(int),0)
        hi=np.minimum(np.ceil(box.max(0)).astype(int),[rw,rh])
        y,x=slice(lo[1],hi[1]),slice(lo[0],hi[0])
        values=g[y,x]
        selected=(values>50)&(values<235)&~rotated_keepout[y,x]&neutral[y,x]
        tones.append(values[selected].ravel())
    if sum(len(v) for v in tones)<50:
        return None
    tone=float(np.percentile(np.concatenate(tones),10))
    feature = np.minimum(255-g, 1.5*(255-tone)).astype(np.float32)
    feature[g < (.94 if sparse else .7)*tone] = 0
    if tone < 140:
        feature = np.minimum(cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, np.ones((size,size),np.uint8)),150)
    options = []
    typical = float(np.median([s[2] for s in seeds]))
    for (box, height, length), token in zip(seeds,seed_texts):
        if not sparse and not .75*typical < length < 1.75*typical:
            continue
        x0,y0 = np.maximum(np.floor(box.min(0)).astype(int)-4, 0)
        x1,y1 = np.minimum(np.ceil(box.max(0)).astype(int)+4, [rw, rh])
        if y1-y0 < 8 or x1-x0 < 40:
            continue
        visible=valid[y0:y1,x0:x1]
        if visible.mean()<(.70 if sparse else .95) and not (sparse and np.count_nonzero(visible.mean(0)>.98)>=40):
            continue
        patch = feature[y0:y1, x0:x1]
        if patch.std() < 4:
            continue
        # Prefer a complete word away from underlying black text. OCR position
        # alone must never authorize erasing all pixels in its rectangle.
        black = float(np.mean(g[y0:y1, x0:x1] < 45))
        if not upright_body and black > .04:
            continue
        options.append((black, -length, x0, y0, x1, y1, token))
        # A boundary copy may expose a complete suffix while clipping its
        # prefix. Match the fully visible columns separately, without inventing
        # pixels outside the source page.
        if sparse and valid[y0:y1,x0:x1].mean()<.95:
            columns=np.flatnonzero(valid[y0:y1,x0:x1].mean(0)>.98)
            if columns.size>=40:
                lo,hi=x0+int(columns.min()),x0+int(columns.max())+1
                options.append((black-.001,-float(hi-lo),lo,y0,hi,y1,token+" visible"))
    chosen = None
    alternatives = []
    choices=[]
    seen=set()
    for option in sorted(options):
        token=''.join(c.lower() for c in option[-1] if c.isalnum())
        if sparse and token in seen:
            continue
        choices.append(option)
        seen.add(token)
    for _, _, x0,y0,x1,y1,_ in choices[:8]:
        patch = feature[y0:y1, x0:x1]
        th, tw = patch.shape
        scores = cv2.matchTemplate(feature, patch, cv2.TM_CCOEFF_NORMED)
        peaks = repeated._peaks(scores, patch.shape, .55, limit=256)
        if len(peaks) < 2:
            continue
        rank = len(peaks)*tw
        alternatives.append((rank,th,tw,peaks))
        if chosen is None or rank > chosen[0]:
            chosen = (rank, th, tw, peaks)
    if chosen is None:
        return None
    _, th, tw, peaks = chosen
    patches, luminance = [], []
    for x,y,_ in peaks:
        seen = valid[y:y+th,x:x+tw]
        patches.append(np.where(seen, feature[y:y+th,x:x+tw], np.nan))
        luminance.append(np.where(seen, g[y:y+th,x:x+tw], np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        recurring = np.nan_to_num(np.nanpercentile(patches, 25 if tone < 140 and len(peaks)>=3 else 60, axis=0))
        light = np.nan_to_num(np.nanpercentile(luminance, 90, axis=0), nan=255)
    threshold = max(7., .18*float(np.percentile(recurring, 98)))
    template = (recurring > threshold).astype(np.uint8)
    if template.sum() < 50 or not .02 < template.mean() < .5:
        return None
    field = np.zeros((rh+2*th, rw+2*tw), np.float32)
    valid = np.pad(valid, ((th,th),(tw,tw)))
    feature = np.pad(feature, ((th,th),(tw,tw)))
    peaks = [(x+tw,y+th,s) for x,y,s in peaks]
    matrix[:,2] += [tw,th]
    partial = repeated._partial_matches(feature, valid, template, peaks)
    for x,y,_ in [*peaks,*partial]:
        np.maximum(field[y:y+th,x:x+tw], template, out=field[y:y+th,x:x+tw])
    # OCR can split a sparse watermark into a Chinese prefix and an ASCII
    # suffix. Keep separately confirmed word parts instead of erasing only the
    # most frequently matched part. Dense grids already supply enough samples.
    if sparse:
        extra=np.zeros((rh,rw),np.float32)
        for _,eh,ew,ep in sorted(alternatives,key=lambda a:a[0],reverse=True)[:6]:
            if sum(v>.65 for _,_,v in ep)<2:
                continue
            stack=[np.where(valid[y+th:y+th+eh,x+tw:x+tw+ew],
                     feature[y+th:y+th+eh,x+tw:x+tw+ew],np.nan) for x,y,_ in ep]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                recurring_part=np.nan_to_num(np.nanpercentile(stack,60,axis=0))
            part=(recurring_part>max(7.,.18*float(np.percentile(recurring_part,98)))).astype(np.float32)
            if not .02<part.mean()<.5:
                continue
            for x,y,_ in ep:
                np.maximum(extra[y:y+eh,x:x+ew],part,out=extra[y:y+eh,x:x+ew])
        np.maximum(field[th:th+rh,tw:tw+rw],extra,out=field[th:th+rh,tw:tw+rw])
    # keepout 已在取色阶段剔过水印行并缓存于局部变量，直接复用
    # 直立正文行 ≥2 才把 keepout 接进反解（有正文页字上水印靠它保护）；
    # 白底页的水印行会被 OCR 误当正文（tilt≤15° 的口径太宽），接进去会把
    # 晕圈 alpha 归零、召回实测掉 0.6pp——无直立正文时传 None，由反解内部
    # 按原逻辑自取（与基线行为一致）。
    _upright = 0
    for _box, _t, _c in img_lines:
        _h, _l, _tilt = ocr._geometry(_box)
        if _tilt <= 2.5 and _l >= 3 * _h:
            _upright += 1
    protected = None if keepout is None else cv2.warpAffine(keepout.astype(np.uint8), matrix,
                    (valid.shape[1],valid.shape[0]),flags=cv2.INTER_NEAREST)>0
    # 页边加宽碎片匹配已退役：晶格行内证据检测直接找到页边实例，不再需要
    # 放宽碎片带（实测宽带会让无验证的纸面直填路在白底小字页多选爆 4 倍）。
    for x,y,tx,ty,patch in repeated._edge_fragments(feature, valid, template, peaks, protected):
        ph,pw=patch.shape
        np.maximum(field[y:y+ph,x:x+pw], patch, out=field[y:y+ph,x:x+pw])
    # 晶格全铺 + 行内证据检测（2026-11-25 实验定案，替换行格外推 ±1 档与
    # 页边加宽膨胀/预算回退补丁）：晶格（真实行心 + 端部行距外推）给出全部
    # 可能章位只负责"提问"；沿每条行带把 NCC 门槛降到 0.30 重新找章——页边
    # 裁切实例 NCC ~0.42，全局 0.55 门槛抓不到，行带约束本身杀假峰，章位
    # 全部来自证据；无证据的外推位也盖章兜底（页角全裁切实例），真假由
    # 逐像素验证裁决。晶格章模板膨胀 1px 盖住抗锯齿晕圈（实验实测：漏选
    # 71% 是模板外晕圈；文档底 11 张改善、效果好 6 张零退化）。
    row_groups = []
    for px, py, _s in sorted(peaks, key=lambda p: p[1]):
        for grp in row_groups:
            if abs(float(np.median([gy for _, gy in grp])) - py) < th * 1.5:
                grp.append((px, py))
                break
        else:
            row_groups.append([(px, py)])
    per_row = []
    for grp in row_groups:
        xs = sorted(x for x, _ in grp)
        d = np.diff(xs)
        per_row.append((float(np.median([gy for _, gy in grp])), xs[0],
                        float(np.median(d)) if len(d) and np.median(d) >= tw * .8 else tw * 1.05))
    per_row.sort()
    field *= valid
    # 纸面直填（尾部单章路）不做逐像素验证，它的风险就是模板占位本身：
    # 必须用未并晶格的保守占位（实测白底小字页被晶格撑爆多选 49 万px）。
    coverage_peaks = cv2.GaussianBlur(cv2.warpAffine(field, cv2.invertAffineTransform(matrix), (w,h)), (0, 0), 1.2)
    region_tail = cv2.dilate((coverage_peaks > .03).astype(np.uint8), np.ones((7,7), np.uint8)) > 0
    if len(per_row) >= 2:
        fh, fw = field.shape
        # 真实行距可交替不均（实测奇偶行差 6-8px）：检测行用真实行心，
        # 端部外侧按最近行距外推，不设档位限制。
        lines = list(per_row)
        top_gap = per_row[1][0] - per_row[0][0]
        y = per_row[0][0] - top_gap
        while y > -th:
            lines.append((y, per_row[0][1], per_row[0][2]))
            y -= top_gap
        bot_gap = per_row[-1][0] - per_row[-2][0]
        y = per_row[-1][0] + bot_gap
        while y < fh:
            lines.append((y, per_row[-1][1], per_row[-1][2]))
            y += bot_gap
        lattice = np.zeros_like(field)
        tf32 = cv2.dilate(template, np.ones((3, 3), np.uint8)).astype(np.float32)
        row_scores = cv2.matchTemplate(feature, template.astype(np.float32),
                                       cv2.TM_CCOEFF_NORMED)
        sh = row_scores.shape[0]

        def _stamp_at(xi, yi):
            xs0, ys0 = max(0, xi), max(0, yi)
            sx0, sy0 = xs0 - xi, ys0 - yi
            hh = min(th - sy0, fh - ys0)
            ww = min(tw - sx0, fw - xs0)
            if hh > 0 and ww > 0:
                np.maximum(lattice[ys0:ys0 + hh, xs0:xs0 + ww],
                           tf32[sy0:sy0 + hh, sx0:sx0 + ww],
                           out=lattice[ys0:ys0 + hh, xs0:xs0 + ww])

        line_infos = []
        for y_line, x0_row, step_x in lines:
            yi = int(round(y_line))
            sy0 = max(0, min(sh - 1, yi - 12))
            sy1 = max(0, min(sh, yi + 13))
            taken = []
            if sy1 > sy0:
                band = row_scores[sy0:sy1]
                prof = band.max(0)
                cand = np.flatnonzero(prof >= 0.30)
                order = cand[np.argsort(prof[cand])[::-1]] if len(cand) else []
                for c in order:
                    if all(abs(int(c) - t[0]) > step_x * .5 for t in taken):
                        ypk = sy0 + int(np.argmax(band[:, c]))
                        taken.append((int(c), ypk))
            line_infos.append([yi, x0_row, step_x, taken])
        # 全局行斜率（2026-11-27，页边漏选真凶）：去旋转角有残差（实测 ~0.63°）
        # 时行仍是斜的，全行 y 落差 11-23px——常数行心盖章沿行越盖越歪，行尾
        # 偏出 ~10px 盖进字缝被验证处决。斜率全页共享一次拟合；拟合成立后按
        # 斜行带重检（平带 ±12px 装不下 23px 落差，行尾实例原本检不出），
        # 证据峰变多再反哺列拟合。
        slope_fit = _fit_row_slope([t for _, _, _, t in line_infos])
        if slope_fit is not None:
            s_sl, c_map = slope_fit
            rw = row_scores.shape[1]
            xs_all = np.arange(rw)
            for li, info in enumerate(line_infos):
                yi, x0_row, step_x, taken = info
                c0 = c_map.get(li)
                base = (c0 + s_sl * xs_all) if c0 is not None else (yi + s_sl * (xs_all - fw / 2))
                centers = np.round(base).astype(int)
                prof = np.zeros(rw, np.float32)
                for off in range(-12, 13):
                    prof = np.maximum(prof, row_scores[np.clip(centers + off, 0, sh - 1), xs_all])
                cand = np.flatnonzero(prof >= 0.30)
                order = cand[np.argsort(prof[cand])[::-1]] if len(cand) else []
                taken2 = []
                for c in order:
                    if all(abs(int(c) - t[0]) > step_x * .5 for t in taken2):
                        cy = int(centers[c])
                        y0w = max(0, min(sh - 1, cy - 12))
                        y1w = max(0, min(sh, cy + 13))
                        ypk = y0w + int(np.argmax(row_scores[y0w:y1w, c])) if y1w > y0w else cy
                        taken2.append((int(c), ypk))
                info[3] = taken2
        for yi, x0_row, step_x, taken in line_infos:
            for c, ypk in taken:
                _stamp_at(c, ypk)
        # 全局列拟合：列位置 = a_行 + b·j + d_行·(j 奇偶)，替换「行首+中位步长
        # ×序号」算术外推（量化噪声累积成行尾漂移）。拟合不成立的行回退旧逻辑。
        steps = [s for _, _, s, t in line_infos if len(t) >= 2] or [s for _, _, s, _ in line_infos]
        fit = _fit_columns([[x for x, _ in t] for _, _, _, t in line_infos], float(np.median(steps)))
        for li, (yi, x0_row, step_x, taken) in enumerate(line_infos):
            params = fit[0][li] if fit is not None else None
            if params is not None:
                # 本行有证据拟合：位置由公式给出，精确到亚像素
                a_row, d_par = params
                b_col = fit[1]
                j_lo = int(np.floor((0 - a_row) / b_col)) - 2
                j_hi = int(np.ceil((fw - a_row) / b_col)) + 2
                positions = [int(round(a_row + j * b_col + d_par * (j % 2)))
                             for j in range(j_lo, j_hi + 1)]
            else:
                # 本行无证据（端部外推行等）：不猜相位，直接密铺地毯——章宽大于
                # 间隔保证整行无缝覆盖，真假全部由逐像素验证裁决（s014 实测：
                # 借相位会带 8-17px 行际误差被验证处决，密铺反而零漏选）。
                dense = max(1, int(tw * .6))
                positions = list(range(-tw, fw + tw, dense))
            # 外推章的行坐标同样按斜行公式，不再用常数行心
            if slope_fit is not None:
                s2, c_map2 = slope_fit
                c0 = c_map2.get(li)
                y_of = (lambda xx, c0=c0: c0 + s2 * xx) if c0 is not None \
                    else (lambda xx, yi=yi: yi + s2 * (xx - fw / 2))
            else:
                y_of = lambda xx, yi=yi: yi
            for xi in positions:
                if xi + tw < 0 or xi > fw:
                    continue
                if all(abs(xi - t[0]) > step_x * .5 for t in taken):
                    _stamp_at(xi, int(round(y_of(xi))))
        field = np.maximum(field, lattice)
        field *= valid
    coverage = cv2.GaussianBlur(cv2.warpAffine(field, cv2.invertAffineTransform(matrix), (w,h)), (0, 0), 1.2)
    # Soft coverage keeps antialiased stroke edges inside the invertible fit:
    # a hard binary template zeroes alpha exactly where the veil feathers out,
    # which shows up as residual watermark at the glyph rims. 印章模板只抓住笔画
    # 深色核心，真实抗锯齿晕圈在模板外约 1–2px；把选区再外扩 1px（总计约
    # 3px），让纸面直填覆盖整条晕圈，避免清完内部后留下空心轮廓。
    region = cv2.dilate((coverage > .03).astype(np.uint8), np.ones((7,7), np.uint8)) > 0
    # 预闸门只防失控：晶格全铺的"提问音量"可达 16%（小字密页实测），真正的
    # 占比闸门卡在验证后的实际声称面积（见下方 fitted 检查，同一 .15 红线）。
    if not .0001 < region.mean() < .25:
        return None
    samples = img[region & (gray>50)]
    if len(samples) < 50:
        return None
    # Prefer an invertible alpha-compositing fit over paper-only cleanup:
    # cleaning just the paper leaves watermark strokes where they cross body
    # text, which is exactly what the recognition condition charges against.
    # The repeated instances supply the covered-paper profile the fit needs.
    # 被盖纸面剖面：只盖真实峰（补章是外推，不拿它估纸面亮度），供反解用。
    light_field = np.full(valid.shape, 255, np.float32)
    for x, y, _ in peaks:
        light_field[y:y+th, x:x+tw] = np.minimum(
            light_field[y:y+th, x:x+tw], np.where(template > 0, light, 255))
    profile = cv2.warpAffine(light_field, cv2.invertAffineTransform(matrix), (w, h),
                             borderValue=255)
    fitted = repeated._document_overlay(img, region, profile,
                                        float(np.median([p[2] for p in peaks])),
                                        allow_broad_ink=True, coverage_hint=coverage,
                                        allow_single_pole=True,
                                        keepout_hint=keepout if _upright >= 2 else None)
    if fitted is not None:
        # 占比闸门卡在验证后的实际声称面积（原 region 预闸门的正确位置）：
        # 晶格全铺会把“提问”铺到 16%，但验证后声称才是真实风险（实测 4-6%）。
        if (fitted.alpha >= .08).mean() >= .15:
            return None
        fitted.method = 'paper_repeat_document_overlay'
        return fitted
    # This path handles neutral lettering only. Coloured stamps have their own
    # measured matting path and must retain it.
    # 无逐像素验证的纸面直填：占位闸门维持旧红线 .15（晶格全铺不适用此路）。
    if not .0001 < region_tail.mean() < .15:
        return None
    eligible = region_tail & (gray>50) & (gray<245)
    if keepout is not None and np.count_nonzero(eligible & ~keepout)>50:
        eligible &= ~keepout
    bins = np.arange(0,258,2)
    histogram,_ = np.histogram(gray[eligible],bins=bins)
    if histogram.max()<20:
        return None
    level = float(bins[histogram.argmax()]+1)
    core = eligible & (np.abs(gray-level)<3)
    color = np.median(img[core],axis=0).astype(np.float32)
    if color.max()-color.min() > 12:
        return None
    paper = np.percentile(img.reshape(-1,3),97,axis=0).astype(np.float32)
    return MaskLayer(alpha=region_tail.astype(np.float32),source=MaskSource.PRIOR,
        confidence=float(np.median([p[2] for p in peaks])),
        veil_strength=np.zeros(3,np.float32),veil_offset=color,
        paper=np.broadcast_to(paper,img.shape).copy(),footprint=region_tail,
        method='single_repeated_paper',repair_mode='paper_repeat_cleanup',text_keepout=keepout)


def restore(img, mask):
    gray = img.astype(np.float32).mean(2)
    covered = float(mask.veil_offset.mean())
    # Real dark strokes are darker than the independently repeated paper glyph.
    # Include their antialiasing rim and coloured links/seals in the keep-out.
    ink = gray < max(35., .72*covered)
    if mask.text_keepout is not None:
        ink |= mask.text_keepout & (gray < .94*covered)
    ink |= img.max(2).astype(float)-img.min(2) > 18
    protected = cv2.dilate(ink.astype(np.uint8), np.ones((3,3),np.uint8))>0
    protected |= table_rules(img)
    region = mask.footprint & ~protected
    output = img.copy()
    output[region] = np.clip(mask.paper[region],0,255).astype(np.uint8)
    return output
